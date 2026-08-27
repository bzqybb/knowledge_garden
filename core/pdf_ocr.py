from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

from core.config import RUNTIME_DIR
from core.storage import GardenStore


OCR_SCRIPT = Path(__file__).with_name("windows_pdf_ocr.ps1")
OCR_STATE_KEY = "textbook_ocr_state_v1"
OCR_JOB_KEY = "textbook_ocr_job_v1"
MIN_TEXT_LENGTH = 40
SURROGATES = re.compile(r"[\ud800-\udfff]")
CJK_SPACES = re.compile(r"(?<=[\u3400-\u9fff])[ \t]+(?=[\u3400-\u9fff])")


def clean_pdf_text(text: str, *, from_ocr: bool = False) -> str:
    """Remove malformed PDF Unicode and repair spaced Chinese OCR tokens."""
    cleaned = SURROGATES.sub("", str(text or ""))
    if from_ocr:
        cleaned = CJK_SPACES.sub("", cleaned)
    return cleaned.strip()


def windows_powershell_path() -> Path:
    system_root = Path(os.getenv("SystemRoot", r"C:\Windows"))
    return system_root / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"


def windows_ocr_available() -> bool:
    return os.name == "nt" and windows_powershell_path().is_file() and OCR_SCRIPT.is_file()


def _process_is_running(pid: int) -> bool:
    if pid <= 0 or os.name != "nt":
        return False
    import ctypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        status = ctypes.c_uint32()
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(status))) and status.value == 259
    finally:
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel.CloseHandle(handle)


def textbook_ocr_status(store: GardenStore) -> dict[str, Any]:
    job = dict(store.setting(OCR_JOB_KEY, {}) or {})
    states = dict(store.setting(OCR_STATE_KEY, {}) or {})
    books = [
        {
            "title": Path(path).stem,
            "page": int(entry.get("last_page", 0)),
            "pages": int(entry.get("page_count", 0)),
            "indexed_pages": int(entry.get("indexed_pages", 0)),
            "completed": bool(entry.get("completed")),
        }
        for path, entry in states.items() if isinstance(entry, dict)
    ]
    active = next((item for item in reversed(books) if not item["completed"]), None)
    pid = int(job.get("pid", 0) or 0)
    return {
        "available": windows_ocr_available(),
        "running": _process_is_running(pid),
        "pid": pid,
        "books": books,
        "current": active,
        "indexed_pages": sum(item["indexed_pages"] for item in books),
        "completed_books": sum(item["completed"] for item in books),
        "provider": "windows-local-ocr",
    }


def start_background_textbook_ocr(
    store: GardenStore, directory: str | Path, *, includes: tuple[str, ...] = (),
) -> dict[str, Any]:
    status = textbook_ocr_status(store)
    if status["running"]:
        return status
    if not status["available"]:
        raise RuntimeError("当前 Windows 用户没有可用的本地 OCR")
    candidates = discover_scanned_textbooks(store, directory, includes=includes)
    if not candidates:
        return {**status, "queued_books": 0}
    command = [
        sys.executable, "-X", "utf8", "-m", "evals.ocr_textbooks",
        "--directory", str(Path(directory).resolve()), "--rebuild-semantic-index",
    ]
    if includes:
        command.extend(["--books", ",".join(includes)])
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    output = RUNTIME_DIR / "textbook-ocr.log"
    errors = RUNTIME_DIR / "textbook-ocr-error.log"
    with output.open("a", encoding="utf-8") as stdout, errors.open("a", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parent.parent),
            stdout=stdout,
            stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    store.set_setting(OCR_JOB_KEY, {
        "pid": process.pid,
        "directory": str(Path(directory).resolve()),
        "queued_books": len(candidates),
        "titles": [item["title"] for item in candidates],
    })
    return {**textbook_ocr_status(store), "queued_books": len(candidates)}


def pdf_fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def iter_windows_pdf_ocr(
    pdf_path: str | Path, *, start_page: int = 1, end_page: int = 0,
    target_width: int = 1800,
) -> Iterator[dict[str, Any]]:
    """Stream page results from the built-in offline Windows Chinese OCR engine."""
    if not windows_ocr_available():
        raise RuntimeError("当前电脑没有可用的 Windows 本地 PDF/OCR 环境")
    command = [
        str(windows_powershell_path()), "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", str(OCR_SCRIPT), "-PdfPath", str(Path(pdf_path).resolve()),
        "-StartPage", str(max(1, start_page)), "-EndPage", str(max(0, end_page)),
        "-TargetWidth", str(max(900, min(2600, target_width))),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert process.stdout is not None
    try:
        for raw in process.stdout:
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload
        stderr = process.stderr.read() if process.stderr is not None else ""
        exit_code = process.wait()
        if exit_code:
            raise RuntimeError(f"Windows OCR 进程退出：{stderr.strip()[:320] or exit_code}")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def discover_scanned_textbooks(
    store: GardenStore, directory: str | Path, *, includes: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Find image-only PDFs without mistaking normal title pages for scanned books."""
    from pypdf import PdfReader

    root = Path(directory).expanduser().resolve()
    state = store.setting(OCR_STATE_KEY, {}) or {}
    note_counts: dict[str, int] = {}
    for note in store.list_notes(limit=50_000):
        if note.get("kind") == "textbook":
            source = str(note.get("source_url") or "")
            note_counts[source] = note_counts.get(source, 0) + 1
    selected = tuple(part.casefold() for part in includes if str(part).strip())
    candidates: list[dict[str, Any]] = []
    for path in root.rglob("*.pdf"):
        if selected and not any(part in path.name.casefold() for part in selected):
            continue
        try:
            reader = PdfReader(str(path))
            page_count = len(reader.pages)
            if page_count < 4:
                continue
            fingerprint = pdf_fingerprint(path)
            entry = state.get(str(path.resolve()), {})
            if entry.get("fingerprint") == fingerprint and entry.get("completed"):
                continue
            sample_indices = sorted({
                0, page_count // 4, page_count // 2,
                page_count * 3 // 4, page_count - 1,
            })
            readable_samples = sum(
                len(clean_pdf_text(reader.pages[index].extract_text() or "")) >= 80
                for index in sample_indices
            )
            indexed_pages = note_counts.get(str(path.resolve()), 0)
            if readable_samples > max(1, len(sample_indices) // 4):
                continue
            if indexed_pages >= max(4, int(page_count * 0.7)):
                continue
            last_page = int(entry.get("last_page", 0)) if entry.get("fingerprint") == fingerprint else 0
            candidates.append({
                "path": str(path.resolve()), "title": path.stem,
                "pages": page_count, "indexed_pages": indexed_pages,
                "readable_samples": readable_samples,
                "last_page": last_page, "remaining_pages": max(0, page_count - last_page),
            })
        except Exception as exc:
            candidates.append({
                "path": str(path.resolve()), "title": path.stem, "pages": 0,
                "indexed_pages": note_counts.get(str(path.resolve()), 0),
                "last_page": 0, "remaining_pages": 0,
                "error": f"{exc.__class__.__name__}: {str(exc)[:160]}",
            })
    foundation_priority = ("普通化学", "高等代数", "复变函数")
    candidates.sort(key=lambda item: (
        next((index for index, marker in enumerate(foundation_priority) if marker in item["title"]), 9),
        item["title"],
    ))
    return candidates


def ocr_textbook_into_store(
    store: GardenStore,
    pdf_path: str | Path,
    *,
    max_pages: int = 0,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Recognize a scanned PDF locally and persist resumable page checkpoints."""
    path = Path(pdf_path).expanduser().resolve()
    state = dict(store.setting(OCR_STATE_KEY, {}) or {})
    key = str(path)
    fingerprint = pdf_fingerprint(path)
    current = dict(state.get(key, {}))
    if current.get("fingerprint") != fingerprint:
        current = {"fingerprint": fingerprint, "last_page": 0, "indexed_pages": 0}
    if current.get("completed"):
        return {"path": key, "title": path.stem, "skipped": True, **current}
    start_page = max(1, int(current.get("last_page", 0)) + 1)
    end_page = start_page + max_pages - 1 if max_pages > 0 else 0
    processed = indexed = errors = 0
    page_count = int(current.get("page_count", 0))
    try:
        for result in iter_windows_pdf_ocr(path, start_page=start_page, end_page=end_page):
            page_number = int(result.get("page", 0))
            page_count = int(result.get("page_count", page_count))
            processed += 1
            if result.get("error"):
                errors += 1
                current["last_error"] = str(result["error"])[:240]
            else:
                text = clean_pdf_text(str(result.get("text") or ""), from_ocr=True)
                if len(text) >= MIN_TEXT_LENGTH:
                    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    store.upsert_note({
                        "path": f"pdf::{path}#page={page_number}",
                        "title": f"{path.stem} · 第 {page_number} 页",
                        "kind": "textbook", "content": text,
                        "tags": ["教材", path.stem, "本地OCR"],
                        "source": "pdf", "source_url": str(path),
                        "content_hash": digest,
                    })
                    indexed += 1
            current.update({
                "fingerprint": fingerprint, "last_page": page_number,
                "page_count": page_count,
                "indexed_pages": int(current.get("indexed_pages", 0)) + int(not result.get("error") and len(clean_pdf_text(str(result.get("text") or ""), from_ocr=True)) >= MIN_TEXT_LENGTH),
                "completed": bool(page_count and page_number >= page_count),
                "provider": "windows-local-ocr", "language": result.get("language", "zh-Hans-CN"),
            })
            if page_number % 5 == 0 or current["completed"]:
                state[key] = dict(current)
                store.set_setting(OCR_STATE_KEY, state)
            if progress:
                progress({
                    "title": path.stem, "path": key,
                    "page": page_number, "pages": page_count,
                    "indexed_in_run": indexed, "processed_in_run": processed,
                    "errors": errors,
                })
    finally:
        state[key] = dict(current)
        store.set_setting(OCR_STATE_KEY, state)
    return {
        "path": key, "title": path.stem, "processed": processed,
        "indexed": indexed, "errors": errors, "page_count": page_count,
        "start_page": start_page, "last_page": int(current.get("last_page", 0)),
        "completed": bool(current.get("completed")), "provider": "windows-local-ocr",
    }
