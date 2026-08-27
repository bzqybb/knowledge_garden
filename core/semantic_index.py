from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from core.config import RUNTIME_DIR
from core.storage import GardenStore


INDEX_DIR = RUNTIME_DIR / "semantic_faiss"
INDEX_FILE = INDEX_DIR / "index.faiss"
METADATA_FILE = INDEX_DIR / "metadata.json"
DEFAULT_MODEL = "intfloat/multilingual-e5-small"
INDEX_SCHEMA_VERSION = 3
INDEX_KINDS = {"concept", "moc", "bridge", "knowledge", "course", "textbook"}
_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {}
PDF_PAGE_PATH_RE = re.compile(r"^(pdf::.+#page=)(\d+)$")


def _adjacent_pdf_paths(path: str) -> list[str]:
    """Return immediate page neighbours without crossing document boundaries."""
    match = PDF_PAGE_PATH_RE.match(str(path or ""))
    if not match:
        return []
    prefix, raw_page = match.groups()
    page = int(raw_page)
    neighbours = []
    if page > 1:
        neighbours.append(f"{prefix}{page - 1}")
    neighbours.append(f"{prefix}{page + 1}")
    return neighbours


def _model_name() -> str:
    return os.getenv("GARDEN_EMBEDDING_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _document_input(text: str, model_name: str) -> str:
    return f"passage: {text}" if "e5" in model_name.lower() else text


def _query_input(text: str, model_name: str) -> str:
    return f"query: {text}" if "e5" in model_name.lower() else text


def _chunks(text: str, *, size: int = 900, overlap: int = 140) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    chunks = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + size)
        if end < len(clean):
            boundary = max(clean.rfind(". ", start, end), clean.rfind("。", start, end), clean.rfind("; ", start, end))
            if boundary > start + size // 2:
                end = boundary + 1
        chunks.append(clean[start:end])
        if end >= len(clean):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _expand_record_window(
    records: list[dict[str, Any]], record: dict[str, Any], *, before: int = 3, after: int = 0,
) -> tuple[str, list[int]]:
    """Return the matched chunk with adjacent chunks from the same source.

    Embedding search should stay chunk-sized, while evidence review needs the
    surrounding textbook argument. This prevents a page from being found via
    a figure/caption chunk while the defining sentence immediately before it
    is silently discarded.
    """
    path = str(record.get("path") or "")
    center = int(record.get("chunk_index", 0) or 0)
    lower = center - max(0, before)
    upper = center + max(0, after)
    window = sorted(
        (
            item for item in records
            if str(item.get("path") or "") == path
            and lower <= int(item.get("chunk_index", 0) or 0) <= upper
        ),
        key=lambda item: int(item.get("chunk_index", 0) or 0),
    )
    indices = [int(item.get("chunk_index", 0) or 0) for item in window]
    return "\n".join(str(item.get("text") or "").strip() for item in window if str(item.get("text") or "").strip()), indices


def _signature(notes: list[dict[str, Any]], model_name: str) -> str:
    digest = hashlib.sha256(model_name.encode("utf-8"))
    for note in sorted(notes, key=lambda item: str(item["path"])):
        digest.update(str(note["path"]).encode("utf-8"))
        content_hash = str(note.get("content_hash") or hashlib.sha256(str(note.get("content", "")).encode("utf-8")).hexdigest())
        digest.update(content_hash.encode("ascii"))
    return digest.hexdigest()


def _content_hash(note: dict[str, Any]) -> str:
    return str(
        note.get("content_hash")
        or hashlib.sha256(str(note.get("content", "")).encode("utf-8")).hexdigest()
    )


def _existing_vectors(model_name: str, notes: list[dict[str, Any]]) -> tuple[dict[str, list[tuple[dict[str, Any], Any]]], dict[str, str]]:
    """Load reusable vectors without loading the embedding model.

    The signature fallback upgrades the pre-incremental index safely when the
    complete note collection is unchanged.
    """
    if not INDEX_FILE.is_file() or not METADATA_FILE.is_file():
        return {}, {}
    try:
        import faiss
        import numpy as np

        metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
        if str(metadata.get("model")) != model_name or metadata.get("schema_version") != INDEX_SCHEMA_VERSION:
            return {}, {}
        index = faiss.deserialize_index(np.frombuffer(INDEX_FILE.read_bytes(), dtype="uint8"))
        records = list(metadata.get("records", []))
        if index.ntotal != len(records):
            return {}, {}
        note_hashes = {str(key): str(value) for key, value in metadata.get("note_hashes", {}).items()}
        if not note_hashes and metadata.get("signature") == _signature(notes, model_name):
            note_hashes = {str(note["path"]): _content_hash(note) for note in notes}
        grouped: dict[str, list[tuple[dict[str, Any], Any]]] = {}
        for position, record in enumerate(records):
            grouped.setdefault(str(record["path"]), []).append((record, index.reconstruct(position)))
        return grouped, note_hashes
    except Exception:
        return {}, {}


def build_semantic_index(
    store: GardenStore, *, kinds: set[str] | None = None, force_rebuild: bool = False,
) -> dict[str, Any]:
    import faiss
    import numpy as np

    allowed = kinds or INDEX_KINDS
    notes = [note for note in store.list_notes(limit=10_000) if note["kind"] in allowed]
    model_name = _model_name()
    signature = _signature(notes, model_name)
    if not force_rebuild and METADATA_FILE.is_file() and INDEX_FILE.is_file():
        current_metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
        if (
            current_metadata.get("schema_version") == INDEX_SCHEMA_VERSION
            and current_metadata.get("model") == model_name
            and current_metadata.get("signature") == signature
        ):
            return {
                "model": model_name, "notes": len(notes),
                "chunks": int(current_metadata.get("chunk_count", 0)),
                "reused_notes": len(notes), "encoded_notes": 0,
                "removed_notes": 0, "encoded_chunks": 0,
                "unchanged": True, "path": str(INDEX_DIR),
            }
    old_by_path, old_hashes = ({}, {}) if force_rebuild else _existing_vectors(model_name, notes)
    records: list[dict[str, Any]] = []
    texts: list[str] = []
    ordered_vectors: list[Any | None] = []
    encoded_record_positions: list[int] = []
    reused_notes = 0
    encoded_notes = 0
    for note in notes:
        path = str(note["path"])
        if old_hashes.get(path) == _content_hash(note) and old_by_path.get(path):
            reused_notes += 1
            for old_record, vector in old_by_path[path]:
                records.append(old_record)
                ordered_vectors.append(vector)
            continue
        encoded_notes += 1
        for chunk_index, chunk in enumerate(_chunks(str(note.get("content", "")))):
            records.append({
                "path": path, "title": str(note["title"]),
                "kind": str(note["kind"]), "chunk_index": chunk_index, "text": chunk,
            })
            ordered_vectors.append(None)
            encoded_record_positions.append(len(records) - 1)
            texts.append(_document_input(f"{note['title']}\n{chunk}", model_name))
    if not records:
        raise RuntimeError("没有可建立语义索引的文本")
    encoded_vectors: list[Any] = []
    if texts:
        from sentence_transformers import SentenceTransformer

        print(f"Loading embedding model: {model_name}", flush=True)
        model = SentenceTransformer(model_name, device="cpu", local_files_only=True)
        print(f"Encoding {len(texts)} changed chunks from {encoded_notes} notes...", flush=True)
        encoded_vectors = list(model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True))
    for position, vector in zip(encoded_record_positions, encoded_vectors):
        ordered_vectors[position] = vector
    if any(vector is None for vector in ordered_vectors):
        raise RuntimeError("语义索引向量与元数据未能一一对应")
    matrix = np.asarray(ordered_vectors, dtype="float32")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    temporary_index = INDEX_FILE.with_suffix(".faiss.tmp")
    temporary_metadata = METADATA_FILE.with_suffix(".json.tmp")
    # FAISS' Windows C++ file API cannot reliably open paths containing
    # non-ASCII characters. Serialize in memory and let Python handle the path.
    temporary_index.write_bytes(faiss.serialize_index(index).tobytes())
    temporary_metadata.write_text(json.dumps({
        "schema_version": INDEX_SCHEMA_VERSION,
        "model": model_name, "signature": signature,
        "note_hashes": {str(note["path"]): _content_hash(note) for note in notes},
        "note_count": len(notes), "chunk_count": len(records), "records": records,
    }, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary_index, INDEX_FILE)
    os.replace(temporary_metadata, METADATA_FILE)
    with _LOCK:
        _CACHE.clear()
    removed_notes = len(set(old_hashes) - {str(note["path"]) for note in notes})
    return {
        "model": model_name, "notes": len(notes), "chunks": len(records),
        "reused_notes": reused_notes, "encoded_notes": encoded_notes,
        "removed_notes": removed_notes, "encoded_chunks": len(texts),
        "path": str(INDEX_DIR),
    }


def _load() -> tuple[Any, Any, dict[str, Any]] | None:
    if not INDEX_FILE.is_file() or not METADATA_FILE.is_file():
        return None
    cache_key = f"{INDEX_FILE.stat().st_mtime_ns}:{METADATA_FILE.stat().st_mtime_ns}"
    with _LOCK:
        if _CACHE.get("key") == cache_key:
            return _CACHE["index"], _CACHE["model"], _CACHE["metadata"]
        import faiss
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer

        from core.inference_runtime import configure_local_inference

        configure_local_inference(torch)
        metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
        index_bytes = np.frombuffer(INDEX_FILE.read_bytes(), dtype="uint8")
        index = faiss.deserialize_index(index_bytes)
        model = SentenceTransformer(
            str(metadata["model"]),
            device="cpu",
            local_files_only=True,
        )
        _CACHE.update(key=cache_key, index=index, model=model, metadata=metadata)
        return index, model, metadata


def semantic_search(
    query: str,
    *,
    limit: int = 20,
    kinds: set[str] | None = None,
    store_notes: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    valid_paths: set[str] | None = None
    if store_notes is not None and METADATA_FILE.is_file():
        metadata_preview = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
        indexed_notes = [note for note in store_notes if note.get("kind") in INDEX_KINDS]
        if metadata_preview.get("signature") != _signature(indexed_notes, str(metadata_preview["model"])):
            old_hashes = metadata_preview.get("note_hashes")
            if (
                not isinstance(old_hashes, dict)
                or metadata_preview.get("schema_version") != INDEX_SCHEMA_VERSION
                or str(metadata_preview.get("model")) != _model_name()
            ):
                return []
            current_hashes = {
                str(note["path"]): _content_hash(note) for note in indexed_notes
            }
            valid_paths = {
                str(path) for path, content_hash in old_hashes.items()
                if current_hashes.get(str(path)) == str(content_hash)
            }
            # Reuse only hash-identical pages after small knowledge updates;
            # changed/deleted pages never leak from an old vector index.
            if not valid_paths:
                return []
    loaded = _load()
    if loaded is None or not query.strip():
        return []
    index, model, metadata = loaded
    vector = model.encode([_query_input(query, str(metadata["model"]))], normalize_embeddings=True)
    scores, positions = index.search(vector, min(index.ntotal, max(limit * 4, 40)))
    best_by_path: dict[str, dict[str, Any]] = {}
    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue
        record = metadata["records"][int(position)]
        if valid_paths is not None and str(record.get("path")) not in valid_paths:
            continue
        if kinds and record["kind"] not in kinds:
            continue
        item = {**record, "semantic_score": round(float(score), 4)}
        old = best_by_path.get(record["path"])
        if old is None or item["semantic_score"] > old["semantic_score"]:
            best_by_path[record["path"]] = item
    # Textbook arguments frequently cross a PDF page boundary. Propagate a
    # strong page hit to its immediate neighbours with a small score penalty,
    # preserving each neighbour as its own auditable path/title.
    records_by_path: dict[str, list[dict[str, Any]]] = {}
    for record in metadata.get("records", []):
        if valid_paths is not None and str(record.get("path")) not in valid_paths:
            continue
        records_by_path.setdefault(str(record.get("path") or ""), []).append(record)
    seeds = sorted(
        best_by_path.values(), key=lambda item: item["semantic_score"], reverse=True,
    )[:8]
    for seed in seeds:
        for neighbour_path in _adjacent_pdf_paths(str(seed.get("path") or "")):
            neighbour_records = records_by_path.get(neighbour_path, [])
            if not neighbour_records:
                continue
            propagated_score = round(max(0.0, float(seed["semantic_score"]) - 0.008), 4)
            existing = best_by_path.get(neighbour_path)
            if existing is not None and float(existing["semantic_score"]) >= propagated_score:
                continue
            first_record = min(
                neighbour_records, key=lambda item: int(item.get("chunk_index", 0) or 0),
            )
            best_by_path[neighbour_path] = {
                **first_record,
                "semantic_score": propagated_score,
                "adjacent_to": str(seed.get("path") or ""),
            }
    expanded: list[dict[str, Any]] = []
    records = list(metadata.get("records", []))
    for item in best_by_path.values():
        # A semantic hit may land on either the explanation before a figure or
        # the caption before the explanatory paragraph. Keep a bounded,
        # same-page window in both directions so neither half is discarded.
        text, indices = _expand_record_window(records, item, before=3, after=3)
        expanded.append({**item, "text": text or str(item.get("text") or ""), "window_chunk_indices": indices})
    return sorted(expanded, key=lambda item: item["semantic_score"], reverse=True)[:limit]
