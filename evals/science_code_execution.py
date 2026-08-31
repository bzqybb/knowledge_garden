from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
import threading
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / "evals" / "execution_plans"
DEFAULT_DOCKER_IMAGE = "zhili-science-runner:latest"
DOCKER_IMAGE_BY_IMPORT = {
    "Bio": "zhili-science-bio:latest",
    "allel": "zhili-science-bio:latest",
    "cobra": "zhili-science-bio:latest",
    "msprime": "zhili-science-bio:latest",
    "pgmpy": "zhili-science-bio:latest",
    "rdkit": "zhili-science-bio:latest",
    "scanpy": "zhili-science-bio:latest",
    "statsmodels": "zhili-science-bio:latest",
    "pymatching": "zhili-science-quantum:latest",
    "qiskit": "zhili-science-quantum:latest",
    "qiskit_aer": "zhili-science-quantum:latest",
    "qiskit_nature": "zhili-science-quantum:latest",
    "pyscf": "zhili-science-quantum:latest",
    "torch": "zhili-science-ai:latest",
}

PYTHON_FENCE_RE = re.compile(
    r"```(?:python|py)\s*\r?\n(.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)

# Static analysis is only a precondition.  Passing this list never means that
# code is safe to run in the host interpreter; every accepted block still
# requires an isolated backend with no host mounts and no network.
BLOCKED_IMPORT_ROOTS = {
    "asyncio", "builtins", "ctypes", "ftplib", "http", "httpx", "importlib",
    "multiprocessing", "os", "pathlib", "pickle", "requests", "shelve",
    "shutil", "socket", "subprocess", "tempfile", "urllib", "webbrowser",
}
BLOCKED_CALL_NAMES = {
    "__import__", "breakpoint", "compile", "delattr", "eval", "exec",
    "getattr", "globals", "input", "locals", "open", "setattr", "vars",
}
BLOCKED_ATTRIBUTE_CALLS = {
    "check_call", "check_output", "chmod", "chown", "connect", "download",
    "kill", "makedirs", "mkdir", "move", "popen", "rename", "request",
    "rmdir", "rmtree", "send", "spawn", "system", "unlink", "urlopen",
    "write_bytes", "write_text",
}


def extract_python_blocks(answer: str) -> list[str]:
    return [block.strip() for block in PYTHON_FENCE_RE.findall(str(answer or ""))]


class _StaticVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports: set[str] = set()
        self.risks: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            self.imports.add(root)
            if root in BLOCKED_IMPORT_ROOTS:
                self.risks.add(f"blocked_import:{root}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        root = str(node.module or "").split(".", 1)[0]
        if node.level:
            self.risks.add("relative_import")
        if root:
            self.imports.add(root)
            if root in BLOCKED_IMPORT_ROOTS:
                self.risks.add(f"blocked_import:{root}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        if node.attr.startswith("_"):
            self.risks.add("private_or_dunder_attribute")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALL_NAMES:
            self.risks.add(f"blocked_call:{node.func.id}")
        if isinstance(node.func, ast.Attribute) and node.func.attr in BLOCKED_ATTRIBUTE_CALLS:
            self.risks.add(f"blocked_attribute_call:{node.func.attr}")
        self.generic_visit(node)


def audit_python_block(code: str, *, index: int) -> dict[str, Any]:
    digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {
            "index": index,
            "sha256": digest,
            "characters": len(code),
            "syntax_valid": False,
            "syntax_error": {
                "line": exc.lineno,
                "offset": exc.offset,
                "message": exc.msg,
            },
            "imports": [],
            "risks": ["syntax_error"],
            "decision": "rejected_syntax",
        }
    visitor = _StaticVisitor()
    visitor.visit(tree)
    risks = sorted(visitor.risks)
    return {
        "index": index,
        "sha256": digest,
        "characters": len(code),
        "syntax_valid": True,
        "syntax_error": None,
        "imports": sorted(visitor.imports),
        "risks": risks,
        "decision": "rejected_static_risk" if risks else "isolated_backend_required",
    }


def audit_answer_code(answer: str) -> dict[str, Any]:
    blocks = extract_python_blocks(answer)
    audited = [audit_python_block(code, index=index) for index, code in enumerate(blocks, 1)]
    decisions = Counter(str(item["decision"]) for item in audited)
    if not audited:
        overall = "no_python_block"
    elif decisions.get("rejected_syntax"):
        overall = "rejected_syntax"
    elif decisions.get("rejected_static_risk"):
        overall = "rejected_static_risk"
    else:
        overall = "isolated_backend_required"
    return {
        "policy_version": "science-code-static-audit-v1",
        "python_blocks": len(audited),
        "syntax_valid_blocks": sum(bool(item["syntax_valid"]) for item in audited),
        "imports": sorted({name for item in audited for name in item["imports"]}),
        "risks": sorted({risk for item in audited for risk in item["risks"]}),
        "decision": overall,
        "host_execution_allowed": False,
        "blocks": audited,
    }


def docker_backend_status() -> dict[str, Any]:
    executable = shutil.which("docker")
    return {
        "backend": "docker",
        "available": bool(executable),
        "executable": executable,
        "auto_install": False,
        "auto_pull": False,
        "host_execution_fallback": False,
    }


def build_docker_python_command(
    *,
    image: str = DEFAULT_DOCKER_IMAGE,
    memory_mb: int = 512,
    cpus: float = 1.0,
    pids_limit: int = 64,
) -> list[str]:
    """Build the only permitted command for executing model-written Python.

    Source is provided on stdin, so there is no host bind mount.  The command
    never pulls an image and cannot fall back to the host interpreter.
    """
    clean_image = str(image or "").strip()
    if not clean_image or any(char.isspace() for char in clean_image):
        raise ValueError("Docker image name is empty or contains whitespace")
    memory_mb = max(128, min(4096, int(memory_mb)))
    cpus = max(0.25, min(4.0, float(cpus)))
    pids_limit = max(16, min(256, int(pids_limit)))
    return [
        "docker", "run", "--rm", "--interactive", "--pull=never",
        "--network=none", "--read-only", "--cap-drop=ALL",
        "--security-opt=no-new-privileges", "--user=65534:65534",
        f"--memory={memory_mb}m", f"--cpus={cpus:g}",
        f"--pids-limit={pids_limit}",
        "--ulimit=nofile=64:64", "--ulimit=fsize=16777216:16777216",
        "--ulimit=core=0:0", "--stop-timeout=1",
        "--tmpfs=/tmp:rw,noexec,nosuid,size=64m",
        "--env=PYTHONDONTWRITEBYTECODE=1", "--env=PYTHONHASHSEED=0",
        "--env=MPLBACKEND=Agg", "--env=HOME=/tmp",
        "--env=XDG_CACHE_HOME=/tmp/.cache",
        "--env=MPLCONFIGDIR=/tmp/matplotlib",
        "--env=OMP_NUM_THREADS=1", "--env=OPENBLAS_NUM_THREADS=1",
        "--env=MKL_NUM_THREADS=1", "--env=NUMEXPR_NUM_THREADS=1",
        clean_image, "python", "-I", "-",
    ]


def execute_python_in_docker(
    code: str,
    *,
    image: str = DEFAULT_DOCKER_IMAGE,
    timeout_seconds: float = 20.0,
    max_output_bytes: int = 65_536,
    memory_mb: int = 512,
    cpus: float = 1.0,
    pids_limit: int = 64,
) -> dict[str, Any]:
    """Execute one statically accepted block inside the locked-down image.

    Output is drained incrementally so an infinite-print program cannot make
    the host process retain unbounded data.  A unique container name lets the
    timeout path force-remove only the container created by this call.
    """
    source = str(code or "")
    audit = audit_python_block(source, index=1)
    if audit["decision"] != "isolated_backend_required":
        return {
            "status": "blocked", "reason": f"static_policy:{audit['decision']}",
            "audit": audit, "backend": "docker", "executed": False,
        }

    backend = docker_backend_status()
    if not backend["available"]:
        return {
            "status": "blocked", "reason": "isolated_backend_unavailable",
            "audit": audit, "backend": "docker", "executed": False,
        }

    timeout_seconds = max(1.0, min(300.0, float(timeout_seconds)))
    max_output_bytes = max(4096, min(4_194_304, int(max_output_bytes)))
    container_name = f"zhili-science-{uuid.uuid4().hex[:16]}"
    command = build_docker_python_command(
        image=image, memory_mb=memory_mb, cpus=cpus, pids_limit=pids_limit,
    )
    command[0] = str(backend["executable"])
    command[2:2] = [f"--name={container_name}"]

    started = time.monotonic()
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creationflags,
    )
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    total = 0
    lock = threading.Lock()
    output_limited = threading.Event()

    def _drain(name: str, stream: Any) -> None:
        nonlocal total
        while True:
            chunk = stream.read(4096)
            if not chunk:
                return
            with lock:
                remaining = max_output_bytes - total
                if remaining > 0:
                    kept = chunk[:remaining]
                    buffers[name].extend(kept)
                    total += len(kept)
                if len(chunk) > remaining or total >= max_output_bytes:
                    output_limited.set()

    def _write_source() -> None:
        assert process.stdin is not None
        try:
            process.stdin.write(source.encode("utf-8"))
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass

    readers = [
        threading.Thread(target=_drain, args=("stdout", process.stdout), daemon=True),
        threading.Thread(target=_drain, args=("stderr", process.stderr), daemon=True),
    ]
    writer = threading.Thread(target=_write_source, daemon=True)
    for thread in readers:
        thread.start()
    writer.start()

    timed_out = False
    while process.poll() is None:
        if output_limited.is_set():
            break
        if time.monotonic() - started >= timeout_seconds:
            timed_out = True
            break
        time.sleep(0.05)

    if process.poll() is None:
        process.kill()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        # The Docker client may die before --rm is processed.  This cleanup is
        # scoped to the unique name created above and never touches user data.
        subprocess.run(
            [str(backend["executable"]), "rm", "-f", container_name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=5, check=False, creationflags=creationflags,
        )

    writer.join(timeout=1)
    for thread in readers:
        thread.join(timeout=2)
    exit_code = process.poll()
    duration = round(time.monotonic() - started, 3)
    if timed_out:
        status, reason = "failed", "timeout"
    elif output_limited.is_set():
        status, reason = "failed", "output_limit"
    elif exit_code == 0:
        status, reason = "passed", "completed"
    else:
        status, reason = "failed", "nonzero_exit"
    return {
        "status": status,
        "reason": reason,
        "executed": True,
        "backend": "docker",
        "image": image,
        "code_sha256": audit["sha256"],
        "exit_code": exit_code,
        "duration_seconds": duration,
        "timed_out": timed_out,
        "output_limited": output_limited.is_set(),
        "stdout": bytes(buffers["stdout"]).decode("utf-8", errors="replace"),
        "stderr": bytes(buffers["stderr"]).decode("utf-8", errors="replace"),
        "audit": audit,
        "isolation": {
            "network": "none", "host_mounts": "none", "root": False,
            "read_only_root": True, "auto_pull": False,
            "memory_mb": max(128, min(4096, int(memory_mb))),
            "cpus": max(0.25, min(4.0, float(cpus))),
            "pids_limit": max(16, min(256, int(pids_limit))),
        },
    }


def execute_answer_in_docker(
    answer: str,
    *,
    image: str | None = None,
    timeout_seconds: float = 20.0,
    max_output_bytes: int = 65_536,
) -> dict[str, Any]:
    """Execute all Python fences from one answer in order and one namespace."""
    blocks = extract_python_blocks(answer)
    answer_audit = audit_answer_code(answer)
    selected_images = {
        DOCKER_IMAGE_BY_IMPORT[name]
        for name in answer_audit.get("imports", [])
        if name in DOCKER_IMAGE_BY_IMPORT
    }
    if image:
        selected_image = image
    elif len(selected_images) == 1:
        selected_image = next(iter(selected_images))
    elif len(selected_images) > 1:
        return {
            "status": "blocked", "reason": "multiple_dependency_profiles_required",
            "executed": False, "backend": "docker", "answer_audit": answer_audit,
            "required_images": sorted(selected_images),
        }
    else:
        selected_image = DEFAULT_DOCKER_IMAGE
    if not blocks:
        return {
            "status": "no_python_block", "reason": "no_python_block",
            "executed": False, "backend": "docker", "answer_audit": answer_audit,
        }
    if answer_audit["decision"] != "isolated_backend_required":
        return {
            "status": "blocked",
            "reason": f"static_policy:{answer_audit['decision']}",
            "executed": False, "backend": "docker", "answer_audit": answer_audit,
        }
    combined = "\n\n".join(
        f"# --- answer block {index} ---\n{source}"
        for index, source in enumerate(blocks, 1)
    )
    result = execute_python_in_docker(
        combined,
        image=selected_image,
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
    )
    result["answer_audit"] = answer_audit
    result["blocks_combined"] = len(blocks)
    result["combined_code_sha256"] = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return result


def execution_preflight(audit: dict[str, Any]) -> dict[str, Any]:
    """Refuse execution until both static policy and strong isolation pass."""
    decision = str(audit.get("decision") or "")
    backend = docker_backend_status()
    allowed_by_static_policy = decision == "isolated_backend_required"
    ready = allowed_by_static_policy and bool(backend["available"])
    if not allowed_by_static_policy:
        reason = f"static_policy:{decision or 'unknown'}"
    elif not backend["available"]:
        reason = "isolated_backend_unavailable"
    else:
        reason = "ready_for_isolated_execution"
    return {
        "ready": ready,
        "reason": reason,
        "backend": backend,
        "command_policy": {
            "network": "none", "host_mounts": "none", "root": False,
            "read_only_root": True, "auto_pull": False,
        },
    }


def build_execution_plan(report: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in report.get("rows") or []:
        for surface in ("gardener", "inspiration"):
            result = dict(row.get(surface) or {})
            audit = audit_answer_code(str(result.get("answer") or ""))
            records.append({
                "schema_version": "science-code-execution-plan-v1",
                "case_id": row.get("id"),
                "discipline": row.get("discipline"),
                "surface": surface,
                "generation_failed": bool(result.get("generation_failed")),
                "audit": audit,
                "preflight": execution_preflight(audit),
                "execution": {
                    "status": "not_executed",
                    "backend": None,
                    "network": "must_be_disabled",
                    "host_mounts": "must_be_none",
                    "reason": (
                        "静态门控未通过。"
                        if audit["decision"].startswith("rejected_") else
                        "等待具备无网络、无宿主挂载、限时限内存的隔离后端。"
                    ),
                },
            })
    decisions = Counter(record["audit"]["decision"] for record in records)
    imports = Counter(name for record in records for name in record["audit"]["imports"])
    risks = Counter(name for record in records for name in record["audit"]["risks"])
    summary = {
        "schema_version": "science-code-execution-plan-summary-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "surface_records": len(records),
        "python_blocks": sum(record["audit"]["python_blocks"] for record in records),
        "executed": 0,
        "decisions": dict(sorted(decisions.items())),
        "imports": dict(imports.most_common()),
        "risks": dict(risks.most_common()),
        "safety_policy": (
            "静态通过不等于安全；禁止在宿主 Python 中运行模型代码。"
            "仅允许无网络、无宿主挂载、限时限内存且保存完整日志的隔离后端。"
        ),
    }
    return records, summary


def execute_report_code(
    report: dict[str, Any],
    *,
    limit: int | None = None,
    image: str = DEFAULT_DOCKER_IMAGE,
    timeout_seconds: float = 20.0,
    max_output_bytes: int = 65_536,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run statically accepted report blocks and retain auditable traces.

    A zero exit code establishes runtime executability only.  It is never
    promoted to scientific correctness without a separate result oracle.
    """
    execution_budget = None if limit is None else max(0, int(limit))
    records: list[dict[str, Any]] = []
    executed = 0
    runtime_counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for row in report.get("rows") or []:
        for surface in ("gardener", "inspiration"):
            result = dict(row.get(surface) or {})
            blocks = extract_python_blocks(str(result.get("answer") or ""))
            block_records = []
            for block_index, code in enumerate(blocks, 1):
                audit = audit_python_block(code, index=block_index)
                if audit["decision"] != "isolated_backend_required":
                    execution = {
                        "status": "blocked", "reason": f"static_policy:{audit['decision']}",
                        "executed": False, "backend": "docker", "audit": audit,
                    }
                elif execution_budget is not None and executed >= execution_budget:
                    execution = {
                        "status": "not_executed", "reason": "execution_limit_reached",
                        "executed": False, "backend": "docker", "audit": audit,
                    }
                else:
                    print(
                        f"[code-run {executed + 1}] {row.get('id')} / {surface} / "
                        f"block {block_index}",
                        flush=True,
                    )
                    execution = execute_python_in_docker(
                        code,
                        image=image,
                        timeout_seconds=timeout_seconds,
                        max_output_bytes=max_output_bytes,
                    )
                    executed += 1
                block_records.append(execution)
                runtime_counts[str(execution.get("status") or "unknown")] += 1
                reasons[str(execution.get("reason") or "unknown")] += 1

            if not blocks:
                surface_status = "no_python_block"
            elif any(item.get("status") == "failed" for item in block_records):
                surface_status = "runtime_failed"
            elif all(item.get("status") == "passed" for item in block_records):
                surface_status = "runtime_passed"
            elif any(item.get("status") == "passed" for item in block_records):
                surface_status = "partially_executed"
            elif any(item.get("status") == "not_executed" for item in block_records):
                surface_status = "not_executed"
            else:
                surface_status = "blocked_static"
            records.append({
                "schema_version": "science-code-runtime-v1",
                "case_id": row.get("id"),
                "discipline": row.get("discipline"),
                "surface": surface,
                "generation_failed": bool(result.get("generation_failed")),
                "runtime_status": surface_status,
                "scientific_correctness_verified": False,
                "blocks": block_records,
            })

    surface_counts = Counter(record["runtime_status"] for record in records)
    summary = {
        "schema_version": "science-code-runtime-summary-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "surface_records": len(records),
        "code_blocks_seen": sum(len(record["blocks"]) for record in records),
        "executed": executed,
        "runtime_counts": dict(sorted(runtime_counts.items())),
        "reason_counts": dict(sorted(reasons.items())),
        "surface_runtime_status": dict(sorted(surface_counts.items())),
        "scientific_correctness_verified": 0,
        "interpretation": (
            "runtime_passed 仅表示代码在隔离环境中以退出码 0 结束；"
            "不表示公式、数值或科学结论正确。"
        ),
        "image": image,
        "timeout_seconds": timeout_seconds,
        "max_output_bytes": max_output_bytes,
    }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execute-ready", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--image", default=DEFAULT_DOCKER_IMAGE)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--max-output-bytes", type=int, default=65_536)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if args.execute_ready:
        records, summary = execute_report_code(
            report,
            limit=args.limit,
            image=args.image,
            timeout_seconds=args.timeout_seconds,
            max_output_bytes=args.max_output_bytes,
        )
        stem = "science100-code-run"
    else:
        records, summary = build_execution_plan(report)
        stem = "science100-code-plan"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    jsonl_path = args.output_dir / f"{stem}-{stamp}.jsonl"
    summary_path = args.output_dir / f"{stem}-{stamp}.summary.json"
    jsonl_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "records": len(records),
        "jsonl": str(jsonl_path),
        "summary": str(summary_path),
        "counts": summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
