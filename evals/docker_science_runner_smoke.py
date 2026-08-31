from __future__ import annotations

import json
import subprocess

from evals.science_code_execution import (
    build_docker_python_command,
    execute_python_in_docker,
)


CASES = (
    (
        "normal",
        """import numpy as np
import scipy
import sympy as sp
import matplotlib
import networkx as nx
import pandas as pd
import sklearn
import z3
print(int(np.sum([1, 3])))
""",
        10,
        65_536,
    ),
    ("timeout", "while True:\n    pass\n", 2, 8_192),
    ("output_limit", "while True:\n    print('x' * 1024)\n", 5, 8_192),
)


def main() -> None:
    results = []
    for name, source, timeout_seconds, max_output_bytes in CASES:
        result = execute_python_in_docker(
            source,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
        results.append({
            "name": name,
            "status": result.get("status"),
            "reason": result.get("reason"),
            "exit_code": result.get("exit_code"),
            "duration_seconds": result.get("duration_seconds"),
            "timed_out": result.get("timed_out"),
            "output_limited": result.get("output_limited"),
            "stdout_bytes": len(str(result.get("stdout") or "").encode("utf-8")),
            "stdout_preview": str(result.get("stdout") or "")[:80],
            "stderr_preview": str(result.get("stderr") or "")[:200],
        })

    # This is evaluator-owned probe code, not model-written code.  It bypasses
    # static model-code policy solely to verify the container boundary itself.
    probe_source = """import json, os, socket
from pathlib import Path
probe = {"uid": os.getuid()}
try:
    Path("/root/should-not-write").write_text("x")
    probe["root_write"] = "unexpected_success"
except Exception as exc:
    probe["root_write"] = "blocked:" + type(exc).__name__
s = socket.socket()
s.settimeout(2)
try:
    s.connect(("1.1.1.1", 53))
    probe["network"] = "unexpected_success"
except Exception as exc:
    probe["network"] = "blocked:" + type(exc).__name__
finally:
    s.close()
print(json.dumps(probe))
"""
    command = build_docker_python_command()
    probe_process = subprocess.run(
        command,
        input=probe_source.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    probe = json.loads(probe_process.stdout.decode("utf-8"))
    payload = {"execution_cases": results, "isolation_probe": probe}
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    normal, timeout, output_limit = results
    healthy = (
        normal["status"] == "passed"
        and normal["stdout_preview"] == "4\n"
        and timeout["reason"] == "timeout"
        and output_limit["reason"] == "output_limit"
        and probe.get("uid") == 65534
        and str(probe.get("root_write", "")).startswith("blocked:")
        and str(probe.get("network", "")).startswith("blocked:")
    )
    if not healthy:
        raise SystemExit("science runner isolation smoke failed")


if __name__ == "__main__":
    main()
