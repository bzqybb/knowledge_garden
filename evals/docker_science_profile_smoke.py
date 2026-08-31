from __future__ import annotations

import json

from evals.science_code_execution import execute_answer_in_docker


PROFILES = {
    "bio": """```python
from rdkit import Chem
import scanpy
import pgmpy
import Bio
import msprime
import allel
import cobra
import statsmodels
print('bio_profile=ok')
```""",
    "quantum": """```python
import qiskit
import qiskit_aer
import qiskit_nature
import pyscf
import pymatching
print('quantum_profile=ok')
```""",
}


def main() -> None:
    results = {}
    for profile, answer in PROFILES.items():
        result = execute_answer_in_docker(answer, timeout_seconds=30)
        results[profile] = {
            "status": result.get("status"),
            "reason": result.get("reason"),
            "image": result.get("image"),
            "exit_code": result.get("exit_code"),
            "stdout": result.get("stdout"),
            "stderr": str(result.get("stderr") or "")[-1000:],
        }
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if any(item["status"] != "passed" for item in results.values()):
        raise SystemExit("one or more science profile imports failed")


if __name__ == "__main__":
    main()
