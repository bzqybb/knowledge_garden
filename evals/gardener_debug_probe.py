from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from evals.adapter import load_cases
from evals.boundary_eval import run_case


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "evals" / "datasets" / "zhili_dual_surface_5_v1.jsonl"
REPORT_DIR = ROOT / "evals" / "reports"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="ZS-MATH-01")
    args = parser.parse_args()
    os.environ["GARDEN_DISABLE_NETWORK"] = "1"
    case = next(item for item in load_cases(DATASET) if item["id"] == args.id)
    row = run_case({"id": case["id"], "question": case["question"], "category": case["discipline"]})
    row["reference"] = case["reference"]
    row["rubric"] = case["rubric"]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = REPORT_DIR / f"gardener-debug-probe-{case['id'].lower()}-{stamp}.json"
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(path), "answer": row.get("answer", ""),
        "latency_ms": row.get("latency_ms"), "node_timings_ms": row.get("node_timings_ms", {}),
        "repair_degraded": row.get("quality_review", {}).get("repair_degraded", row.get("repair_degraded")),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
