import json
import unittest
from pathlib import Path

from core.reasoning_capability import classify_reasoning_task, evidence_route


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evals" / "datasets" / "zhili_transfer_15_v1.jsonl"
SYMBOLIC_DATASET = ROOT / "evals" / "datasets" / "zhili_transfer_symbolic_checks_v1.json"


class TransferDatasetTests(unittest.TestCase):
    def test_transfer_pack_is_well_formed_and_unseen(self):
        cases = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(cases), 15)
        self.assertEqual(len({case["id"] for case in cases}), 15)
        self.assertEqual({case["discipline"] for case in cases}, {
            "数学", "物理", "化学", "物理化学", "计算机科学", "理论计算机", "生物学", "生态学", "遗传学",
        })
        for case in cases:
            self.assertTrue(case["question"].strip())
            self.assertTrue(case["reference"].strip())
            self.assertGreaterEqual(len(case["rubric"]), 3)

    def test_transfer_pack_never_searches(self):
        failures = []
        for line in DATASET.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            profile = classify_reasoning_task(case["question"])
            route = evidence_route(case["question"], profile=profile)
            if route["routing_target"] != "MUST_NOT_SEARCH":
                failures.append((case["id"], route))
        self.assertEqual(failures, [])

    def test_transfer_pack_has_independent_symbolic_coverage(self):
        cases = {
            json.loads(line)["id"]
            for line in DATASET.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        checks = json.loads(SYMBOLIC_DATASET.read_text(encoding="utf-8"))
        self.assertEqual(set(checks) - cases, set())
        self.assertGreaterEqual(len(checks), 10)
        self.assertTrue(all(items for items in checks.values()))


if __name__ == "__main__":
    unittest.main()
