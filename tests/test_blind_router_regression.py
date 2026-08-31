import json
import unittest
from pathlib import Path

from core.reasoning_capability import classify_reasoning_task, evidence_route


ROOT = Path(__file__).resolve().parents[1]
BLIND_DATASET = ROOT / "evals" / "datasets" / "zhili_blind_20_v1.jsonl"


class BlindRouterRegressionTests(unittest.TestCase):
    def test_all_twenty_blind_cases_are_true_closed_loop_tasks(self):
        cases = [
            json.loads(line)
            for line in BLIND_DATASET.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(cases), 20)
        failures = []
        for case in cases:
            profile = classify_reasoning_task(case["question"])
            route = evidence_route(case["question"], profile=profile)
            if route["routing_target"] != "MUST_NOT_SEARCH":
                failures.append((case["id"], route))
        self.assertEqual(failures, [])

    def test_current_facts_still_require_search(self):
        question = "请检索截至目前某项政策的最新官方统计，并据此计算增长率。"
        profile = classify_reasoning_task(question)
        route = evidence_route(question, profile=profile)
        self.assertEqual(route["routing_target"], "SEARCH_FIRST_THEN_PROVE")


if __name__ == "__main__":
    unittest.main()
