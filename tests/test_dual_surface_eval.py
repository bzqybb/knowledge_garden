import unittest
from collections import Counter
from unittest.mock import patch

from evals.adapter import load_cases
from evals.adversarial_foundations_eval import answer_symbolic_grounding
from evals.dual_surface_capability_eval import (
    DATASET,
    attach_symbolic_checks,
    attach_runtime_code_execution,
    bounded_symbolic_grounding,
    bounded_surface_pair,
    build_blind_judge_payload,
    classify_surface_infrastructure,
    deterministic_answer_oracle,
    enforce_hard_failure_gates,
    freeze_rubric,
    normalize_capability_case,
    repair_failed_runtime_answers,
    route_matches_expectation,
    refresh_row_infrastructure,
    summarize,
    verify_frozen_rubric,
)


class DualSurfaceEvalTests(unittest.TestCase):
    BLIND_DATASET = DATASET.parent / "zhili_blind_20_v1.jsonl"
    SCIENCE_100_DATASET = DATASET.parent / "science_exploration_100_v1.jsonl"

    def test_all_five_cases_freeze_atomic_rubric_and_symbolic_check(self):
        cases = load_cases(DATASET)
        self.assertEqual(len(cases), 5)
        for case in cases:
            with self.subTest(case=case["id"]):
                enriched = attach_symbolic_checks(case)
                rubric, digest = freeze_rubric(enriched)
                self.assertGreaterEqual(len(rubric), 3)
                self.assertLessEqual(len(rubric), 5)
                self.assertAlmostEqual(sum(item["weight"] for item in rubric), 1.0)
                self.assertEqual(len(digest), 64)
                self.assertTrue(enriched["symbolic_checks"])

    def test_rubric_hash_binds_case_question_reference_and_rejects_tamper(self):
        case = load_cases(DATASET)[0]
        rubric, digest = freeze_rubric(case)
        row = {**case, "scoring_rubric": rubric, "rubric_hash": digest}
        verify_frozen_rubric(row)
        row["reference"] += " 篡改"
        with self.assertRaisesRegex(ValueError, "已被篡改"):
            verify_frozen_rubric(row)

    def test_blind_judge_payload_contains_one_anonymous_answer_only(self):
        case = load_cases(DATASET)[0]
        rubric, digest = freeze_rubric(case)
        row = {
            **case, "scoring_rubric": rubric, "rubric_hash": digest,
            "gardener": {"answer": "G-only", "symbolic_grounding": {"passed": True}},
            "inspiration": {"answer": "I-only", "symbolic_grounding": {"passed": False}},
        }
        payload = build_blind_judge_payload(row, "gardener")
        self.assertEqual(payload["anonymous_answer"], "G-only")
        self.assertNotIn("I-only", str(payload))
        self.assertNotIn("gardener", payload)
        self.assertNotIn("inspiration", payload)
        self.assertNotIn("surface", payload)
        self.assertFalse(payload["tool_execution_verified"])

    def test_trusted_runtime_trace_is_forwarded_to_blind_judge(self):
        case = load_cases(DATASET)[0]
        rubric, digest = freeze_rubric(case)
        trace = {"status": "passed", "stdout": "4\n", "stderr": ""}
        row = {
            **case, "scoring_rubric": rubric, "rubric_hash": digest,
            "gardener": {"answer": "G", "symbolic_grounding": {}, "local_checks": {
                "tool_execution_verified": True, "tool_execution": trace,
            }},
            "inspiration": {"answer": "I", "symbolic_grounding": {}},
        }
        payload = build_blind_judge_payload(row, "gardener")
        self.assertTrue(payload["tool_execution_verified"])
        self.assertEqual(payload["trusted_tool_execution"], trace)

    def test_runtime_execution_helper_sets_verified_only_on_pass(self):
        row = {
            "run_id": "r", "id": "x",
            "gardener": {"answer": "```python\nprint(1)\n```", "local_checks": {}},
            "inspiration": {"answer": "```python\nraise ValueError()\n```", "local_checks": {}},
            "audit_events": [],
        }
        with patch(
            "evals.dual_surface_capability_eval.execute_answer_in_docker",
            side_effect=[
                {"status": "passed", "reason": "completed", "executed": True},
                {"status": "failed", "reason": "nonzero_exit", "executed": True},
            ],
        ):
            attach_runtime_code_execution(row)
        self.assertTrue(row["gardener"]["local_checks"]["tool_execution_verified"])
        self.assertFalse(row["inspiration"]["local_checks"]["tool_execution_verified"])

    def test_runtime_repair_preserves_initial_answer_and_accepts_verified_candidate(self):
        initial_trace = {
            "executed": True, "status": "failed", "reason": "nonzero_exit",
            "stderr": "AttributeError: bad api",
        }
        row = {
            "run_id": "r", "id": "SCI-X-01", "question": "请推导并验证",
            "gardener": {"answer": "首稿", "revision_count": 0, "local_checks": {
                "tool_execution": initial_trace,
            }},
            "inspiration": {"answer": "无代码", "local_checks": {
                "tool_execution": {"executed": False, "status": "no_python_block", "reason": "no_python_block"},
            }},
            "audit_events": [],
        }
        candidate = "完整回答\n```python\nprint('fixed')\n```"
        with patch(
            "evals.dual_surface_capability_eval.repair_answer_with_retries",
            side_effect=[
                {
                    "eligible": True, "attempted": True, "accepted": True,
                    "reason": "candidate_execution_passed", "candidate_answer": candidate,
                    "candidate_execution": {"status": "passed", "executed": True, "stdout": "fixed\n"},
                },
                {
                    "eligible": False, "attempted": False, "accepted": False,
                    "reason": "runtime_failure_not_repairable",
                },
            ],
        ):
            repair_failed_runtime_answers(row)
        self.assertEqual(row["gardener"]["answer_before_runtime_repair"], "首稿")
        self.assertEqual(row["gardener"]["answer"], candidate)
        self.assertTrue(row["gardener"]["local_checks"]["tool_execution_verified"])
        self.assertEqual(row["gardener"]["revision_count"], 1)
        self.assertFalse(row["inspiration"]["runtime_repair"]["attempted"])

    def test_runtime_repair_resumes_from_latest_candidate_without_repeating_attempts(self):
        failed_trace = {
            "executed": True, "status": "failed", "reason": "nonzero_exit",
            "stderr": "third api error",
        }
        prior = [
            {"attempt": 1, "candidate_answer": "候选一", "candidate_execution": failed_trace},
            {"attempt": 2, "candidate_answer": "候选二", "candidate_execution": failed_trace},
        ]
        row = {
            "run_id": "r", "id": "SCI-X-02", "question": "请修复",
            "gardener": {
                "answer": "首稿", "local_checks": {"tool_execution": failed_trace},
                "runtime_repair": {
                    "accepted": False, "attempts": prior,
                    "initial_answer_sha256": "original", "initial_execution": failed_trace,
                },
            },
            "inspiration": {"answer": "无代码", "local_checks": {
                "tool_execution": {"executed": False, "status": "no_python_block", "reason": "no_python_block"},
            }},
            "audit_events": [],
        }
        final_answer = "最终回答\n```python\nprint('ok')\n```"

        def fake_repair(**kwargs):
            if kwargs["answer"] == "候选二":
                self.assertEqual(kwargs["max_attempts"], 1)
                return {
                    "eligible": True, "attempted": True, "accepted": True,
                    "reason": "candidate_execution_passed",
                    "candidate_answer": final_answer,
                    "candidate_execution": {"status": "passed", "executed": True},
                    "attempts": [{"attempt": 1, "candidate_answer": final_answer}],
                }
            return {
                "eligible": False, "attempted": False, "accepted": False,
                "reason": "runtime_failure_not_repairable", "attempts": [],
            }

        with patch(
            "evals.dual_surface_capability_eval.repair_answer_with_retries",
            side_effect=fake_repair,
        ):
            repair_failed_runtime_answers(row)
        self.assertEqual(row["gardener"]["answer"], final_answer)
        self.assertEqual(len(row["gardener"]["runtime_repair"]["attempts"]), 3)
        self.assertEqual(row["gardener"]["runtime_repair"]["attempts"][-1]["attempt"], 3)

    def test_dataset_can_embed_held_out_symbolic_checks(self):
        embedded = [{"id": "held-out", "target_lhs": "x", "symbols": [], "rhs": "1"}]
        enriched = attach_symbolic_checks({"id": "NEW-01", "symbolic_checks": embedded})
        self.assertEqual(enriched["symbolic_checks"], embedded)
        round_tripped = attach_symbolic_checks({"id": "NEW-01", **enriched})
        self.assertEqual(round_tripped["symbolic_checks"], embedded)

    def test_bounded_symbolic_grounding_preserves_normal_result(self):
        case = {"symbolic_checks": [{
            "id": "bounded", "target_lhs": "x", "symbols": [], "rhs": "1",
        }]}
        result = bounded_symbolic_grounding(case, "$x=1$", timeout_seconds=20)
        self.assertTrue(result["passed"], result)

    def test_surface_outer_timeout_returns_auditable_failure(self):
        case = {"id": "TIMEOUT-01", "discipline": "数学"}
        result = bounded_surface_pair(case, "测试", timeout_seconds=0.01)
        for surface in ("gardener", "inspiration"):
            payload, latency_ms = result[surface]
            self.assertTrue(payload["generation_failed"])
            self.assertEqual(payload["agent_trace"][0]["node"], "evaluation_outer_timeout")
            self.assertGreater(latency_ms, 0)

    def test_task_only_case_gets_frozen_four_part_science_rubric(self):
        case = normalize_capability_case({
            "id": "SCI-TEST-01", "discipline": "数学", "topic": "测试",
            "question": "推导并用 Python 验证一个结论。",
        })
        self.assertEqual(len(case["rubric"]), 5)
        self.assertIn("实际执行结果", case["reference"])
        self.assertTrue(case["requires_tool_execution"])
        self.assertIn("plan_sources", case["forbidden_routes"])
        self.assertEqual(len(freeze_rubric(case)[1]), 64)

    def test_routing_accuracy_uses_explicit_expected_route_only(self):
        matched = {"expected_route": "prepare_closed_loop", "gardener": {"agent_trace": [{"node": "prepare_closed_loop"}]}}
        mismatched = {"expected_route": "prepare_closed_loop", "gardener": {"agent_trace": [{"node": "prepare_model_knowledge"}]}}
        forbidden = {"expected_route": "prepare_closed_loop", "forbidden_routes": ["plan_sources"], "gardener": {"agent_trace": [{"node": "prepare_closed_loop"}, {"node": "plan_sources"}]}}
        unscored = {"gardener": {"agent_trace": [{"node": "prepare_closed_loop"}]}}
        self.assertTrue(route_matches_expectation(matched))
        self.assertFalse(route_matches_expectation(mismatched))
        self.assertFalse(route_matches_expectation(forbidden))
        self.assertIsNone(route_matches_expectation(unscored))

    def test_summary_counts_generation_failures_as_zero_dimension_score(self):
        rows = []
        for failed in (False, True):
            rows.append({
                "expected_route": "prepare_closed_loop",
                "auxiliary_judge": {
                    "rubric_results": [{}],
                    "gardener_dimensions": {"correctness": 5},
                    "inspiration_dimensions": {"correctness": 5},
                },
                "gardener": {"latency_ms": 1, "generation_failed": failed, "agent_trace": [{"node": "prepare_closed_loop"}], "local_checks": {"not_defensive_refusal": True}, "symbolic_grounding": {}},
                "inspiration": {"latency_ms": 1, "generation_failed": failed, "local_checks": {"not_defensive_refusal": True}, "symbolic_grounding": {}},
            })
        summary = summarize(rows)
        self.assertEqual(summary["routing_evaluated"], 2)
        self.assertEqual(summary["routing_accuracy"], 1.0)
        self.assertEqual(summary["gardener"]["dimension_averages"]["correctness"], 2.5)
        self.assertEqual(summary["gardener"]["dimension_success_only_averages"]["correctness"], 5.0)

    def test_infrastructure_classifier_separates_fatal_auth_from_transient_timeout(self):
        auth = classify_surface_infrastructure({
            "generation_failed": True,
            "agent_trace": [{"data": {"generation_error": "Error code: 401，令牌已过期或验证不正确"}}],
        })
        timeout = classify_surface_infrastructure({
            "generation_failed": True,
            "generation_diagnostics": {"errors": ["Request timed out."]},
        })
        capability = classify_surface_infrastructure({
            "generation_failed": True,
            "generation_diagnostics": {"errors": ["模型返回了空 JSON"]},
        })
        self.assertEqual(auth["category"], "credential")
        self.assertTrue(auth["fatal"])
        self.assertEqual(timeout["category"], "timeout")
        self.assertFalse(timeout["fatal"])
        self.assertFalse(capability["detected"])

    def test_legacy_report_infrastructure_is_backfilled_from_trace(self):
        row = {
            "gardener": {
                "generation_failed": True,
                "agent_trace": [{"data": {"generation_error": "Request timed out"}}],
            },
            "inspiration": {"generation_failed": False, "answer": "正常回答"},
        }
        refresh_row_infrastructure(row)
        self.assertTrue(row["gardener"]["infrastructure_failure"]["detected"])
        self.assertEqual(row["gardener"]["infrastructure_failure"]["category"], "timeout")
        self.assertFalse(row["inspiration"]["infrastructure_failure"]["detected"])

    def test_summary_excludes_infrastructure_failure_from_capability_denominator(self):
        rows = []
        for infrastructure in (False, True):
            score = 0.0 if infrastructure else 80.0
            dimensions = {"correctness": 1 if infrastructure else 5}
            surface = {
                "latency_ms": 10, "generation_failed": infrastructure,
                "infrastructure_failure": {"detected": infrastructure, "fatal": infrastructure},
                "agent_trace": [], "local_checks": {"not_defensive_refusal": True},
                "symbolic_grounding": {},
            }
            rows.append({
                "auxiliary_judge": {
                    "rubric_results": [{}], "gardener_score": score, "inspiration_score": score,
                    "gardener_dimensions": dimensions, "inspiration_dimensions": dimensions,
                },
                "gardener": dict(surface), "inspiration": dict(surface),
            })
        summary = summarize(rows)
        self.assertEqual(summary["gardener_infrastructure_failures"], 1)
        self.assertEqual(summary["gardener_capability_generation_failures"], 0)
        self.assertEqual(summary["gardener"]["capability_scorable"], 1)
        self.assertEqual(summary["gardener"]["average_hard_gated_score"], 80.0)
        self.assertEqual(summary["gardener"]["dimension_averages"]["correctness"], 5.0)
        self.assertEqual(summary["gardener"]["dimension_denominators"]["correctness"], 1)

    def test_generation_and_symbolic_failures_override_judge_scores(self):
        row = {
            "gardener": {"generation_failed": True, "symbolic_grounding": {}},
            "inspiration": {"generation_failed": False, "symbolic_grounding": {"applicable": True, "passed": False}},
        }
        judged = {
            "gardener_dimensions": {key: 5 for key in ("correctness", "derivation_rigor", "mechanism_discrimination", "uncertainty_calibration", "naturalness", "followup_value")},
            "inspiration_dimensions": {key: 5 for key in ("correctness", "derivation_rigor", "mechanism_discrimination", "uncertainty_calibration", "naturalness", "followup_value")},
            "gardener_verdict": "pass", "inspiration_verdict": "pass",
            "gardener_score": 100.0, "inspiration_score": 100.0,
            "failures": {"gardener": [], "inspiration": []},
        }
        result = enforce_hard_failure_gates(row, judged)
        self.assertEqual(result["gardener_score"], 0.0)
        self.assertEqual(result["gardener_verdict"], "fail")
        self.assertEqual(result["inspiration_score"], 50.0)
        self.assertEqual(result["inspiration_dimensions"]["correctness"], 1)
        self.assertIn("SYMBOLIC_GROUNDING_FAILED_HARD_GATE", result["failures"]["inspiration"])

    def test_infrastructure_failure_is_unscorable_not_capability_zero(self):
        row = {
            "gardener": {"generation_failed": True, "infrastructure_failure": {"detected": True, "category": "credential"}},
            "inspiration": {"generation_failed": False, "symbolic_grounding": {}},
        }
        judged = {
            "gardener_dimensions": {"correctness": 1}, "inspiration_dimensions": {"correctness": 5},
            "gardener_verdict": "fail", "inspiration_verdict": "pass",
            "gardener_score": 0.0, "inspiration_score": 100.0,
            "failures": {"gardener": [], "inspiration": []},
        }
        result = enforce_hard_failure_gates(row, judged)
        self.assertIsNone(result["gardener_score"])
        self.assertEqual(result["gardener_verdict"], "unscorable")
        self.assertEqual(result["gardener_dimensions"], {})
        self.assertIn("INFRASTRUCTURE_FAILURE_UNSCORABLE:credential", result["failures"]["gardener"])

    def test_unverified_tool_execution_caps_execution_rubrics_deterministically(self):
        row = {
            "requires_tool_execution": True,
            "scoring_rubric": [
                {"id": "R1", "criterion": "原理推导", "weight": 0.2},
                {"id": "R2", "criterion": "提供可执行代码或工具调用", "weight": 0.2},
                {"id": "R3", "criterion": "给出结果验证与实际执行结果", "weight": 0.2},
                {"id": "R4", "criterion": "理论反思", "weight": 0.2},
                {"id": "R5", "criterion": "完整性", "weight": 0.2},
            ],
            "gardener": {"generation_failed": False, "symbolic_grounding": {}, "local_checks": {"tool_execution_verified": False}},
            "inspiration": {"generation_failed": False, "symbolic_grounding": {}, "local_checks": {"tool_execution_verified": False}},
        }
        judged = {
            "rubric_results": [
                {"rubric_id": f"R{i}", "gardener_score": 2, "inspiration_score": 2}
                for i in range(1, 6)
            ],
            "gardener_dimensions": {key: 5 for key in ("correctness",)},
            "inspiration_dimensions": {key: 5 for key in ("correctness",)},
            "gardener_verdict": "pass", "inspiration_verdict": "pass",
            "gardener_score": 100.0, "inspiration_score": 100.0,
            "failures": {"gardener": [], "inspiration": []},
        }
        result = enforce_hard_failure_gates(row, judged)
        self.assertEqual(result["gardener_score"], 60.0)
        self.assertEqual(result["inspiration_score"], 60.0)
        self.assertEqual(result["rubric_results"][1]["gardener_score"], 1.0)
        self.assertEqual(result["rubric_results"][2]["inspiration_score"], 1.0)
        self.assertEqual(result["gardener_verdict"], "fail")
        self.assertIn("TOOL_EXECUTION_NOT_VERIFIED_HARD_GATE", result["failures"]["gardener"])

    def test_actual_sandbox_failure_caps_score_and_correctness(self):
        row = {
            "requires_tool_execution": True,
            "scoring_rubric": [{"id": "R1", "criterion": "实际执行结果", "weight": 1.0}],
            "gardener": {"generation_failed": False, "symbolic_grounding": {}, "local_checks": {
                "tool_execution_verified": False,
                "tool_execution": {"executed": True, "status": "failed", "reason": "timeout"},
            }},
            "inspiration": {"generation_failed": False, "symbolic_grounding": {}, "local_checks": {
                "tool_execution_verified": True,
            }},
        }
        judged = {
            "rubric_results": [{"rubric_id": "R1", "gardener_score": 2, "inspiration_score": 2}],
            "gardener_dimensions": {"correctness": 5},
            "inspiration_dimensions": {"correctness": 5},
            "gardener_verdict": "pass", "inspiration_verdict": "pass",
            "gardener_score": 100.0, "inspiration_score": 100.0,
            "failures": {"gardener": [], "inspiration": []},
        }
        result = enforce_hard_failure_gates(row, judged)
        self.assertEqual(result["gardener_score"], 50.0)
        self.assertEqual(result["gardener_dimensions"]["correctness"], 1)
        self.assertIn("SANDBOX_EXECUTION_FAILED:timeout", result["failures"]["gardener"])

    def test_math_oracle_catches_independently_audited_failures(self):
        samples = {
            "SCI-MATH-02": "```python\nprint('x')\n``` n ∈ {1,2,3,5,7,9,15} 时阶数确实决定群结构",
            "SCI-MATH-03": "```python\nimport numpy as np\nv=np.ones(3)\nprint(np.linalg.norm(v))\n``` 双曲面归一化使用 np.linalg.norm(v)",
            "SCI-MATH-08": "```python\ndef f(default_prec=60):\n return tau[m*n] + Delta*x\n``` Delta*... default_prec=60 tau[m*n]",
            "SCI-MATH-10": "```python\ntour=[]\ntour.append(tour)\n```",
        }
        expected = {
            "SCI-MATH-02": "GROUP_ORDER_9_FALSE_UNIQUENESS_CLAIM",
            "SCI-MATH-03": "HYPERBOLIC_TANGENT_USES_EUCLIDEAN_NORM",
            "SCI-MATH-08": "TAU_COEFFICIENT_RANGE_CAN_EXCEED_PRECISION",
            "SCI-MATH-10": "SELF_REFERENTIAL_TOUR_APPEND",
        }
        for case_id, answer in samples.items():
            with self.subTest(case=case_id):
                oracle = deterministic_answer_oracle(case_id, answer)
                self.assertFalse(oracle["passed"])
                self.assertIn(expected[case_id], oracle["issues"])

    def test_math_oracle_catches_geodesic_branch_and_inner_product_defects(self):
        answer = """```python
res = minimize_scalar(dist, bounds=(1e-6, 3*np.pi), method='bounded')
v = B + np.dot(A, B) * A  # 闵氏投影
```"""
        result = deterministic_answer_oracle("SCI-MATH-03", answer)
        self.assertIn(
            "SPHERE_SHOOTING_INTERVAL_INCLUDES_NONSHORTEST_BRANCHES",
            result["issues"],
        )
        self.assertIn(
            "HYPERBOLIC_PROJECTION_USES_EUCLIDEAN_DOT",
            result["issues"],
        )

    def test_summary_reports_actual_judge_models(self):
        row = {
            "generator_model": "glm-5.2", "generator_base_host": "open.bigmodel.cn",
            "requires_tool_execution": True,
            "auxiliary_judge": {"rubric_results": [{}], "model": "kimi-k2.6", "gardener_dimensions": {}, "inspiration_dimensions": {}},
            "gardener": {"latency_ms": 1, "generation_failed": False, "agent_trace": [], "local_checks": {"not_defensive_refusal": True}, "symbolic_grounding": {}},
            "inspiration": {"latency_ms": 1, "generation_failed": False, "local_checks": {"not_defensive_refusal": True}, "symbolic_grounding": {}},
        }
        summary = summarize([row])
        self.assertEqual(summary["judge_models"], ["kimi-k2.6"])
        self.assertEqual(summary["generator_models"], ["glm-5.2"])
        self.assertEqual(summary["generator_base_hosts"], ["open.bigmodel.cn"])
        self.assertEqual(summary["tool_execution_required"], 1)
        self.assertEqual(summary["gardener"]["tool_execution_verified"], 0)
        self.assertEqual(summary["gardener"]["hard_gated_score_denominator"], 0)

    def test_blind_twenty_is_balanced_frozen_and_reference_grounded(self):
        cases = load_cases(self.BLIND_DATASET)
        self.assertEqual(len(cases), 20)
        self.assertEqual(Counter(case["discipline"] for case in cases), {
            "数学": 4, "理论物理": 4, "物理化学": 4, "理论计算机": 4, "数理生物": 4,
        })
        for case in cases:
            with self.subTest(case=case["id"]):
                rubric, digest = freeze_rubric(case)
                self.assertGreaterEqual(len(rubric), 3)
                self.assertLessEqual(len(rubric), 5)
                self.assertEqual(len(digest), 64)
                if case.get("symbolic_checks"):
                    formulas = []
                    for check in case["symbolic_checks"]:
                        relation = check.get("relation", "=")
                        formulas.append(f"${check['target_lhs']}{relation}{check['rhs']}$")
                    grounding = answer_symbolic_grounding(case, "\n".join(formulas))
                    self.assertTrue(grounding["passed"], grounding)

    def test_science_exploration_pack_has_ten_balanced_domains_and_frozen_requirements(self):
        cases = load_cases(self.SCIENCE_100_DATASET)
        self.assertEqual(len(cases), 100)
        counts = Counter(case["discipline"] for case in cases)
        self.assertEqual(len(counts), 10)
        self.assertEqual(set(counts.values()), {10})
        for raw_case in cases:
            case = normalize_capability_case(raw_case)
            with self.subTest(case=case["id"]):
                self.assertEqual(case["expected_route"], "prepare_closed_loop")
                self.assertTrue(case["requires_tool_execution"])
                self.assertEqual(len(case["rubric"]), 5)
                self.assertEqual(len(freeze_rubric(case)[1]), 64)


if __name__ == "__main__":
    unittest.main()
