from __future__ import annotations

import json
import unittest
from pathlib import Path

from evals.adapter import load_cases
from core.reasoning_capability import classify_reasoning_task, is_self_contained_reasoning
from evals.advanced_learning_eval import (
    enforce_hard_assertion_verdict,
    frontier_agent_payload,
    frontier_judge_payload,
    frontier_judge_prompt,
    frontier_judge_task_type,
    hard_assertion_instruction,
    run_hard_assertions,
)
from evals.generate_advanced_learning_packs import build_frontier, build_hard
from evals.structural_debug import (
    PACK,
    assert_phase_allowed,
    audit_pack,
    build_agent_prompt,
    load_pack,
    merge_case_result,
    observable_checks,
    prompt_is_leak_free,
    public_case,
    summarize,
)
from evals.zhili_structural_benchmark import observable_checks as legacy_observable_checks
from evals.structural_judge import (
    judge_payload,
    judge_system_prompt,
    select_rows,
    summarize as judge_summary,
    valid_judge,
)


class StructuralDebugPackTests(unittest.TestCase):
    def test_advanced_learning_pack_counts_and_freeze_policy(self):
        hard_dev, hard_val, hard_challenge = build_hard()
        frontier_dev, frontier_val = build_frontier()
        self.assertEqual((len(hard_dev), len(hard_val), len(hard_challenge)), (30, 20, 10))
        self.assertEqual((len(frontier_dev), len(frontier_val)), (20, 10))
        self.assertTrue(all(not row["frozen"] for row in hard_dev + frontier_dev))
        self.assertTrue(all(row["frozen"] for row in hard_val + hard_challenge + frontier_val))

    def test_frontier_agent_and_judge_use_positive_field_whitelists(self):
        frontier_dev, _ = build_frontier()
        case = {
            **frontier_dev[0],
            "answer": "被测回答",
            "curator_inference": ["__PRIVATE_CURATOR_SENTINEL__"],
            "reference": "__PRIVATE_REFERENCE_SENTINEL__",
            "common_failures": ["__PRIVATE_FAILURE_SENTINEL__"],
            "rule_target": "__PRIVATE_RULE_SENTINEL__",
        }
        agent_payload = frontier_agent_payload(case)
        self.assertEqual(set(agent_payload), {"question", "reading_brief"})
        judge_fields = {"case_id", "question", "reading_brief", "source_claims", "rubric", "answer"}
        safe_payload = frontier_judge_payload(case)
        self.assertEqual(set(safe_payload), judge_fields)
        self.assertEqual(set(judge_payload(case)), judge_fields)
        serialized = json.dumps(safe_payload, ensure_ascii=False)
        for sentinel in (
            "__PRIVATE_CURATOR_SENTINEL__",
            "__PRIVATE_REFERENCE_SENTINEL__",
            "__PRIVATE_FAILURE_SENTINEL__",
            "__PRIVATE_RULE_SENTINEL__",
        ):
            self.assertNotIn(sentinel, serialized)

    def test_frontier_router_selects_three_independent_scoring_prompts(self):
        cases = [
            {"question": "解释材料中的预测与机制为何不同。", "reasoning_structure_id": "FR02_PREDICTION_MECHANISM_GAP"},
            {"question": "比较离线基准与部署外推的边界。", "reasoning_structure_id": "FR05_BENCHMARK_DEPLOYMENT_GAP"},
            {"question": "设计一个实验，写出阴性对照、干预和失败判据。", "reasoning_structure_id": "FR02_INTERVENTION_PROXY_TEST"},
        ]
        self.assertEqual([frontier_judge_task_type(case) for case in cases], ["A", "E", "B"])
        prompts = [frontier_judge_prompt(case) for case in cases]
        self.assertEqual(len(set(prompts)), 3)
        self.assertIn("概念解析", prompts[0])
        self.assertIn("证据与边界", prompts[1])
        self.assertIn("实验设计", prompts[2])
        self.assertIn("实验设计", judge_system_prompt({**cases[2], "suite": "zhili_frontier_guided_reading_v1"}))
        frontier_dev, frontier_val = build_frontier()
        self.assertEqual(
            {frontier_judge_task_type(case) for case in frontier_dev + frontier_val},
            {"A", "E", "B"},
        )

    def test_hc02_and_hc10_hard_assertions_precede_semantic_verdict(self):
        _, _, challenge = build_hard()
        cases = {case["structure_id"]: case for case in challenge}
        self.assertEqual(
            {key for key, case in cases.items() if case.get("hard_assertion")},
            {"HC02", "HC10"},
        )
        hc02_payload = {
            "curves": [
                {"center": [0, 0], "radius": 1, "orientation": "ccw"},
                {"center": [3, 0], "radius": 1, "orientation": "ccw"},
            ],
            "claimed_lengths": [2 * 3.141592653589793, 2 * 3.141592653589793],
            "claimed_winding_numbers": [1, 0],
        }
        hc02_answer = (
            "两条圆均为简单闭曲线，长度相同；绕数在穿孔平面的同伦下不变。"
            f"<hard_assertion>{json.dumps(hc02_payload)}</hard_assertion>"
        )
        self.assertTrue(run_hard_assertions(cases["HC02"], hc02_answer)["passed"])

        hc10_payload = {
            "rho": [
                [[0.7, 0], [0.3, 0.1]],
                [[0.3, -0.1], [0.3, 0]],
            ],
        }
        hc10_answer = (
            "由 Bloch 表示重建矩阵，并检查 Hermitian、迹与特征值。"
            f"<hard_assertion>{json.dumps(hc10_payload)}</hard_assertion>"
        )
        self.assertTrue(run_hard_assertions(cases["HC10"], hc10_answer)["passed"])
        self.assertIn("<hard_assertion>", hard_assertion_instruction(cases["HC10"]))

        invalid_payload = {
            "rho": [
                [[0.7, 0], [0.3, -0.1]],
                [[0.3, 0.1], [0.3, 0]],
            ],
        }
        invalid_answer = f"<hard_assertion>{json.dumps(invalid_payload)}</hard_assertion>"
        hard_result = run_hard_assertions(cases["HC10"], invalid_answer)
        self.assertFalse(hard_result["passed"])
        final = enforce_hard_assertion_verdict(
            {"verdict": "pass", "first_material_error": ""},
            hard_result,
        )
        self.assertEqual(final["semantic_verdict_before_hard_assertion"], "pass")
        self.assertEqual(final["verdict"], "fail")
        self.assertTrue(final["hard_assertion_override"])

    def test_advanced_learning_pack_has_atomic_scoring_and_primary_sources(self):
        hard_dev, hard_val, hard_challenge = build_hard()
        frontier_dev, frontier_val = build_frontier()
        rows = hard_dev + hard_val + hard_challenge + frontier_dev + frontier_val
        self.assertEqual(len({row["id"] for row in rows}), len(rows))
        self.assertTrue(all(row.get("scoring_rubric", {}).get("required_claims") for row in rows))
        self.assertTrue(all(row.get("scoring_rubric", {}).get("fatal_errors") for row in rows))
        self.assertTrue(all(row.get("source_url", "").startswith("https://") for row in frontier_dev + frontier_val))
        self.assertEqual(len({row["structure_id"] for row in frontier_dev}), 10)
        # Exact structures may intentionally recur across different concepts:
        # that surface-different/same-structure pairing is part of the transfer
        # design. The bank must still be far richer than the former A/E pair.
        self.assertGreaterEqual(len({row["reasoning_structure_id"] for row in frontier_dev}), 15)
        family_counts = {
            family: sum(row["reasoning_family_id"] == family for row in frontier_dev)
            for family in {row["reasoning_family_id"] for row in frontier_dev}
        }
        self.assertEqual(set(family_counts.values()), {2})
        self.assertTrue(all(row["scoring_rubric"]["required_inference_links"] for row in rows))
        self.assertFalse(any(
            row["scoring_rubric"]["required_inference_links"] == ["结论必须由题中条件或给定材料推出"]
            for row in rows
        ))

    def test_advanced_frozen_sources_are_disjoint_and_challenges_have_no_polarity_shortcut(self):
        _, hard_val, hard_challenge = build_hard()
        frontier_dev, frontier_val = build_frontier()
        self.assertTrue(
            {row["source_url"] for row in frontier_dev}.isdisjoint(
                {row["source_url"] for row in frontier_val}
            )
        )
        self.assertTrue(
            {row["reading_brief"] for row in frontier_dev}.isdisjoint(
                {row["reading_brief"] for row in frontier_val}
            )
        )
        self.assertTrue(all(row["transfer_mode"] == "different_source" for row in frontier_val))
        self.assertEqual(
            {row["structure_id"] for row in frontier_dev},
            {row["structure_id"] for row in frontier_val},
        )
        non_binary = [
            row for row in hard_challenge
            if "能否" not in row["question"] and "是否" not in row["question"]
        ]
        self.assertGreaterEqual(len(non_binary), 6)
        self.assertTrue(any("构造" in row["question"] for row in hard_challenge))
        self.assertTrue(any("重建密度矩阵" in row["question"] for row in hard_challenge))
        hf08 = next(row for row in hard_val if row["id"] == "ZHV-HF08-1")
        self.assertEqual(hf08["scoring_rubric"]["minimum_required_claims"], 3)

    def test_pack_has_96_development_32_validation_and_12_challenge_cases(self):
        pack = load_pack()
        self.assertEqual({key: len(rows) for key, rows in pack.items()}, {
            "development": 96,
            "transfer_validation": 32,
            "author_visible_challenge": 12,
        })
        self.assertTrue(audit_pack(pack)["passed"])

    def test_each_development_structure_has_exactly_a_to_f(self):
        rows = load_cases(PACK / "development_96.jsonl")
        groups = {}
        for row in rows:
            groups.setdefault(row["structure_id"], set()).add(row["variant"])
        self.assertEqual(len(groups), 16)
        self.assertTrue(all(variants == set("ABCDEF") for variants in groups.values()))

    def test_validation_reuses_structures_but_not_questions(self):
        pack = load_pack()
        development = pack["development"]
        validation = pack["transfer_validation"]
        self.assertEqual(
            {row["structure_id"] for row in development},
            {row["structure_id"] for row in validation},
        )
        self.assertTrue(all(row["frozen"] for row in validation))
        self.assertTrue(
            {row["question"] for row in development}.isdisjoint({row["question"] for row in validation})
        )

    def test_author_visible_challenge_uses_disjoint_structures(self):
        pack = load_pack()
        seen = {
            row["structure_id"]
            for split in ("development", "transfer_validation")
            for row in pack[split]
        }
        challenge = {row["structure_id"] for row in pack["author_visible_challenge"]}
        self.assertTrue(seen.isdisjoint(challenge))
        self.assertFalse(audit_pack(pack)["strict_blind"])

    def test_phase_guard_blocks_using_frozen_rows_as_development(self):
        validation = load_pack()["transfer_validation"][:1]
        with self.assertRaisesRegex(ValueError, "phase=develop"):
            assert_phase_allowed(validation, "develop")
        assert_phase_allowed(validation, "validate")

    def test_only_public_fields_are_passed_to_agent(self):
        case = load_pack()["development"][0]
        visible = public_case(case)
        self.assertEqual(set(visible), {"id", "question", "structure_group", "variant"})
        self.assertNotIn("reference", visible)
        self.assertNotIn("common_failures", visible)
        self.assertNotIn("rule_target", visible)

    def test_runtime_result_cannot_erase_post_answer_judge_fields(self):
        case = load_pack()["development"][0]
        row = merge_case_result(case, {"answer": "实质回答", "reference": "", "common_failures": []})
        self.assertEqual(row["answer"], "实质回答")
        self.assertEqual(row["reference"], case["reference"])
        self.assertEqual(row["common_failures"], case["common_failures"])

    def test_prompts_do_not_leak_reference_or_failure_labels(self):
        for phase, split in (
            ("develop", "development"),
            ("validate", "transfer_validation"),
            ("challenge", "author_visible_challenge"),
        ):
            for case in load_pack()[split]:
                prompt = build_agent_prompt(case, phase=phase, rules="先检查必要条件，再选择方法。")
                self.assertTrue(prompt_is_leak_free(prompt, case), case["id"])

    def test_all_development_prompts_route_as_closed_without_weakening_freshness_guard(self):
        for case in load_pack()["development"]:
            prompt = build_agent_prompt(case, phase="develop", rules="当前事实与最新实验仍需查证。")
            profile = classify_reasoning_task(prompt)
            self.assertTrue(profile["activated"], case["id"])
            self.assertTrue(is_self_contained_reasoning(prompt, profile), case["id"])
        current = build_agent_prompt(
            {"question": "请查截至今天的最新实验结果并给出论文来源。"},
            phase="develop",
        )
        current_profile = classify_reasoning_task(current)
        self.assertFalse(is_self_contained_reasoning(current, current_profile))

    def test_every_case_has_non_null_failure_labels_and_reference(self):
        rows = [row for values in load_pack().values() for row in values]
        self.assertTrue(all(row["reference"].strip() for row in rows))
        self.assertTrue(all(row["common_failures"] for row in rows))
        self.assertTrue(all(all(item for item in row["common_failures"]) for row in rows))

    def test_local_observable_checks_do_not_claim_semantic_correctness(self):
        case = load_pack()["development"][0]
        checks = observable_checks(
            case,
            "不能直接推出。因为前提只覆盖各个方向，而可微还需要统一的线性近似条件；"
            "因此应构造反例并分别验证方向导数存在与连续性失败。",
        )
        self.assertIsNone(checks["semantic_score"])
        self.assertIn("不代表答案语义正确", checks["note"])

    def test_local_observable_checks_accept_generic_missing_condition_wording(self):
        answer = (
            "不能仅凭当前两次测量就唯一判定机制，因为多个模型会给出同样的观测。"
            "最小缺失信息是独立控制变量下的响应曲线；若补上该信息，则可以区分这些解释。"
            "在该信息到来之前，只能列出条件性结论和各模型对应的可证伪预测，不能选定唯一答案。"
        )
        checks = observable_checks({"variant": "C"}, answer)
        self.assertTrue(checks["variant_requirement"])
        self.assertTrue(checks["passed"])

    def test_local_observable_checks_accept_generic_wrong_premise_wording(self):
        answer = (
            "该式并非恒等式，原结论不能推出；它只在额外的对称条件成立时有效。"
            "题目把充分条件和必要条件的方向写反了，因此应先撤销这个过强前提再推导。"
            "取一个不满足对称性的具体对象即可构成反例，也能说明正确命题必须明确限定适用范围。"
        )
        checks = observable_checks({"variant": "D"}, answer)
        self.assertTrue(checks["variant_requirement"])
        self.assertTrue(checks["passed"])

    def test_local_observable_checks_accept_generic_error_correction_wording(self):
        answer = (
            "最小错误在于把相关性当成因果性，所以原论证无效，不能推出所声称的机制。"
            "有效改写是把结论降为相关关系，并补充随机干预这一条件；也可用反例说明原说法过强。"
            "修正后的论证还应报告混杂因素和替代解释，使结论强度与现有证据保持一致。"
        )
        checks = observable_checks({"variant": "F"}, answer)
        self.assertTrue(checks["variant_requirement"])
        self.assertTrue(checks["passed"])

    def test_local_observable_variant_aliases_cover_inference_language(self):
        answers = {
            "C": "现有观测不能推出参数唯一，因为无从确定是否存在等价参数族；最小缺失信息是模型的秩条件。",
            "E": "仅凭一次稳定返回无法证明唯一，缺少全局结构和多初值检验条件，因此证据不足。",
            "F": "核心错误是把未显著当成无效应；更严谨的表述应报告区间并说明当前实验不能排除哪些效应。",
        }
        for variant, answer in answers.items():
            with self.subTest(variant=variant):
                self.assertTrue(observable_checks({"variant": variant}, answer)["variant_requirement"])

    def test_development_summary_cannot_be_labeled_generalization(self):
        summary = summarize([], phase="develop")
        self.assertIn("不代表泛化", summary["generalization_claim"])
        challenge = summarize([], phase="challenge")
        self.assertIn("不是严格盲测", challenge["generalization_claim"])

    def test_infrastructure_failures_are_excluded_from_capability_denominator(self):
        row = {
            "id": "infra",
            "structure_id": "S02",
            "answer": "这次自足推理模型没有返回可解析的实质答案。请重试本题。",
            "agent_trace": [{
                "node": "generate_answer",
                "data": {"generation_error": "Error code: 402 - quota exhausted"},
            }],
            "observable_checks": {"passed": False},
        }
        from evals.structural_debug import classify_execution

        row["execution"] = classify_execution(row)
        self.assertEqual(row["execution"]["status"], "infrastructure_failure")
        self.assertFalse(row["execution"]["scorable"])
        summary = summarize([row], phase="develop")
        self.assertEqual(summary["scorable_cases"], 0)
        self.assertEqual(summary["observable_pass_rate"], None)
        self.assertEqual(summary["failure_ids"], [])
        self.assertEqual(summary["unscorable_ids"], ["infra"])
        self.assertFalse(summary["score_valid"])

    def test_recovered_retry_with_substantive_answer_remains_scorable(self):
        from evals.structural_debug import classify_execution

        row = {
            "answer": (
                "原命题不能仅由一个式子推出。首先需要补充相对性原理；"
                "若这些前提成立，则可导出对应变换，并应明确适用边界和参考系。"
            ),
            "agent_trace": [{
                "node": "generate_answer",
                "data": {"generation_error": "first attempt timed out"},
            }],
        }
        execution = classify_execution(row)
        self.assertEqual(execution["status"], "completed")
        self.assertTrue(execution["scorable"])

    def test_independent_judge_skips_explicitly_unscorable_rows(self):
        rows = [
            {
                "id": "infra",
                "execution": {"scorable": False},
                "observable_checks": {"passed": False},
            },
            {
                "id": "answer",
                "execution": {"scorable": True},
                "observable_checks": {"passed": False},
            },
        ]
        self.assertEqual(
            [row["id"] for row in select_rows(rows, only_local_failures=True)],
            ["answer"],
        )

    def test_frozen_qualification_requires_matching_complete_development_run(self):
        import json
        import tempfile
        from core.config import LLMConfig
        from evals.structural_debug import qualification_audit

        payload = {
            "summary": {
                "phase": "develop",
                "benchmark": "suite",
                "cases": 4,
                "scorable_cases": 4,
                "score_valid": True,
                "run_contract": {
                    "complete_dataset_run": True,
                    "rules_sha256": "rules",
                    "tested_model": "glm-5.2",
                    "tested_base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
                },
            },
            "rows": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "develop.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            passed = qualification_audit(
                path,
                benchmark="suite",
                rules_sha256="rules",
                tested_config=LLMConfig(
                    "secret",
                    "https://open.bigmodel.cn/api/coding/paas/v4",
                    "glm-5.2",
                ),
            )
            changed_model = qualification_audit(
                path,
                benchmark="suite",
                rules_sha256="rules",
                tested_config=LLMConfig(
                    "secret",
                    "https://open.bigmodel.cn/api/coding/paas/v4",
                    "other-model",
                ),
            )
        self.assertTrue(passed["passed"])
        self.assertFalse(changed_model["passed"])
        self.assertTrue(any("模型" in reason for reason in changed_model["reasons"]))

    def test_prompt_pack_contains_full_debug_lifecycle(self):
        prompt_path = Path(__file__).resolve().parent.parent / "evals" / "prompts" / "structural_debug_prompts.md"
        text = prompt_path.read_text(encoding="utf-8")
        for marker in ("无规则基线", "失败归因", "通用规则归纳", "规则去过拟合审查", "回归测试选择", "冻结迁移运行", "污染审计", "停止条件"):
            self.assertIn(marker, text)

    def test_legacy_missing_condition_check_accepts_missing_link_wording(self):
        case = {"variant": "C"}
        answer = (
            "学生的推理缺失了不同惯性系之间的变换规则。麦克斯韦方程只给出选定参考系中的特征速度，"
            "不能单独推出所有惯性系测得相同。这里还需要相对性原理或洛伦兹协变性的物理假设，"
            "因此原结论并不完整，若采用不同变换规则，结论也会变化。"
        )
        checks = legacy_observable_checks(case, {"answer": answer, "reasoning": {"type": "argument_analysis", "self_contained": True}})
        self.assertTrue(checks["checks"]["states_condition_or_boundary"])
        self.assertTrue(checks["checks"]["variant_requirement"])

    def test_independent_judge_only_receives_post_answer_fields(self):
        row = {
            "id": "x", "question": "q", "reference": "r", "common_failures": ["f"],
            "answer": "a", "agent_trace": ["private"], "rule_target": "private-rule",
        }
        self.assertEqual(set(judge_payload(row)), {"id", "question", "reference", "common_failures", "answer"})

    def test_independent_judge_can_select_only_local_failures(self):
        rows = [
            {"id": "ok", "observable_checks": {"passed": True}},
            {"id": "bad", "observable_checks": {"passed": False}},
        ]
        self.assertEqual([row["id"] for row in select_rows(rows, only_local_failures=True)], ["bad"])

    def test_independent_judge_schema_is_bounded(self):
        result = {
            "structure_identification": 2, "premise_check": 2, "method_correctness": 1,
            "derivation_completeness": 1, "boundary_calibration": 2,
            "verdict": "warn", "first_material_error": "少一步验证",
        }
        self.assertTrue(valid_judge(result))
        result["method_correctness"] = 3
        self.assertFalse(valid_judge(result))

    def test_independent_judge_allows_pass_without_material_error(self):
        result = {
            "structure_identification": 2, "premise_check": 2, "method_correctness": 2,
            "derivation_completeness": 2, "boundary_calibration": 2,
            "verdict": "pass", "first_material_error": None,
        }
        self.assertTrue(valid_judge(result))

    def test_independent_judge_reports_actual_fallback_model(self):
        rows = [{"judge": {
            "structure_identification": 2, "premise_check": 2, "method_correctness": 2,
            "derivation_completeness": 2, "boundary_calibration": 2,
            "verdict": "pass", "judge_model_used": "deepseek-v4-pro-202606",
        }}]
        self.assertEqual(judge_summary(rows, "deepseek-v4-flash-202605")["models_used"], ["deepseek-v4-pro-202606"])


if __name__ == "__main__":
    unittest.main()
