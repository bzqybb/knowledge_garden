from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.gardener_graph import generate_answer
from core.llm import LLMError
from core.inspiration import explore_inspiration
from core.learning_memory import LearningMemoryService
from core.reasoning_capability import (
    CATEGORY_ALIASES,
    classify_reasoning_task,
    is_self_contained_reasoning,
    reasoning_subject,
    reasoning_prompt,
    review_reasoning_answer,
)
from core.storage import GardenStore
from evals.adapter import load_cases
from evals.general_reasoning_benchmark import summarize, validate_dataset


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "evals" / "datasets" / "general_reasoning_15_v1.jsonl"


class ReasoningCapabilityTests(unittest.TestCase):
    def test_user_reasoning_dataset_preserves_all_cases_and_rubrics(self):
        cases = load_cases(DATASET)
        self.assertEqual(len(cases), 15)
        self.assertEqual(len({case["id"] for case in cases}), 15)
        self.assertTrue(all(case.get("reference") for case in cases))
        self.assertTrue(all(case.get("reasoning_trace") for case in cases))
        self.assertTrue(all(
            abs(sum(float(item["weight"]) for item in case["evaluation_points"]) - 1.0) < 1e-9
            for case in cases
        ))

    def test_router_matches_every_seed_benchmark_category(self):
        cases = load_cases(DATASET)
        failures = []
        for case in cases:
            expected = CATEGORY_ALIASES[case["category"]]
            result = classify_reasoning_task(case["question"])
            if not result["activated"] or result["key"] != expected:
                failures.append((case["id"], expected, result["key"], result["score"]))
        self.assertEqual(failures, [])

    def test_self_contained_reasoning_does_not_bypass_current_fact_lookup(self):
        closed = classify_reasoning_task("甲乙丙中恰有一人说真话，请穷举谁偷了文件。")
        current = classify_reasoning_task("请查截至今天官方公布的真实失业率，并给出处。")
        self.assertTrue(is_self_contained_reasoning("甲乙丙中恰有一人说真话，请穷举谁偷了文件。", closed))
        self.assertFalse(is_self_contained_reasoning("请查截至今天官方公布的真实失业率，并给出处。", current))

    def test_structural_wrapper_does_not_turn_closed_question_into_current_fact_lookup(self):
        wrapped = (
            "【致理结构调试·develop】\n当前冻结的通用规则：最新实验另行查证，论文来源必须核对。"
            "\n\n题目：\n所有方向导数存在能否推出二元函数可微？请证明或反驳。"
        )
        profile = classify_reasoning_task(wrapped)
        self.assertEqual(reasoning_subject(wrapped), "所有方向导数存在能否推出二元函数可微？请证明或反驳。")
        self.assertIn(profile["key"], {"mathematical_proof", "argument_analysis"})
        self.assertTrue(is_self_contained_reasoning(wrapped, profile))

        current = wrapped.rsplit("题目：", 1)[0] + "题目：\n请查截至今天最新实验结果并给论文来源。"
        current_profile = classify_reasoning_task(current)
        self.assertFalse(is_self_contained_reasoning(current, current_profile))

    def test_theoretical_derivation_and_supplied_claim_evaluation_are_self_contained(self):
        derivation = classify_reasoning_task("从真空中的基本场方程推导电场波动方程，并列出成立条件。")
        implication = classify_reasoning_task("恒温恒压下某判据成立，是否意味着过程一定很快？")
        false_premise = classify_reasoning_task("前提：某处理会改变状态函数。请判断正误并解释。")
        self.assertEqual(derivation["key"], "physical_modelling")
        self.assertTrue(derivation["activated"])
        self.assertEqual(implication["key"], "argument_analysis")
        self.assertTrue(false_premise["activated"])
        self.assertTrue(is_self_contained_reasoning("从真空中的基本场方程推导电场波动方程，并列出成立条件。", derivation))
        self.assertTrue(is_self_contained_reasoning("恒温恒压下某判据成立，是否意味着过程一定很快？", implication))
        self.assertTrue(is_self_contained_reasoning("前提：某处理会改变状态函数。请判断正误并解释。", false_premise))

        current = classify_reasoning_task("请查截至今天某实验是否已经成功，并给出论文来源。")
        self.assertFalse(is_self_contained_reasoning("请查截至今天某实验是否已经成功，并给出论文来源。", current))

    def test_personal_choice_and_research_question_route_without_external_fact_lookup(self):
        choice = classify_reasoning_task("我不知道该选数学还是生物，你能判断我最适合哪个方向吗？")
        research = classify_reasoning_task("请评价这个想法，并把它转化为可研究的问题。")
        self.assertEqual(choice["key"], "decision_analysis")
        self.assertEqual(research["key"], "experimental_design")
        self.assertTrue(is_self_contained_reasoning("我不知道该选数学还是生物，你能判断我最适合哪个方向吗？", choice))
        self.assertTrue(is_self_contained_reasoning("请评价这个想法，并把它转化为可研究的问题。", research))

    def test_self_contained_answer_retries_invalid_json_once(self):
        question = "证明一个命题成立的充要条件。"
        profile = classify_reasoning_task(question)
        state = {
            "question": question,
            "dialogue": "",
            "intent": {"primary_intent": "apply", "concepts": [], "query_plan": {}},
            "reasoning_profile": profile,
            "accepted_sources": [],
            "evidence_review": {"sufficient": False, "source_roles": {}, "usable_claims": [], "gaps": []},
            "teaching_strategy": {"preference_directives": []},
            "retrieval_errors": [],
        }
        recovered = {
            "answer": "若条件 A 成立，则逐步推出 B；反向由 B 推回 A，因此两个方向均成立。该结论以题设定义域为条件。\\boxed{A\\iff B}",
            "followup": "要检查一个反例吗？", "discussion_prompts": [],
        }
        with patch("core.gardener_graph._agent_json", side_effect=[LLMError("invalid json"), recovered]) as mocked:
            result = generate_answer(state)
        self.assertEqual(mocked.call_count, 2)
        self.assertIn("两个方向", result["answer"])
        self.assertEqual(result["trace"][-1]["data"]["generation_provider"], "project-model-self-contained-json-retry")

    def test_self_contained_answer_ignores_irrelevant_sources_and_retries_empty_payload(self):
        question = "在均匀线性介质中推导波速，并说明何时不能写成固定的 1/√(με)。"
        profile = classify_reasoning_task(question)
        state = {
            "question": question,
            "dialogue": "",
            "intent": {"primary_intent": "apply", "concepts": ["波速"], "query_plan": {}},
            "reasoning_profile": profile,
            "accepted_sources": [{
                "source_id": "L1",
                "title": "事实核验与信息甄别",
                "text": "与电磁推导无关的页面",
                "source_type": "local_wiki",
                "local": True,
            }],
            "evidence_review": {
                "sufficient": True,
                "source_roles": {"L1": "direct_evidence"},
                "usable_claims": ["无关论断"],
                "gaps": [],
            },
            "teaching_strategy": {"preference_directives": []},
            "retrieval_errors": [],
        }
        recovered = {
            "answer": (
                "由 Maxwell 方程取旋度并用本构关系可得波动方程，"
                "在线性、均匀、各向同性且无色散近似下有 "
                "\\boxed{v=1/\\sqrt{\\mu\\varepsilon}}。色散、耗散或非线性时不能把 μ、ε 当固定常数。"
            ),
            "followup": "",
            "discussion_prompts": [],
        }
        with patch("core.gardener_graph._agent_json", side_effect=[{}, recovered]) as mocked:
            result = generate_answer(state)
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(result["generation_sources"], [])
        self.assertIn("波动方程", result["answer"])
        self.assertNotIn("事实核验与信息甄别", result["answer"])

    def test_self_contained_timeout_retry_is_bounded_and_strips_debug_wrapper(self):
        task = "由麦克斯韦方程得到 c 后，能否单独推出洛伦兹变换？"
        wrapped = "【致理结构调试·develop】\n" + ("冗长调试规则；" * 600) + f"\n\n题目：\n{task}"
        state = {
            "question": wrapped,
            "dialogue": "",
            "intent": {"primary_intent": "apply", "concepts": ["麦克斯韦方程"], "query_plan": {}},
            "reasoning_profile": classify_reasoning_task(task),
            "accepted_sources": [],
            "evidence_review": {"sufficient": False, "source_roles": {}, "usable_claims": [], "gaps": []},
            "teaching_strategy": {"preference_directives": []},
            "retrieval_errors": [],
        }
        recovered = {
            "answer": "不能。得到一个特征速度不足以唯一推出坐标变换；还需相对性原理与光速不变假设。"
            "不同变换群可以保留不同结构，因此必须补足运动学公设。\\boxed{不能单独推出}",
            "followup": "",
            "discussion_prompts": [],
        }
        with patch("core.gardener_graph._agent_json", side_effect=[LLMError("timeout"), recovered]) as mocked:
            result = generate_answer(state)
        self.assertEqual(mocked.call_count, 2)
        retry_prompt = mocked.call_args_list[1].args[1]
        self.assertIn(task, retry_prompt)
        self.assertNotIn("冗长调试规则", retry_prompt)
        self.assertIn("不能单独推出", result["answer"])

        with patch(
            "core.gardener_graph._agent_json",
            side_effect=[LLMError("timeout"), LLMError("timeout again")],
        ) as mocked:
            failed = generate_answer(state)
        self.assertEqual(mocked.call_count, 2)
        self.assertIn("没有返回可解析", failed["answer"])

    def test_precision_prompts_cover_nonconvex_second_order_and_uq_types(self):
        examples = [
            (
                "只知优化器损失不再下降，判断是否最优。",
                "Hessian 半正定通常只是局部极小的二阶必要条件",
            ),
            (
                "模型预测新催化剂活性高但预测区间很宽，应如何安排实验？",
                "先区分模型认识不确定性、过程固有随机性和实验测量误差",
            ),
            (
                "只知优化器损失不再下降，判断是否最优。",
                "边界约束情形应使用 KKT",
            ),
            (
                "模型预测中等但区间窄，与上一候选如何比较？",
                "区间是否重叠不能直接替代差值检验",
            ),
        ]
        recovered = {
            "answer": "先区分条件，再给出可核验步骤与边界。该结论只在所列条件成立时有效。\\boxed{条件性结论}",
            "followup": "",
            "discussion_prompts": [],
        }
        for question, anchor in examples:
            with self.subTest(question=question):
                state = {
                    "question": question,
                    "dialogue": "",
                    "intent": {"primary_intent": "apply", "concepts": [], "query_plan": {}},
                    "reasoning_profile": classify_reasoning_task(question),
                    "accepted_sources": [],
                    "evidence_review": {
                        "sufficient": False, "source_roles": {}, "usable_claims": [], "gaps": [],
                    },
                    "teaching_strategy": {"preference_directives": []},
                    "retrieval_errors": [],
                }
                with patch("core.gardener_graph._agent_json", return_value=recovered) as mocked:
                    generate_answer(state)
                self.assertIn(anchor, mocked.call_args_list[0].args[1])

    def test_gardener_and_inspiration_share_reasoning_but_not_rigid_format(self):
        profile = classify_reasoning_task("两个数据库方案A和B应该如何选择？迁移成本不同。")
        gardener = reasoning_prompt(profile, surface="gardener_chat")
        inspiration = reasoning_prompt(profile, surface="inspiration")
        self.assertIn("可迁移推理协议", gardener)
        self.assertIn("\\boxed", gardener)
        self.assertIn("不机械套四个固定标题", inspiration)

    def test_observable_review_requires_steps_limits_and_box_for_gardener(self):
        profile = classify_reasoning_task("证明一个命题成立的充要条件。")
        weak = review_reasoning_answer(profile, "显然可得，所以成立。")
        strong = review_reasoning_answer(
            profile,
            "假设条件 C 成立，则由引理一推出 A⇒B；反向由 B 构造 C，因此两个方向均成立。"
            "该结论仅在给定数域与边界条件下成立，其他情形仍需反例检验。最终有 \\boxed{A\\iff B}。",
        )
        self.assertFalse(weak["passed"])
        self.assertTrue(strong["passed"])

    def test_validation_log_does_not_claim_semantic_accuracy(self):
        rows = validate_dataset(load_cases(DATASET))
        summary = summarize(rows, executed=False)
        self.assertEqual(summary["cases"], 15)
        self.assertEqual(summary["semantic_score"], None)
        self.assertIn("不冒充语义正确率", summary["semantic_score_note"])

    def test_self_contained_logic_question_bypasses_external_evidence_refusal(self):
        question = "甲乙丙中恰有一人说真话，请穷举谁偷了文件。"
        profile = classify_reasoning_task(question)
        model_answer = (
            "假设甲偷，则两句为真；若乙偷，仍有两句为真；若丙偷，则仅乙的话为真。"
            "因此只有第三种情况满足题设。该结论以恰有一人偷且恰有一句真话为前提。"
            "最终有 \\boxed{丙偷了文件}。"
        )
        state = {
            "question": question,
            "dialogue": "",
            "intent": {"primary_intent": "apply", "concepts": [], "query_plan": {}},
            "reasoning_profile": profile,
            "accepted_sources": [],
            "evidence_review": {"sufficient": False, "source_roles": {}, "usable_claims": [], "gaps": []},
            "teaching_strategy": {"preference_directives": []},
            "retrieval_errors": [],
        }
        with patch("core.gardener_graph._agent_json", return_value={
            "answer": model_answer, "followup": "是否要画真值表？", "discussion_prompts": [],
        }):
            result = generate_answer(state)
        self.assertEqual(result["answer"], model_answer)
        self.assertNotIn("证据不足", result["answer"])

    def test_inspiration_feedback_is_recalled_by_same_reasoning_type(self):
        with tempfile.TemporaryDirectory() as folder:
            store = GardenStore(Path(folder) / "garden.db")
            payload = {
                "primary_type": "open_exploration",
                "secondary_types": [],
                "answer": "可以先比较方案的可逆性，再设容量阈值；当前数据不足，所以结论应保持条件性。",
                "acknowledgement": "这个选择值得拆开看。",
                "assumptions": ["增长预测不确定"],
                "claims": [],
                "counter_view": "也要考虑迁移窗口。",
                "branches": [{"title": "测容量", "question": "A 的容量上限是多少？"}],
            }
            with patch("core.inspiration.search_notes", return_value=[]), patch(
                "core.inspiration.chat_json", return_value=payload,
            ):
                result = explore_inspiration(
                    store, "数据库方案A和B应该如何选择？迁移成本不同。",
                )
            feedback = LearningMemoryService(store).record_personalization_feedback(
                request_id=result["request_id"],
                helpful=True,
                feedback_note="同类决策题请先给条件分支，再给触发阈值。",
            )
            recalled = LearningMemoryService(store).active_memory_context(
                surface="inspiration",
                task_keys=[result["reasoning"]["task_key"]],
            )
            self.assertTrue(feedback["recorded"])
            self.assertTrue(any("触发阈值" in item["claim_text"] for item in recalled["claims"]))


if __name__ == "__main__":
    unittest.main()
