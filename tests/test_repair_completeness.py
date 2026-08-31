import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from core.gardener_graph import (
    DiagramSpec,
    QualityReview,
    _latex_syntax_review,
    prepare_closed_loop,
    repair_outputs,
    route_after_personalization,
)
from core.storage import GardenStore
from evals.boundary_eval import _run_turn


class RepairCompletenessTests(unittest.TestCase):
    def _state(self):
        original = (
            "由分布定义，题设等价于 $T_f''=0$。取磨光 $f_ε=f*η_ε$，则 $f_ε''=0$，"
            "所以 $f_ε=a_εx+b_ε$。令 ε→0，得到 $f=ax+b$ 几乎处处。"
        )
        return {
            "question": "设 f 局部可积且分布二阶导数为零，证明 f 几乎处处为仿射函数。",
            "answer": original,
            "initial_answer": original,
            "quality_review": {
                "repair_target": "text", "issues": ["结论需要装框"],
                "missing_rubric_ids": ["B3"], "critique": "补全极限步骤。",
            },
            "content_blueprint": {"required_claims": [
                {"id": "B1", "claim": "分布二阶导数为零"},
                {"id": "B2", "claim": "磨光后二阶导数为零"},
                {"id": "B3", "claim": "极限得到几乎处处仿射"},
            ]},
            "evidence_review": {"sufficient": False, "source_roles": {}},
            "intent": {"primary_intent": "explain_mechanism"},
            "reasoning_profile": {"activated": True, "key": "mathematical_proof"},
            "visualization": DiagramSpec(status="suppressed", kind="none").model_dump(),
            "trace": [],
        }

    def test_reflector_schema_cannot_emit_revised_answer(self):
        self.assertNotIn("revised_answer", QualityReview.model_fields)
        parsed = QualityReview.model_validate({"passed": False, "revised_answer": "只保留主体"})
        self.assertNotIn("revised_answer", parsed.model_dump())

    def test_meta_language_patch_is_rejected_and_original_restored(self):
        state = self._state()
        with patch("core.gardener_graph._agent_json", return_value={
            "answer": "证明主体予以保留，仅需将最后一句修改为完整结论。",
        }):
            result = repair_outputs(state)
        self.assertEqual(result["answer"], state["initial_answer"])
        self.assertTrue(result["repair_degraded"])
        self.assertTrue(result["repair_diagnostics"]["meta_language_detected"])

    def test_lower_blueprint_recall_is_rejected(self):
        state = self._state()
        with patch("core.gardener_graph._agent_json", return_value={"answer": "最终可得结论。"}):
            result = repair_outputs(state)
        self.assertEqual(result["answer"], state["initial_answer"])
        self.assertTrue(result["repair_degraded"])
        self.assertLess(
            result["repair_diagnostics"]["candidate_coverage"]["coverage"],
            result["repair_diagnostics"]["baseline_coverage"]["coverage"],
        )

    def test_latex_gate_rejects_unclosed_environment(self):
        review = _latex_syntax_review(r"\\begin{align} x &= 1")
        self.assertFalse(review["passed"])

    def test_must_not_search_prunes_retrieval_without_dropping_preferences(self):
        state = {
            "question": "仅利用给定条件证明该恒等式。",
            "intent": {"primary_intent": "explain_mechanism"},
            "reasoning_profile": {"activated": True, "key": "mathematical_proof"},
            "personalization_plan": {"status": "applied", "strategy_summary": "先直觉后推导"},
            "trace": [],
        }
        self.assertEqual(route_after_personalization(state), "prepare_closed_loop")
        prepared = prepare_closed_loop(state)
        self.assertEqual(prepared["evidence_review"]["routing_target"], "MUST_NOT_SEARCH")
        self.assertEqual(state["personalization_plan"]["strategy_summary"], "先直觉后推导")

    def test_closed_loop_full_graph_calls_no_retriever(self):
        answer = (
            "由题设先把命题写成分布恒等式。再取测试对象逐步变形，得到目标量恒为零；"
            "反向代回也成立。该论证只使用题面给定的定义域与边界条件，不需要外部事实。"
            "最后有 \\boxed{结论成立}。"
        )
        case = {"id": "closed-loop-zero-search", "category": "数学", "question": "仅利用题设证明给定恒等式成立，并检查边界条件。"}
        with tempfile.TemporaryDirectory() as folder:
            store = GardenStore(Path(folder) / "garden.db")
            with patch("core.gardener_graph.search_notes") as local_search, patch(
                "core.gardener_graph.search_public_web",
            ) as public_search, patch(
                "core.gardener_graph.search_academic_articles",
            ) as academic_search, patch(
                "core.gardener_graph.search_wikipedia",
            ) as wikipedia_search, patch(
                "core.gardener_graph.understanding_chat_json", return_value=None,
            ), patch(
                "core.gardener_graph._agent_json", return_value=None,
            ), patch(
                "core.gardener_graph.chat", return_value=answer,
            ):
                result = _run_turn(store, case)
        for retriever in (local_search, public_search, academic_search, wikipedia_search):
            retriever.assert_not_called()
        nodes = [event.get("node") for event in result["agent_trace"]]
        self.assertIn("prepare_closed_loop", nodes)
        self.assertNotIn("retrieve_sources", nodes)
        self.assertNotIn("audit_evidence", nodes)


if __name__ == "__main__":
    unittest.main()
