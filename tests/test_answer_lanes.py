from __future__ import annotations

import unittest
from unittest.mock import patch

from core.gardener_graph import (
    _active_preference_directives,
    gate_personalization,
    select_answer_lane,
)


class AnswerLaneTests(unittest.TestCase):
    def test_ordinary_reasoning_uses_direct_model_floor(self) -> None:
        route = select_answer_lane("观察到吸烟者恢复更快，能否说明吸烟有助于恢复？")
        self.assertEqual(route["lane"], "direct_model")

    def test_philosophical_dynamics_analogy_uses_direct_model(self) -> None:
        route = select_answer_lane("为什么人生总是如此混乱，哲学可以用动力学来解释吗？")
        self.assertEqual(route["lane"], "direct_model")
        self.assertEqual(route["evidence_route"], "MODEL_KNOWLEDGE_ALLOWED")

    def test_personal_knowledge_request_uses_garden(self) -> None:
        route = select_answer_lane("结合我的笔记解释代理梯度，并安排复习")
        self.assertEqual(route["lane"], "garden_enhanced")

    def test_requested_major_level_uses_memory_aware_lane(self) -> None:
        route = select_answer_lane("我想了解概率论初步，请按数学系的概率论难度带我入门")
        self.assertEqual(route["lane"], "garden_enhanced")
        self.assertTrue(any("自适应" in reason for reason in route["reasons"]))

    def test_current_or_sourced_request_uses_garden(self) -> None:
        route = select_answer_lane("请检索最新论文并给出来源")
        self.assertEqual(route["lane"], "garden_enhanced")

    def test_direct_material_uses_garden(self) -> None:
        route = select_answer_lane("帮我理解这篇材料", direct_material={"abstract": "x"})
        self.assertEqual(route["lane"], "garden_enhanced")

    def test_evidence_followup_stays_in_garden(self) -> None:
        route = select_answer_lane(
            "那这个为什么？",
            history=[{"role": "assistant", "content": "...", "evidence_layer": "wiki"}],
        )
        self.assertEqual(route["lane"], "garden_enhanced")

    def test_clarification_reply_stays_in_garden_without_pronoun(self) -> None:
        route = select_answer_lane(
            "学术界",
            history=[{
                "role": "assistant", "content": "您指的是哪个领域？",
                "evidence_layer": "clarification",
            }],
        )
        self.assertEqual(route["lane"], "garden_enhanced")

    def test_closed_loop_proof_uses_auditable_no_search_graph(self) -> None:
        route = select_answer_lane("仅利用题设证明给定恒等式成立，并检查边界条件。")
        self.assertEqual(route["lane"], "garden_enhanced")
        self.assertEqual(route["evidence_route"], "MUST_NOT_SEARCH")

    def test_explicit_teaching_preference_survives_execution_gate(self) -> None:
        state = {
            "intent": {
                "primary_intent": "explain_mechanism",
                "task_demand": "understand",
                "response_mode": "standard",
            },
            "reasoning_profile": {"task_key": "explain"},
            "learner_context": {
                "explicit_teaching_preferences": ["先给几何直觉，再给严格定义"],
                "active_memory_claims": [],
                "concept_mastery": [],
            },
            "trace": [],
        }
        plan = gate_personalization(state)["personalization_plan"]
        self.assertEqual(plan["status"], "applied")
        self.assertEqual(
            _active_preference_directives(plan),
            ["先给几何直觉，再给严格定义"],
        )


if __name__ == "__main__":
    unittest.main()
