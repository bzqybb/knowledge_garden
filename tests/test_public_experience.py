from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.agent import save_agent_insight
from core.engine import _fallback_bridge, analyze_frontier
from core.storage import GardenStore


class PublicExperienceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = GardenStore(Path(self.temp.name) / "garden.db")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_bridge_without_textbook_stays_grounded_in_source(self) -> None:
        source = (
            "检索到内容并不等于获得证据。"
            "相关文字可能只共享关键词，却没有支持问题所需的因果关系。"
            "如果把背景材料当成直接证据，就会产生有引用的幻觉。"
        )
        payload = _fallback_bridge(
            "证据角色分层", [], "本科入门", source,
            ["检索到内容并不等于获得证据。"],
        )
        rendered = " ".join(str(value) for value in payload.values())
        self.assertIn("检索到内容并不等于获得证据", rendered)
        self.assertIn("有引用的幻觉", rendered)
        self.assertNotIn("悬浮种子", rendered)
        self.assertNotIn("嫁接的新枝", rendered)
        self.assertNotIn("补充课程大纲", rendered)
        self.assertEqual(payload["textbook_mapping"], "")

    def test_public_writeback_creates_isolated_cloud_notes(self) -> None:
        result = save_agent_insight(
            self.store, "为什么检索更多不一定更可靠？",
            "无关上下文会稀释关键证据，来源之间也可能相互矛盾。", [], [],
        )
        self.assertEqual(result["storage"], "isolated_cloud_garden")
        self.assertTrue(result["pending_obsidian_sync"])
        notes = self.store.list_notes(limit=20)
        self.assertTrue(any(note["kind"] == "spark" for note in notes))
        self.assertTrue(any(note["kind"] == "concept" for note in notes))

    def test_public_analysis_persists_source_for_reopening(self) -> None:
        extraction = {
            "concepts": ["证据审查"],
            "evidence": {"证据审查": ["每条来源都应绑定到具体结论。"]},
            "chunks": ["每条来源都应绑定到具体结论。"],
            "source_text": "每条来源都应绑定到具体结论。",
        }
        guide = {
            "overview": "材料说明了证据应与具体结论绑定。",
            "chapter_outline": [], "key_points": [],
            "concepts": ["证据审查"], "caveats": [], "questions": [],
        }
        with patch("core.engine.extract_concepts_with_evidence", return_value=extraction), patch(
            "core.engine._frontier_json", return_value=guide,
        ):
            result = analyze_frontier(
                self.store, "RAG 证据审查", extraction["source_text"], "",
            )
        self.assertEqual(result["saved_to"], "isolated_cloud_garden")
        notes = self.store.list_notes(kind="frontier", limit=10)
        self.assertEqual(notes[0]["title"], "RAG 证据审查")
        self.assertIn("具体结论", notes[0]["content"])

    def test_fast_public_analysis_skips_per_card_model_calls(self) -> None:
        extraction = {
            "concepts": ["证据审查"],
            "evidence": {"证据审查": ["检索结果必须绑定到具体结论。"]},
            "chunks": ["检索结果必须绑定到具体结论。"],
            "source_text": "检索结果必须绑定到具体结论。",
        }
        guide = {
            "overview": "材料说明了证据应与具体结论绑定。",
            "chapter_outline": [], "key_points": [],
            "concepts": ["证据审查"], "caveats": [], "questions": [],
        }
        with patch("core.engine.extract_concepts_with_evidence", return_value=extraction), patch(
            "core.engine._frontier_json", return_value=guide,
        ) as generate:
            result = analyze_frontier(
                self.store, "快速导读", extraction["source_text"], "", fast=True,
            )
        self.assertEqual(result["analysis_mode"], "glm_deep_read")
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(result["cards"], [])

    def test_vault_receives_substantive_glm_guide_not_placeholder_bridge(self) -> None:
        vault = Path(self.temp.name) / "vault"
        vault.mkdir()
        self.store.set_setting("vault_path", str(vault))
        extraction = {
            "concepts": ["证据审查"],
            "evidence": {"证据审查": ["每条来源都应绑定到具体结论。"]},
            "chunks": ["每条来源都应绑定到具体结论。"],
            "source_text": "每条来源都应绑定到具体结论。",
        }
        guide = {
            "overview": "文章强调检索结果只有绑定具体结论后才能成为证据。",
            "chapter_outline": [{"title": "证据条件", "summary": "区分检索命中与证据支持。"}],
            "key_points": [{
                "point": "关键词相似不等于支持结论",
                "evidence": "每条来源都应绑定到具体结论。",
                "boundary": "当前材料没有讨论来源真实性的独立核验。",
            }],
            "concepts": ["证据审查"],
            "caveats": ["仍需核验来源真实性"],
            "questions": ["哪条来源支持了哪一个结论？"],
        }
        with patch("core.engine.extract_concepts_with_evidence", return_value=extraction), patch(
            "core.engine._frontier_json", return_value=guide,
        ):
            result = analyze_frontier(self.store, "RAG 证据审查", extraction["source_text"])
        written = Path(result["guide_path"]).read_text(encoding="utf-8")
        self.assertIn("## 核心论点与证据", written)
        self.assertIn("关键词相似不等于支持结论", written)
        self.assertIn("每条来源都应绑定到具体结论", written)
        self.assertNotIn("等待阅读全文后补充笔记", written)
        self.assertNotIn("它解释了原始材料中的一个关键机制或假设", written)


if __name__ == "__main__":
    unittest.main()
