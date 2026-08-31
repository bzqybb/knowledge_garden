from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.bilibili_mcp import _video_analysis, analyze_video_transcript, inspect_public_video, read_video, runtime_status
from core.paper_reader import _connect_local_knowledge, deep_read_paper
from core.storage import GardenStore
from core.transcript import split_timestamped_text


class BilibiliMCPAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = GardenStore(Path(self.temp.name) / "garden.db")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_runtime_status_reports_real_mcp_credential_state(self) -> None:
        with (
            patch("core.bilibili_mcp._find_node", return_value=Path("node.exe")),
            patch("core.bilibili_mcp._find_package_file", return_value=Path("index.js")),
            patch("core.bilibili_mcp.call_tool", return_value={
                "configured": True, "logged_in": True, "source": "global_config",
            }),
        ):
            status = runtime_status()
        self.assertTrue(status["installed"])
        self.assertTrue(status["logged_in"])
        self.assertIn("读取字幕", status["message"])

    def test_video_read_uses_mcp_transcript_and_persists_evidence_note(self) -> None:
        def tool(name: str, arguments: dict, **_: object) -> dict:
            if name == "check_bilibili_credentials":
                return {"configured": True, "logged_in": True}
            if name == "get_video_metadata":
                return {"title": "矩阵的秩", "author": "测试UP"}
            return {
                "data_source": "subtitle",
                "source_url": "https://www.bilibili.com/video/BV1owner0000",
                "transcript": "[00:00:01 --> 00:00:05] 矩阵的秩是线性无关列向量的最大数目。",
            }

        with (
            patch("core.bilibili_mcp.inspect_public_video", return_value={
                "status": "no_subtitle", "message": "没有公开字幕",
            }),
            patch("core.bilibili_mcp.call_tool", side_effect=tool),
            patch("core.bilibili_mcp._video_analysis", return_value={
                "overview": "视频解释了矩阵的秩。", "key_points": [], "concepts": ["矩阵的秩"],
                "caveats": [], "chapter_outline": [], "questions": [],
            }),
        ):
            result = read_video(self.store, "https://www.bilibili.com/video/BV1owner0000")
        self.assertEqual(result["data_source"], "subtitle")
        notes = self.store.list_notes(kind="frontier")
        self.assertEqual(len(notes), 1)
        self.assertIn("带时间戳的转录", notes[0]["content"])
        self.assertIn("B站", notes[0]["tags"])

    def test_public_subtitle_fallback_preserves_timestamps_without_login(self) -> None:
        responses = [
            {"data": {"title": "公开字幕视频", "owner": {"name": "测试UP"}, "pages": [{"cid": 7}]}},
            {"data": {"subtitle": {"subtitles": [{
                "lan": "zh-Hans", "lan_doc": "中文（简体）", "subtitle_url": "//example.com/subtitle.json",
            }]}}},
            {"body": [
                {"from": 1.2, "to": 4.8, "content": "第一句"},
                {"from": 65, "to": 70, "content": "第二句"},
            ]},
        ]
        with patch("core.bilibili_mcp._request_json", side_effect=responses):
            result = inspect_public_video("BV1owner0000")
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["data_source"], "public_subtitle")
        self.assertIn("[00:00:01 --> 00:00:04] 第一句", result["transcript"])
        self.assertIn("[00:01:05 --> 00:01:10] 第二句", result["transcript"])

    def test_video_read_can_return_transcript_before_deep_analysis(self) -> None:
        transcript = "[00:00:01 --> 00:00:05] 这是可先展示的完整字幕。" * 5
        with (
            patch("core.bilibili_mcp.inspect_public_video", return_value={
                "status": "ready", "transcript": transcript, "data_source": "public_subtitle",
                "source_url": "https://www.bilibili.com/video/BV1owner0000",
                "metadata": {"title": "分阶段视频"},
            }),
            patch("core.bilibili_mcp._video_analysis") as analysis,
        ):
            result = read_video(self.store, "BV1owner0000", analyze=False)
        analysis.assert_not_called()
        self.assertTrue(result["analysis_deferred"])
        self.assertEqual(result["transcript"], transcript)
        with patch("core.bilibili_mcp._video_analysis", return_value={
            "overview": "完成导读。", "key_points": [], "concepts": [], "caveats": [],
            "chapter_outline": [], "questions": [],
        }):
            guided = analyze_video_transcript(
                self.store, bvid_or_url=result["bvid"], title=result["title"],
                source_url=result["source_url"], data_source=result["data_source"],
                transcript=result["transcript"],
            )
        self.assertFalse(guided["analysis_deferred"])
        self.assertEqual(guided["analysis"]["overview"], "完成导读。")

    def test_long_video_analysis_covers_late_timestamped_chunks(self) -> None:
        early = "\n".join(
            f"[00:{index // 60:02d}:{index % 60:02d} --> 00:{index // 60:02d}:{(index + 1) % 60:02d}] "
            + "前段背景说明" * 28
            for index in range(150)
        )
        late_line = "[01:20:00 --> 01:20:08] 后半段提出时间晶体机制，并说明它与普通周期驱动的区别。"
        transcript = early + "\n" + late_line

        def analyze(_: str, prompt: str, **__: object) -> dict:
            if "后半段提出时间晶体机制" in prompt:
                return {
                    "overview": "后段讨论时间晶体机制。",
                    "key_points": [{"point": "提出时间晶体机制", "evidence": late_line, "timestamp": "01:20:00"}],
                    "concepts": ["时间晶体机制"], "caveats": [],
                    "chapter_outline": [{"title": "时间晶体", "timestamp": "01:20:00", "summary": "后段核心"}],
                    "questions": ["如何验证？"],
                }
            return {
                "overview": "前段介绍背景。", "key_points": [], "concepts": ["背景"],
                "caveats": [], "chapter_outline": [], "questions": [],
            }

        with patch("core.bilibili_mcp.chat_json", side_effect=analyze) as mocked:
            result = _video_analysis("长视频", transcript, "public_subtitle")
        self.assertGreater(mocked.call_count, 1)
        self.assertEqual(result["coverage"]["processed_chunks"], result["coverage"]["chunks"])
        self.assertIn("时间晶体机制", result["concepts"])
        self.assertTrue(any(item.get("timestamp") == "01:20:00" for item in result["key_points"]))

    def test_timestamp_chunking_preserves_every_line_in_order(self) -> None:
        lines = [
            f"[00:{index // 60:02d}:{index % 60:02d} --> 00:{(index + 1) // 60:02d}:{(index + 1) % 60:02d}] 第{index}条字幕"
            for index in range(80)
        ]
        transcript = "\n".join(lines)
        chunks = split_timestamped_text(transcript, max_chars=120)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("\n".join(chunks), transcript)
        flattened = "\n".join(chunks)
        for line in lines:
            self.assertEqual(flattened.count(line), 1)

    def test_video_analysis_removes_hallucinated_timestamp_evidence(self) -> None:
        transcript = "\n".join(
            f"[00:00:{index:02d} --> 00:00:{index + 1:02d}] 真实字幕第{index}句，讨论可验证事实。"
            for index in range(8)
        )
        hallucinated = {
            "overview": "测试导读。",
            "key_points": [{"point": "并不存在的结论", "evidence": "伪造原文", "timestamp": "99:99:99"}],
            "concepts": ["测试概念"], "caveats": [],
            "chapter_outline": [{"title": "伪造章节", "timestamp": "99:99:99", "summary": "不存在"}],
            "questions": [],
        }
        with patch("core.bilibili_mcp.chat_json", return_value=hallucinated):
            result = _video_analysis("测试", transcript, "public_subtitle")
        self.assertEqual(result["key_points"], [])
        self.assertEqual(result["chapter_outline"], [])
        self.assertNotIn("99:99:99", str(result["key_points"]))
        self.assertTrue(any("无法" in item or "没有可定位" in item for item in result["caveats"]))


class PaperDeepReadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = GardenStore(Path(self.temp.name) / "garden.db")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_abstract_read_labels_scope_and_validates_verbatim_evidence(self) -> None:
        abstract = "本文提出一种新的材料预测方法，并在公开数据集上比较了三个基线模型。结果显示该方法降低了预测误差。" * 3
        model = {
            "problem": "材料性质预测", "novelty": "新的预测方法", "method": "比较三个基线模型",
            "findings": [
                {"claim": "误差降低", "evidence": "结果显示该方法降低了预测误差"},
                {"claim": "训练快十倍", "evidence": "训练速度提高十倍"},
            ],
            "limitations": ["摘要没有报告误差数值"], "prerequisites": ["监督学习"],
            "reading_routes": {"ten_minutes": ["读摘要"], "thirty_minutes": ["核对方法"]},
            "questions": ["基线是否公平？"], "confidence": 0.9,
        }
        with patch("core.paper_reader.chat_json", return_value=model):
            result = deep_read_paper(self.store, {"title": "材料预测", "abstract": abstract})
        self.assertEqual(result["scope"], "abstract")
        self.assertLessEqual(result["analysis"]["confidence"], 0.68)
        self.assertTrue(result["analysis"]["findings"][0]["grounded"])
        self.assertFalse(result["analysis"]["findings"][1]["grounded"])

    def test_open_pdf_failure_falls_back_to_abstract_with_visible_error(self) -> None:
        with (
            patch("core.paper_reader.fetch_open_access_pdf_text", side_effect=ValueError("PDF受限")),
            patch("core.paper_reader.chat_json", return_value=None),
        ):
            result = deep_read_paper(self.store, {
                "title": "测试论文", "pdf_url": "https://example.org/paper.pdf",
                "abstract": "这是一个足够长的摘要，用于说明当开放全文读取失败时系统必须回退，并且明确告知用户证据范围。" * 3,
            })
        self.assertEqual(result["scope"], "abstract")
        self.assertIn("PDF受限", result["fulltext_error"])
        self.assertIn("仅依据摘要", result["source_note"])

    def test_agent_b_can_only_cite_retrieved_local_source_indices(self) -> None:
        hit = {
            "path": "pdf::linear-algebra#page=12", "title": "线性代数 · 第 12 页",
            "kind": "textbook", "source_url": "D:/books/linear-algebra.pdf",
            "snippet": "特征向量在该线性变换下方向保持不变，特征值给出对应的伸缩比例。",
            "matched_terms": ["特征向量", "特征值"], "relevance_score": 0.91,
            "knowledge_status": "substantive",
        }
        proposal = {"connections": [
            {"source_index": 99, "relation_type": "analogy", "bridge": "不存在的资料连接",
             "why_useful": "不应显示", "confidence": 0.99},
            {"source_index": 1, "relation_type": "prerequisite", "bridge": "先复习特征向量与特征值的关系",
             "why_useful": "它是理解论文谱方法的前置知识", "confidence": 0.86},
        ]}
        with (
            patch("core.paper_reader.search_notes", return_value=[hit]),
            patch("core.paper_reader.chat_json", return_value=proposal),
        ):
            result = _connect_local_knowledge(
                self.store, "谱方法论文", {"concepts": ["特征值"]}, allow_model=True,
            )
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(len(result["connections"]), 1)
        self.assertEqual(result["connections"][0]["title"], "线性代数 · 第 12 页")
        self.assertIsNone(result["connections"][0]["mastery"])


if __name__ == "__main__":
    unittest.main()
