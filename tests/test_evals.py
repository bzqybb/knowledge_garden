from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from core.storage import GardenStore
from evals.adapter import load_cases, retrieval_metrics, temporary_store
from evals.run_eval import install_ragas_langchain_compat


class EvaluationTests(unittest.TestCase):
    def test_seed_dataset_is_valid(self):
        root = Path(__file__).resolve().parent.parent
        cases = load_cases(root / "evals" / "datasets" / "seed_v1.jsonl")
        self.assertGreaterEqual(len(cases), 5)
        self.assertEqual(len({case["id"] for case in cases}), len(cases))

    def test_retrieval_metrics_use_rank_cutoff(self):
        metrics = retrieval_metrics(["A", "noise", "B"], ["A", "B"], 2)
        self.assertEqual(metrics["recall_at_2"], 0.5)
        self.assertEqual(metrics["precision_at_2"], 0.5)

    def test_temporary_store_does_not_mutate_source_database(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.db"
            original = GardenStore(source)
            original.set_setting("marker", "original")
            with temporary_store(source) as copied:
                copied.set_setting("marker", "evaluation")
                self.assertEqual(copied.setting("marker"), "evaluation")
            self.assertEqual(original.setting("marker"), "original")

    def test_ragas_collections_api_imports_with_langchain_compatibility_shim(self):
        install_ragas_langchain_compat()
        from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness

        self.assertTrue(ContextPrecision)
        self.assertTrue(ContextRecall)
        self.assertTrue(Faithfulness)

    def test_semantic_search_rejects_mismatched_store_before_loading_model(self):
        from core import semantic_index

        with tempfile.TemporaryDirectory() as folder:
            metadata = Path(folder) / "metadata.json"
            metadata.write_text('{"model":"test-model","signature":"different"}', encoding="utf-8")
            with patch.object(semantic_index, "METADATA_FILE", metadata), patch.object(semantic_index, "_load") as load:
                self.assertEqual(semantic_index.semantic_search("KCL", store_notes=[]), [])
                load.assert_not_called()

    def test_semantic_index_signature_does_not_depend_on_database_order(self):
        from core.semantic_index import _signature

        left = {"path": "b.md", "content_hash": "b"}
        right = {"path": "a.md", "content_hash": "a"}
        self.assertEqual(_signature([left, right], "model"), _signature([right, left], "model"))

    def test_reranker_fuses_pairwise_rank_with_existing_rrf(self):
        from core.reranker import apply_reranker_scores

        candidates = [
            {"title": "lexical", "fusion_score": 0.02},
            {"title": "semantic", "fusion_score": 0.019},
        ]
        ranked = apply_reranker_scores(candidates, [0.1, 0.9], model_name="test")
        self.assertEqual(ranked[0]["title"], "semantic")
        self.assertEqual(ranked[0]["reranker_rank"], 1)
        self.assertEqual(ranked[0]["reranker_score"], 0.9)

    def test_textbook_taxonomy_samples_pages_in_numeric_order(self):
        from core.retrieval import _pdf_page_number

        notes = [
            {"path": "pdf::book.pdf#page=100"},
            {"path": "pdf::book.pdf#page=9"},
            {"path": "pdf::book.pdf#page=10"},
        ]
        self.assertEqual(
            [item["path"] for item in sorted(notes, key=_pdf_page_number)],
            ["pdf::book.pdf#page=9", "pdf::book.pdf#page=10", "pdf::book.pdf#page=100"],
        )

    def test_reranker_can_be_disabled_without_loading_model(self):
        from core import reranker

        with patch.dict("os.environ", {"GARDEN_DISABLE_RERANKER": "1"}), patch.object(reranker, "_load_model") as load:
            candidates = [{"title": "fallback", "fusion_score": 0.1}]
            self.assertEqual(reranker.rerank_candidates("query", candidates), candidates)
            load.assert_not_called()

    def test_query_plan_preserves_original_constraints_and_adds_bilingual_aliases(self):
        from core.query_understanding import build_query_plan

        question = "一个有五个节点的网络有多少个线性独立的KCL方程？"
        plan = build_query_plan(question)
        self.assertEqual(plan["queries"][0]["text"], question)
        self.assertIn("五个节点", plan["queries"][0]["text"])
        self.assertTrue(any("Kirchhoff current law" in item["text"] for item in plan["queries"]))
        self.assertTrue(any("linearly independent" in item["text"] for item in plan["queries"]))

    def test_query_rewrite_can_be_disabled_for_ablation(self):
        from core.query_understanding import build_query_plan

        with patch.dict("os.environ", {"GARDEN_DISABLE_QUERY_REWRITE": "1"}):
            plan = build_query_plan("请问 KCL 是什么？", suggested_queries=["Kirchhoff current law"])
        self.assertEqual(plan["method"], "disabled")
        self.assertEqual(len(plan["queries"]), 1)

    def test_adaptive_query_router_skips_rewrite_for_clear_exact_term(self):
        from core.query_understanding import build_query_plan

        plan = build_query_plan("KCL是什么？")
        self.assertEqual(plan["strategy"], "single_query")
        self.assertEqual(len(plan["queries"]), 1)

    def test_adaptive_query_router_drops_ambiguous_raw_followup(self):
        from core.query_understanding import build_query_plan

        plan = build_query_plan(
            "那它为什么只有四个？",
            resolved_question="五节点网络为什么只有四个线性独立的KCL方程？",
        )
        self.assertEqual(plan["strategy"], "resolved_followup")
        self.assertEqual(len(plan["queries"]), 1)
        self.assertNotIn("那它", plan["queries"][0]["text"])

    def test_agent_rewrite_must_preserve_numeric_constraints(self):
        from core.query_understanding import build_query_plan

        plan = build_query_plan(
            "V1为4V、V2为-2V，两点电压差是多少？",
            suggested_queries=["node voltage difference", "V1 4V V2 -2V node voltage difference"],
        )
        texts = [item["text"] for item in plan["queries"]]
        self.assertNotIn("node voltage difference", texts)
        self.assertTrue(any("4V" in item and "-2V" in item for item in texts))

    def test_node_voltage_expression_adds_auditable_english_aliases(self):
        from core.query_understanding import build_query_plan

        plan = build_query_plan("节点1相对于节点2的电压是多少？")
        self.assertIn("node voltage", plan["aliases"])
        self.assertIn("reference node", plan["aliases"])

    def test_camera_flash_question_adds_auditable_english_aliases(self):
        from core.query_understanding import build_query_plan

        plan = build_query_plan("在相机闪光灯电路模型中，哪个元件用于模拟能量储存？")
        self.assertEqual(plan["strategy"], "bilingual_expand")
        alias_query = " ".join(item["text"] for item in plan["queries"][1:])
        self.assertIn("camera flash", alias_query)
        self.assertIn("xenon lamp", alias_query)
        self.assertIn("capacitor", alias_query)

    def test_unit_positive_charge_question_adds_auditable_english_aliases(self):
        from core.query_understanding import build_query_plan

        plan = build_query_plan("当单位正电荷从高电势移动到低电势时，其能量如何变化？")
        self.assertEqual(plan["strategy"], "bilingual_expand")
        self.assertIn("unit positive charge", plan["aliases"])
        self.assertIn("give up energy", plan["aliases"])

    def test_negative_voltage_question_adds_reference_direction_aliases(self):
        from core.query_understanding import build_query_plan

        plan = build_query_plan("若电压为-5V且参考方向假设A点电势高于B点，哪一点实际更高？")
        self.assertIn("reference direction", plan["aliases"])
        self.assertIn("higher potential", plan["aliases"])

    def test_adjacent_pdf_paths_stay_inside_the_same_document(self):
        from core.semantic_index import _adjacent_pdf_paths

        self.assertEqual(
            _adjacent_pdf_paths("pdf::D:/books/circuit.pdf#page=49"),
            ["pdf::D:/books/circuit.pdf#page=48", "pdf::D:/books/circuit.pdf#page=50"],
        )
        self.assertEqual(_adjacent_pdf_paths("domain::knowledge::KCL"), [])

    def test_semantic_hit_expands_to_adjacent_textbook_chunks_only(self):
        from core.semantic_index import _expand_record_window

        records = [
            {"path": "page-110", "chunk_index": 0, "text": "definition before figure"},
            {"path": "page-110", "chunk_index": 1, "text": "matched figure caption"},
            {"path": "page-110", "chunk_index": 2, "text": "next worked example"},
            {"path": "another-page", "chunk_index": 1, "text": "must not leak"},
        ]
        text, indices = _expand_record_window(records, records[1], before=3, after=0)
        self.assertEqual(indices, [0, 1])
        self.assertIn("definition before figure", text)
        self.assertIn("matched figure caption", text)
        self.assertNotIn("next worked example", text)
        self.assertNotIn("must not leak", text)

        text, indices = _expand_record_window(records, records[1], before=3, after=3)
        self.assertEqual(indices, [0, 1, 2])
        self.assertIn("next worked example", text)
        self.assertNotIn("must not leak", text)


if __name__ == "__main__":
    unittest.main()
