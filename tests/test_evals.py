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


if __name__ == "__main__":
    unittest.main()
