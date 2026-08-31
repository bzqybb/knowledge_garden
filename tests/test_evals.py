from __future__ import annotations

import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

from core.storage import GardenStore
from evals.adapter import load_cases, retrieval_metrics, run_retrieval_case, temporary_store
from evals.boundary_eval import local_checks, summarize
from evals.deep_reasoning_eval import deep_local_checks, premise_identified
from evals.judge_config import (
    DEEPSEEK_KEY_PATH,
    GLM_GENERATOR_KEY_PATH,
    GLM_KEY_PATH,
    LEGACY_KEY_PATH,
    judge_api_key,
    judge_base_url,
    judge_model,
    judge_independence,
    judge_request_options,
)
from evals.personalization_adoption_eval import _observable_preference_gaps
from evals.run_eval import grouped_summary, install_ragas_langchain_compat
from evals.zhili_three_layer import classify_case, matched_evidence_groups


class EvaluationTests(unittest.TestCase):
    def test_judge_provider_uses_separate_glm_deepseek_and_tokenhub_credentials(self):
        def secret_for(path):
            if path == GLM_KEY_PATH:
                return "glm-secret"
            if path == DEEPSEEK_KEY_PATH:
                return "deepseek-secret"
            return "tokenhub-secret"

        with patch.dict("os.environ", {}, clear=True), patch(
            "evals.judge_config.load_secret", side_effect=secret_for,
        ) as load:
            self.assertEqual(judge_model(), "deepseek-v4-pro")
            self.assertEqual(judge_api_key("glm-5.2"), "glm-secret")
            self.assertEqual(judge_api_key("deepseek-v4-pro"), "deepseek-secret")
            self.assertEqual(judge_api_key("deepseek-v4-flash-0731"), "tokenhub-secret")
            self.assertEqual(
                judge_base_url("glm-5.2"),
                "https://open.bigmodel.cn/api/coding/paas/v4",
            )
            self.assertEqual(
                judge_base_url("deepseek-v4-pro"),
                "https://api.deepseek.com",
            )
            self.assertEqual(
                judge_base_url("deepseek-v4-flash-0731"),
                "https://tokenhub.tencentmaas.com/v1",
            )
        self.assertEqual(
            [call.args[0] for call in load.call_args_list],
            [GLM_KEY_PATH, DEEPSEEK_KEY_PATH, LEGACY_KEY_PATH],
        )

    def test_explicit_generator_credential_fallback_is_disclosed(self):
        with patch.dict("os.environ", {"JUDGE_USE_GENERATOR_CREDENTIAL": "true"}, clear=True), patch(
            "evals.judge_config.load_secret", return_value="generator-secret",
        ) as load:
            self.assertEqual(judge_api_key("glm-4.5-airx"), "generator-secret")
            self.assertEqual(judge_independence("glm-4.5-airx"), "same_provider_and_credential_lane_as_generator")
        load.assert_called_once_with(GLM_GENERATOR_KEY_PATH)

    def test_tokenhub_kimi_k3_uses_provider_required_temperature(self):
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("JUDGE_TEMPERATURE", None)
            self.assertEqual(judge_request_options("kimi-k3")["temperature"], 1.0)

    def test_personalization_judge_cannot_reward_a_refusal_plan(self):
        refusal = "## 这次先不补写答案\n\n当前证据不足，请补充教材。"
        complete = (
            "从几何直觉看，它表示空间中的伸缩方向。严格定义为 Av=λv，"
            "因此变换后方向不变。例如 A=diag(2,3)，v=(1,0) 时 Av=2v。"
        )
        self.assertIn("回答没有实际讲解知识内容", _observable_preference_gaps(refusal))
        self.assertEqual(_observable_preference_gaps(complete), [])

    def test_seed_dataset_is_valid(self):
        root = Path(__file__).resolve().parent.parent
        cases = load_cases(root / "evals" / "datasets" / "seed_v1.jsonl")
        self.assertGreaterEqual(len(cases), 5)
        self.assertEqual(len({case["id"] for case in cases}), len(cases))

    def test_foundational_challenge_dataset_has_grounded_cases_and_explicit_boundaries(self):
        root = Path(__file__).resolve().parent.parent
        cases = load_cases(root / "evals" / "datasets" / "zhili_foundations_challenge_v1.jsonl")
        self.assertGreaterEqual(len(cases), 30)
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        self.assertTrue(all(case.get("discipline") and case.get("difficulty") for case in cases))
        self.assertTrue(all(case.get("reference_titles") or case.get("should_abstain") for case in cases))
        self.assertGreaterEqual(len({case["discipline"] for case in cases}), 5)

    def test_zhili_college_question_bank_preserves_all_54_user_questions(self):
        root = Path(__file__).resolve().parent.parent
        cases = load_cases(root / "evals" / "datasets" / "zhili_college_54_v1.jsonl")
        self.assertEqual(len(cases), 54)
        self.assertEqual(len({case["id"] for case in cases}), 54)
        self.assertTrue(all(case.get("reference") and case.get("evidence_terms") for case in cases))
        self.assertEqual({case["section"] for case in cases}, {"基础概念", "理论辨析", "交叉前沿"})

    def test_boundary_question_bank_preserves_all_22_user_questions(self):
        root = Path(__file__).resolve().parent.parent
        cases = load_cases(root / "evals" / "datasets" / "zhili_boundary_22_v1.jsonl")
        self.assertEqual(len(cases), 22)
        self.assertEqual(len({case["id"] for case in cases}), 22)
        self.assertTrue(all(case.get("boundary") and case.get("expected_style") for case in cases))
        self.assertIn("health", {case["category"] for case in cases})
        self.assertIn("philosophy", {case["category"] for case in cases})

    def test_deep_reasoning_question_bank_preserves_all_50_user_questions(self):
        root = Path(__file__).resolve().parent.parent
        cases = load_cases(root / "evals" / "datasets" / "zhili_deep_reasoning_50_v1.jsonl")
        self.assertEqual(len(cases), 50)
        self.assertEqual({case["id"] for case in cases}, {f"P-{index:02}" for index in range(1, 51)})
        self.assertTrue(all(case.get("reference") and case.get("evidence_terms") for case in cases))
        self.assertEqual({case["discipline"] for case in cases}, {"数学", "物理", "化学", "生物", "信息", "交叉"})
        traps = {case["id"] for case in cases if case.get("premise_status") != "valid"}
        self.assertTrue({"P-06", "P-22", "P-28", "P-42", "P-44", "P-45"}.issubset(traps))

    def test_systematic_agent_bank_preserves_six_suites_and_memory_sequences(self):
        root = Path(__file__).resolve().parent.parent
        cases = load_cases(root / "evals" / "datasets" / "systematic_agent_42_v1.jsonl")

        self.assertEqual(len(cases), 42)
        self.assertEqual(
            {case["suite"] for case in cases},
            {
                "deep_reasoning", "knowledge_connection", "memory_sensitive",
                "evidence_boundary", "multimodal", "comprehensive",
            },
        )
        self.assertEqual(sum(case["suite"] == "evidence_boundary" for case in cases), 8)
        self.assertEqual(sum(case["suite"] == "memory_sensitive" for case in cases), 12)

    def test_physics_gradient_bank_preserves_23_foundation_and_20_challenge_questions(self):
        root = Path(__file__).resolve().parent.parent
        cases = load_cases(root / "evals" / "datasets" / "physics_gradient_43_v1.jsonl")

        self.assertEqual(len(cases), 43)
        self.assertEqual(sum(case["id"].startswith("PHY-P-") for case in cases), 23)
        self.assertEqual(sum(case["id"].startswith("PHY-D-") for case in cases), 20)

    def test_deep_reasoning_detects_thermodynamic_and_poisson_premise_corrections(self):
        thermo = {"id": "P-28", "premise_status": "false"}
        poisson = {"id": "P-22", "premise_status": "missing_condition"}
        self.assertTrue(premise_identified(
            thermo, "恒温恒容应使用亥姆霍兹自由能 ΔF；恒温恒压时才可使用 Gibbs 自由能。",
        ))
        self.assertFalse(premise_identified(thermo, "恒温恒容时 ΔG < 0，所以题设正确。"))
        self.assertTrue(premise_identified(poisson, "若 A 显含时间，应补上 ∂A/∂t 项。"))
        self.assertFalse(premise_identified(poisson, "对于任意 A，总有 dA/dt={A,H}。"))

    def test_deep_reasoning_checks_unknown_citations_and_exact_calculations(self):
        case = {"id": "P-10", "premise_status": "valid"}
        row = {
            "answer": "特征值为 (5±√33)/2。[L9]",
            "citations": [{"source_id": "L1", "title": "高等代数"}],
            "local_checks": {}, "coverage_status": "textbook_grounded",
        }
        checks = deep_local_checks(case, row)
        self.assertEqual(checks["unknown_citation_ids"], ["L9"])
        self.assertTrue(checks["calculation_reference_matched"])
        self.assertTrue(checks["textbook_coverage_available"])

    def test_physics_proof_query_expands_pendulum_and_poisson_english_terms(self):
        from core.query_understanding import build_query_plan

        pendulum = build_query_plan("推导单摆的小角度运动方程，并求出其周期公式。", concepts=["单摆"])
        poisson = build_query_plan("用泊松括号推导哈密顿系统的时间演化。", concepts=["泊松括号"])
        self.assertIn("pendulum", [alias.casefold() for alias in pendulum["aliases"]])
        self.assertIn("poisson bracket", [alias.casefold() for alias in poisson["aliases"]])
        self.assertIn("物理", pendulum["foundation_fields"])

    def test_local_inference_caps_cpu_threads_without_breaking_started_runtime(self):
        import core.inference_runtime as runtime

        calls = []
        torch = SimpleNamespace(
            set_num_threads=lambda count: calls.append(("intra", count)),
            set_num_interop_threads=lambda count: calls.append(("inter", count)),
            get_num_threads=lambda: 3,
        )
        with patch.object(runtime, "_CONFIGURED", False), patch.dict("os.environ", {"GARDEN_LOCAL_MODEL_THREADS": "3"}):
            self.assertEqual(runtime.configure_local_inference(torch), 3)
        self.assertEqual(calls, [("intra", 3), ("inter", 2)])

    def test_overnight_summary_distinguishes_local_checks_from_independent_judging(self):
        from evals.overnight_summary import render

        empty = (Path("baseline.json"), {"summary": {"cases": 1, "verdicts": {"pass": 1}}, "rows": []})
        full = (
            Path("full.json"),
            {
                "summary": {"cases": 1, "premise_traps": 1, "premise_traps_identified": 0},
                "rows": [{
                    "id": "P-45", "discipline": "信息", "question": "推导线性回归闭式解",
                    "premise_status": "missing_condition", "latency_ms": 1000,
                    "local_checks": {"premise_identified": False},
                }],
            },
        )
        sample = (
            Path("sample.json"),
            {
                "summary": {"cases": 1, "verdicts": {"pass": 1}},
                "rows": [{
                    "id": "P-45", "latency_ms": 800, "local_checks": {"premise_identified": True},
                    "judge": {"verdict": "pass"},
                }],
            },
        )
        content = render(
            boundary_before=empty, boundary_after=empty,
            reasoning_before=empty, reasoning_after=empty,
            full=full, sample=sample,
        )
        self.assertIn("Kimi 结论仅覆盖明确选定的代表题", content)
        self.assertIn("修复并复测后识别 1 道", content)
        self.assertIn("P-45", content)

    def test_boundary_local_checks_find_medical_risk_and_mechanical_structure(self):
        checks = local_checks(
            {"category": "health", "must_not": ["一定是血栓"]},
            {
                "answer": "## 先说结论\n一定是血栓。\n\n## 为什么\n因为疼。\n\n## 成立边界\n没有。",
                "evidence_review": {"sufficient": False}, "quality_review": {"passed": False},
            },
        )
        self.assertTrue(checks["mechanical_framework"])
        self.assertFalse(checks["medical_escalation"])
        self.assertEqual(checks["forbidden_claims"], ["一定是血栓"])

    def test_boundary_local_checks_recognize_honest_premise_correction_abstention(self):
        checks = local_checks(
            {"category": "proof", "must_not": []},
            {
                "answer": "设计矩阵必须满列秩。不过，当前教材中仍然证据不足，没有找到能直接核对这条修正的相关正文。",
                "evidence_review": {"sufficient": False},
                "quality_review": {"passed": True},
            },
        )
        self.assertTrue(checks["refused"])

    def test_boundary_summary_tracks_judge_scores_latency_and_refusals(self):
        summary = summarize([
            {
                "category": "philosophy", "latency_ms": 9000,
                "local_checks": {"mechanical_framework": True, "refused": True},
                "node_timings_ms": {"generate_answer": 6000},
                "judge": {"naturalness": 2, "boundary_safety": 4, "hallucination": False, "verdict": "warn"},
            },
            {
                "category": "health", "latency_ms": 15000,
                "local_checks": {"mechanical_framework": False, "refused": False},
                "node_timings_ms": {"generate_answer": 8000},
                "judge": {"naturalness": 4, "boundary_safety": 5, "hallucination": True, "verdict": "fail"},
            },
        ])
        self.assertEqual(summary["cases"], 2)
        self.assertEqual(summary["average_latency_seconds"], 12.0)
        self.assertEqual(summary["mechanical_answers"], 1)
        self.assertEqual(summary["unsupported_open_refusals"], 1)
        self.assertEqual(summary["hallucinations"], 1)
        self.assertEqual(summary["mean_scores_out_of_5"]["naturalness"], 3.0)

    def test_three_layer_coverage_identifies_new_textbook_and_missing_subject(self):
        case = {
            "id": "chem", "question": "反应商如何判断化学平衡？",
            "section": "基础概念", "discipline": "化学", "reference": "Q 与 K 的比较",
            "evidence_terms": [["反应商"], ["平衡常数"]],
            "book_patterns": ["化学"],
        }
        note = {
            "title": "普通化学 · 第 100 页", "kind": "textbook", "source": "pdf",
            "content": "反应商与平衡常数的比较可以判断反应方向。" * 3,
        }
        grounded = classify_case(case, [note])
        self.assertEqual(grounded["coverage_status"], "textbook_grounded")
        self.assertFalse(grounded["should_abstain"])
        missing = classify_case(case, [])
        self.assertEqual(missing["coverage_status"], "no_local_evidence")
        self.assertTrue(missing["should_abstain"])

    def test_keypoint_groups_ignore_ocr_spacing(self):
        self.assertEqual(
            matched_evidence_groups("化 学 平 衡 与 反 应 商", [["化学平衡"], ["反应商"]]),
            {0, 1},
        )

    def test_three_layer_coverage_accepts_every_strong_matching_textbook_page(self):
        case = {
            "id": "chem", "question": "如何判断化学平衡？", "section": "基础概念",
            "evidence_terms": [["反应商"], ["平衡常数"]], "book_patterns": ["化学"],
        }
        notes = [{
            "title": f"普通化学 · 第 {page} 页", "kind": "textbook", "source": "pdf",
            "content": "反应商与平衡常数共同决定反应方向。" * (page + 1),
        } for page in range(1, 7)]
        grounded = classify_case(case, notes)
        self.assertEqual(len(grounded["reference_titles"]), 6)
        self.assertIn("普通化学 · 第 1 页", grounded["reference_titles"])

    def test_three_layer_coverage_excludes_unrelated_private_bridge_notes(self):
        case = {
            "id": "group", "question": "什么是群？", "section": "基础概念",
            "evidence_terms": [["群"], ["运算"]], "book_patterns": ["代数"],
        }
        private_bridge = {
            "title": "微信讨论群", "kind": "bridge", "source": "derived",
            "content": "讨论群里安排下次活动和运算练习。",
        }
        self.assertEqual(classify_case(case, [private_bridge])["coverage_status"], "no_local_evidence")

    def test_grouped_summary_separates_disciplines_and_abstention(self):
        rows = [
            {"discipline": "数学", "hit_at_5": 1.0, "reciprocal_rank": 1.0},
            {"discipline": "数学", "hit_at_5": 0.0, "reciprocal_rank": 0.0},
            {"discipline": "知识边界", "should_abstain": True, "hit_at_5": 0.0, "retrieval_abstention_correct": 1.0},
        ]
        groups = grouped_summary(rows, "discipline")
        self.assertEqual(groups["数学"]["cases"], 2)
        self.assertEqual(groups["数学"]["hit_at_5"], 0.5)
        self.assertEqual(groups["知识边界"]["retrieval_abstention_correct"], 1.0)
        self.assertNotIn("hit_at_5", groups["知识边界"])
        mixed = grouped_summary([
            {"discipline": "数学", "hit_at_5": 1.0, "should_abstain": False},
            {"discipline": "数学", "hit_at_5": 0.0, "should_abstain": True,
             "retrieval_abstention_correct": 1.0},
        ], "discipline")
        self.assertEqual(mixed["数学"]["hit_at_5"], 1.0)
        self.assertEqual(mixed["数学"]["retrieval_abstention_correct"], 1.0)

    def test_retrieval_case_preserves_user_visible_question_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            store = GardenStore(Path(folder) / "garden.db")
            case = {
                "id": "meta", "question": "不存在的专门术语", "reference": "应拒答",
                "discipline": "数学", "difficulty": "较难", "reasoning_type": "反例",
                "reference_titles": [], "should_abstain": True,
            }
            with patch("evals.adapter.search_notes", return_value=[]):
                row = run_retrieval_case(store, case)
        self.assertEqual(row["discipline"], "数学")
        self.assertEqual(row["difficulty"], "较难")
        self.assertEqual(row["reasoning_type"], "反例")
        self.assertEqual(row["reference"], "应拒答")
        self.assertTrue(row["should_abstain"])

    def test_stale_semantic_index_reuses_unchanged_pages_and_drops_modified_pages(self):
        from types import SimpleNamespace
        from core import semantic_index

        unchanged = {"path": "book#page=1", "kind": "textbook", "content_hash": "same"}
        modified = {"path": "book#page=2", "kind": "textbook", "content_hash": "new"}
        added = {"path": "new#page=1", "kind": "textbook", "content_hash": "added"}
        metadata = {
            "model": semantic_index.DEFAULT_MODEL,
            "schema_version": semantic_index.INDEX_SCHEMA_VERSION,
            "signature": "stale",
            "note_hashes": {"book#page=1": "same", "book#page=2": "old"},
            "records": [
                {"path": "book#page=1", "kind": "textbook", "chunk_index": 0,
                 "title": "保留页", "text": "unchanged source"},
                {"path": "book#page=2", "kind": "textbook", "chunk_index": 0,
                 "title": "过期页", "text": "must not appear"},
            ],
        }
        index = SimpleNamespace(ntotal=2, search=lambda vector, limit: ([[0.92, 0.88]], [[0, 1]]))
        model = SimpleNamespace(encode=lambda values, normalize_embeddings: values)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metadata.json"
            path.write_text(__import__("json").dumps(metadata), encoding="utf-8")
            with patch.object(semantic_index, "METADATA_FILE", path), patch.object(
                semantic_index, "_load", return_value=(index, model, metadata),
            ):
                result = semantic_index.semantic_search(
                    "力学问题", store_notes=[unchanged, modified, added],
                )
        self.assertEqual([item["title"] for item in result], ["保留页"])

    def test_foundation_retrieval_prioritizes_actual_pdf_over_derived_concept(self):
        from core.retrieval import _target_kind_weight

        self.assertGreater(
            _target_kind_weight("textbook", "pdf"),
            _target_kind_weight("knowledge", "derived_domain"),
        )

    def test_foundational_retrieval_penalizes_indexes_but_not_body_pages(self):
        from core.retrieval import _textbook_navigation_weight

        index = {"kind": "textbook", "source": "pdf", "content": "534 INDEX angular momentum capacitor"}
        contents = {"kind": "textbook", "source": "pdf", "content": "vi CONTENTS mechanics circuits"}
        chinese_contents = {"kind": "textbook", "source": "pdf", "content": "目录 第一章 化学平衡"}
        split_chinese_index = {
            "kind": "textbook", "source": "pdf",
            "content": "索 温室效应 448 污水处理 447 化学材料 431 光纤 432 超导 434 药物 455",
        }
        body = {"kind": "textbook", "source": "pdf", "content": "125 Simple harmonic motion and equilibrium"}
        self.assertLess(_textbook_navigation_weight(index), 0.5)
        self.assertLess(_textbook_navigation_weight(contents), 0.5)
        self.assertLess(_textbook_navigation_weight(chinese_contents), 0.5)
        self.assertLess(_textbook_navigation_weight(split_chinese_index), 0.5)
        self.assertEqual(_textbook_navigation_weight(body), 1.0)

    def test_foundational_domain_weight_favors_matching_subject_over_novel(self):
        from core.retrieval import _foundation_domain_weight

        chemistry = {"kind": "textbook", "source": "pdf", "title": "普通化学原理 · 第 50 页"}
        calculus = {"kind": "textbook", "source": "pdf", "title": "高等微积分1讲义 · 第 50 页"}
        novel = {"kind": "textbook", "source": "pdf", "title": "念念远山 · 第 50 页"}
        self.assertGreater(_foundation_domain_weight(chemistry, {"化学"}), 1.0)
        self.assertLess(_foundation_domain_weight(calculus, {"化学"}), 1.0)
        self.assertLess(_foundation_domain_weight(novel, {"化学"}), 1.0)

    def test_absent_biology_textbook_does_not_fabricate_stem_cell_evidence(self):
        from core.retrieval import search_notes

        with tempfile.TemporaryDirectory() as folder:
            store = GardenStore(Path(folder) / "garden.db")
            store.upsert_note({
                "path": "chem.pdf#page=1", "title": "普通化学原理 · 第 1 页",
                "kind": "textbook", "source": "pdf", "content_hash": "chem",
                "content": "原子轨道的分化与不同类型的化学反应。",
            })
            results = search_notes(
                store, "什么是干细胞？它有哪几种类型？分别有什么分化潜能？",
                kinds={"textbook", "concept"}, semantic_enabled=False,
            )
        self.assertEqual(results, [])

    def test_cross_subject_exact_concept_remains_available_without_subject_textbook(self):
        from core.retrieval import search_notes

        with tempfile.TemporaryDirectory() as folder:
            store = GardenStore(Path(folder) / "garden.db")
            store.upsert_note({
                "path": "chem.pdf#page=1", "title": "普通化学原理 · 第 1 页",
                "kind": "textbook", "source": "pdf", "content_hash": "chem",
                "content": "干细胞具有自我更新和分化潜能，可以形成多种细胞类型。",
            })
            results = search_notes(
                store, "什么是干细胞？它有哪几种类型？分别有什么分化潜能？",
                kinds={"textbook", "concept"}, semantic_enabled=False, rerank_enabled=False,
            )
        self.assertTrue(results)

    def test_rare_specialist_term_is_not_drowned_out_by_generic_subject_words(self):
        from core.retrieval import search_notes

        with tempfile.TemporaryDirectory() as folder:
            store = GardenStore(Path(folder) / "garden.db")
            for number in range(8):
                store.upsert_note({
                    "path": f"chem.pdf#page={number + 1}",
                    "title": f"普通化学原理 · 第 {number + 1} 页", "kind": "textbook",
                    "source": "pdf", "content_hash": str(number),
                    "content": "分子药物设计与分子结构分析。" * 8,
                })
            store.upsert_note({
                "path": "chem.pdf#page=99", "title": "普通化学原理 · 第 99 页",
                "kind": "textbook", "source": "pdf", "content_hash": "rare",
                "content": "左手和右手互成镜像，这种性质称为手性。",
            })
            results = search_notes(
                store, "什么是手性分子？它对药物设计有什么影响？",
                semantic_enabled=False, rerank_enabled=False, limit=5,
            )
        self.assertIn("普通化学原理 · 第 99 页", [item["title"] for item in results])
        specialist = next(item for item in results if item["title"] == "普通化学原理 · 第 99 页")
        self.assertGreater(specialist.get("specialist_concept_bonus", 0.0), 0.0)

    def test_center_dogma_routes_to_foundational_biology(self):
        from core.query_understanding import build_query_plan

        plan = build_query_plan("什么是‘中心法则’？它在生命科学中有什么意义？")
        self.assertEqual(plan["subject_mode"], "foundational")
        self.assertIn("生物", plan["foundation_fields"])

    def test_reinforcement_learning_does_not_accidentally_route_to_chemistry(self):
        from core.query_understanding import build_query_plan

        plan = build_query_plan("请比较监督学习、无监督学习和强化学习，并说明应用场景。")
        self.assertIn("计算机", plan["foundation_fields"])
        self.assertNotIn("化学", plan["foundation_fields"])
        self.assertNotIn("物理", plan["foundation_fields"])

    def test_scientific_philosophy_is_not_mistaken_for_physics(self):
        from core.query_understanding import build_query_plan

        plan = build_query_plan("波普尔的证伪主义和库恩的范式理论有什么区别？")
        self.assertIn("哲学", plan["foundation_fields"])
        self.assertNotIn("物理", plan["foundation_fields"])

    def test_absent_compound_concept_rejects_generic_subject_pages(self):
        from core.retrieval import search_notes

        with tempfile.TemporaryDirectory() as folder:
            store = GardenStore(Path(folder) / "garden.db")
            store.upsert_note({
                "path": "physics.pdf#page=7", "title": "物理学 · 第 7 页",
                "kind": "textbook", "source": "pdf", "content_hash": "symmetry",
                "content": "对称性可以帮助描述物理系统的结构与运动。",
            })
            results = search_notes(
                store, "什么是对称性破缺？它在物理学中有什么意义？",
                semantic_enabled=False, rerank_enabled=False,
            )
        self.assertEqual(results, [])

    def test_missing_subject_rejects_mechanism_question_without_definition_prefix(self):
        from core.retrieval import search_notes

        with tempfile.TemporaryDirectory() as folder:
            store = GardenStore(Path(folder) / "garden.db")
            store.upsert_note({
                "path": "chem.pdf#page=2", "title": "普通化学 · 第 2 页",
                "kind": "textbook", "source": "pdf", "content_hash": "chem",
                "content": "分子结构和能量变化影响化学反应。",
            })
            results = search_notes(
                store, "神经递质如何在神经元之间传递信号？",
                semantic_enabled=False, rerank_enabled=False,
            )
        self.assertEqual(results, [])

    def test_compound_concept_preserves_separately_grounded_components(self):
        from core.retrieval import search_notes

        with tempfile.TemporaryDirectory() as folder:
            store = GardenStore(Path(folder) / "garden.db")
            for index, content in enumerate(("实数系包含有理数和无理数。", "完备性与柯西列收敛相关。")):
                store.upsert_note({
                    "path": f"math.pdf#page={index}", "title": f"数学分析 · 第 {index} 页",
                    "kind": "textbook", "source": "pdf", "content_hash": str(index), "content": content,
                })
            results = search_notes(
                store, "什么是实数系的完备性？", semantic_enabled=False, rerank_enabled=False,
            )
        self.assertTrue(results)

    def test_uniform_convergence_routes_to_foundational_mathematics(self):
        from core.query_understanding import build_query_plan

        plan = build_query_plan("请解释什么是‘一致收敛’？它和逐点收敛有什么不同？")
        self.assertEqual(plan["subject_mode"], "foundational")
        self.assertIn("数学", plan["foundation_fields"])

    def test_source_priority_preserves_two_character_specialist_concept(self):
        from core.gardener_graph import _source_argument_priority

        definition = {"title": "化学教材", "text": "左手和右手互成镜像，这种性质称为手性。", "note": {}}
        application = {
            "title": "化学教材",
            "text": "生物分子结构和计算化学用于药物设计和筛选。",
            "note": {"fusion_score": 0.99},
        }
        concepts = ["手性分子", "药物设计"]
        self.assertGreater(
            _source_argument_priority(definition, [], [], concepts),
            _source_argument_priority(application, [], [], concepts),
        )

    def test_source_priority_prefers_reaction_quotient_over_generic_chemistry_prefix(self):
        from core.gardener_graph import _source_argument_priority

        direct = {
            "title": "普通化学原理",
            "text": "平衡常数 K 与反应商 Q 的比较决定正向和逆向反应。",
            "note": {"fusion_score": 0.1},
        }
        adjacent = {
            "title": "普通化学原理",
            "text": "化学反应和平衡常数是本章学习重点，应注意反应方向。",
            "note": {"fusion_score": 0.99},
        }
        concepts = ["化学平衡常数K", "反应商Q", "反应方向"]
        self.assertGreater(
            _source_argument_priority(direct, [], [], concepts),
            _source_argument_priority(adjacent, [], [], concepts),
        )

    def test_structured_model_answer_is_rendered_as_readable_markdown(self):
        from core.gardener_graph import _format_answer_payload

        result = _format_answer_payload({
            "conclusion": "焓与自由能不同。",
            "mechanism": ["焓反映等压热效应。", "自由能综合焓变和熵变。"],
            "boundary": ["Gibbs 判据适用于恒温恒压条件。"],
        })
        self.assertIn("## 先说结论", result)
        self.assertIn("- 自由能综合焓变和熵变。", result)
        self.assertNotIn("'conclusion':", result)

    def test_nested_model_answer_content_is_unwrapped_as_markdown(self):
        from core.gardener_graph import _format_answer_payload

        result = _format_answer_payload({"content": "## 先说结论\n手性分子不能与镜像重合。"})
        self.assertEqual(result, "## 先说结论\n手性分子不能与镜像重合。")
        self.assertNotIn("content：", result)

    def test_grounding_rule_detects_missing_link_between_definition_and_application(self):
        from core.gardener_graph import _evidence_grounding_rule

        sources = [
            {"title": "普通化学", "text": "旋光异构体类似左手与右手，具有手性与镜像关系。"},
            {"title": "普通化学", "text": "药物设计可以结合计算机模拟与分子结构筛选。"},
        ]
        rule = _evidence_grounding_rule(["手性分子", "药物设计"], sources)
        self.assertIn("没有直接论证", rule)
        self.assertIn("禁止用模型常识", rule)

    def test_grounding_rule_accepts_directly_supported_concept_relationship(self):
        from core.gardener_graph import _evidence_grounding_rule

        sources = [{"title": "普通化学", "text": "手性分子的空间构型会影响药物设计。"}]
        self.assertEqual(_evidence_grounding_rule(["手性分子", "药物设计"], sources), "")

    def test_lexical_retrieval_reuses_full_note_snapshot(self):
        from core.retrieval import search_notes

        with tempfile.TemporaryDirectory() as folder:
            store = GardenStore(Path(folder) / "garden.db")
            store.upsert_note({
                "path": "chem.pdf#page=360", "title": "普通化学 · 第 360 页",
                "kind": "textbook", "source": "pdf", "content_hash": "specialist",
                "content": "旋光异构体具有手性与镜像关系。",
            })
            with patch.object(store, "list_notes", wraps=store.list_notes) as list_notes:
                results = search_notes(
                    store, "什么是手性？", semantic_enabled=False, rerank_enabled=False,
                )
        self.assertTrue(results)
        list_notes.assert_called_once_with(limit=10_000)

    def test_thermodynamics_can_retrieve_both_physics_and_chemistry_textbooks(self):
        from core.query_understanding import build_query_plan

        plan = build_query_plan("热力学中，焓和自由能的区别是什么？")
        self.assertIn("物理", plan["foundation_fields"])
        self.assertIn("化学", plan["foundation_fields"])

    def test_rerank_pool_keeps_new_textbook_lexical_hits_without_vectors(self):
        from core.retrieval import _diverse_rerank_candidates

        ranked = [
            {"path": f"old-{index}", "fusion_score": 1 - index / 100}
            for index in range(8)
        ] + [{"path": "new-chemistry", "fusion_score": 0.2, "lexical_rank": 1}]
        selected, remainder = _diverse_rerank_candidates(ranked, limit=5)
        self.assertIn("new-chemistry", [item["path"] for item in selected])
        self.assertEqual(len(selected) + len(remainder), len(ranked))

    def test_comparison_subjects_strip_domain_prefix_and_keep_one_character_quantity(self):
        from core.gardener_graph import _comparison_subjects

        question = "热力学中，焓和自由能的区别是什么？"
        intent = {"primary_intent": "compare", "core_question": question, "concepts": []}
        self.assertEqual(_comparison_subjects(intent, question), ["焓", "自由能"])

    def test_foundational_fusion_applies_navigation_penalty_before_ranking(self):
        from core.retrieval import search_notes

        with tempfile.TemporaryDirectory() as folder:
            store = GardenStore(Path(folder) / "garden.db")
            notes = [
                {"path": "physics.pdf#page=1", "title": "物理教材 · 索引",
                 "kind": "textbook", "source": "pdf", "content_hash": "index",
                 "content": "INDEX angular momentum momentum equilibrium"},
                {"path": "physics.pdf#page=2", "title": "物理教材 · 正文",
                 "kind": "textbook", "source": "pdf", "content_hash": "body",
                 "content": "Angular momentum is conserved in an isolated physical system."},
            ]
            for note in notes:
                store.upsert_note(note)
            lexical_hits = [{key: value for key, value in note.items() if key != "content"}
                            for note in notes]
            plan = {
                "resolved": "angular momentum", "subject_mode": "foundational",
                "strategy": "single_query",
                "queries": [{"text": "angular momentum", "source": "resolved", "weight": 1.0}],
            }
            with patch("core.retrieval._search_notes_lexical", return_value=lexical_hits):
                results = search_notes(
                    store, "angular momentum", query_plan=plan,
                    semantic_enabled=False, rerank_enabled=False,
                )
        self.assertEqual(results[0]["title"], "物理教材 · 正文")

    def test_foundational_retrieval_protects_lexical_semantic_consensus(self):
        from core.retrieval import _channel_consensus_bonus

        agreement = [
            {"source": "bilingual_alias", "channel": "lexical", "rank": 4},
            {"source": "bilingual_alias", "channel": "semantic", "rank": 3},
        ]
        weak = [
            {"source": "bilingual_alias", "channel": "lexical", "rank": 19},
            {"source": "bilingual_alias", "channel": "semantic", "rank": 2},
        ]
        original = [
            {"source": "resolved", "channel": "lexical", "rank": 1},
            {"source": "resolved", "channel": "semantic", "rank": 1},
        ]
        self.assertGreater(_channel_consensus_bonus(agreement), 0.0)
        near_agreement = [
            {"source": "bilingual_alias", "channel": "lexical", "rank": 2},
            {"source": "bilingual_alias", "channel": "semantic", "rank": 7},
        ]
        self.assertEqual(_channel_consensus_bonus(weak), 0.0)
        self.assertGreater(_channel_consensus_bonus(original), 0.0)
        self.assertGreater(_channel_consensus_bonus(near_agreement), 0.0)

    def test_explicit_missing_chemistry_textbook_does_not_return_unrelated_math_notes(self):
        from core.retrieval import search_notes

        with tempfile.TemporaryDirectory() as folder:
            store = GardenStore(Path(folder) / "garden.db")
            store.upsert_note({
                "path": "math.pdf#page=1", "title": "高等微积分讲义 · 第 1 页",
                "kind": "textbook", "source": "pdf", "content": "方程与连续函数",
                "content_hash": "math",
            })
            results = search_notes(
                store, "请根据本地化学教材说明SN1反应的速率方程",
                kinds={"textbook", "concept"}, semantic_enabled=False,
            )
        self.assertEqual(results, [])

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
        from ragas.metrics.collections import AnswerCorrectness, ContextPrecision, ContextRecall, Faithfulness

        self.assertTrue(ContextPrecision)
        self.assertTrue(ContextRecall)
        self.assertTrue(Faithfulness)
        self.assertTrue(AnswerCorrectness)

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

    def test_cross_language_reranking_uses_focused_english_evidence_query(self):
        from core.query_understanding import build_query_plan
        from core.retrieval import _rerank_query

        question = "简谐运动的一般解和两个常数由什么初始条件决定？"
        plan = build_query_plan(question)
        rerank_query = _rerank_query(plan, question)
        self.assertIn("simple harmonic motion", rerank_query)
        self.assertIn("initial conditions", rerank_query)
        self.assertNotIn("简谐运动", rerank_query)

    def test_monolingual_reranking_preserves_original_question(self):
        from core.retrieval import _rerank_query

        plan = {"resolved": "函数在闭区间连续", "subject_mode": "foundational", "strategy": "single_query"}
        self.assertEqual(_rerank_query(plan, "fallback"), "函数在闭区间连续")

    def test_galilean_relativity_question_routes_to_foundational_physics(self):
        from core.query_understanding import build_query_plan

        plan = build_query_plan("伽利略变换在参考系相对速度接近光速时为什么失效？")
        self.assertEqual(plan["subject_mode"], "foundational")
        self.assertIn("物理", plan["foundation_fields"])
        self.assertEqual(plan["strategy"], "bilingual_expand")


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

    def test_shared_reranker_serializes_concurrent_cpu_predictions(self):
        from core import reranker

        state = {"active": 0, "maximum": 0}
        state_lock = threading.Lock()

        class FakeModel:
            def predict(self, pairs, **kwargs):
                with state_lock:
                    state["active"] += 1
                    state["maximum"] = max(state["maximum"], state["active"])
                time.sleep(0.02)
                with state_lock:
                    state["active"] -= 1
                return [0.7] * len(pairs)

        candidates = [{"title": "教材", "snippet": "可核验正文", "fusion_score": 0.1}]
        with patch.object(reranker, "_load_model", return_value=FakeModel()):
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(
                    lambda index: reranker.rerank_candidates(f"问题 {index}", candidates),
                    range(4),
                ))
        self.assertEqual(state["maximum"], 1)
        self.assertEqual(len(results), 4)

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
        alias_query = next(item["text"] for item in plan["queries"] if item["source"] == "bilingual_alias")
        self.assertNotIn("五个节点", alias_query)

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

    def test_mechanics_question_adds_auditable_english_textbook_aliases(self):
        from core.query_understanding import build_query_plan

        plan = build_query_plan("简谐运动为什么在平衡位置附近成立？")
        self.assertEqual(plan["strategy"], "bilingual_expand")
        self.assertIn("simple harmonic motion", plan["aliases"])
        self.assertTrue(any("equilibrium" in item["text"] for item in plan["queries"]))

    def test_angular_momentum_boundary_keeps_central_force_alias(self):
        from core.query_understanding import build_query_plan

        plan = build_query_plan("角动量守恒为什么还要求中心力与内力矩条件？")
        self.assertIn("angular momentum", plan["aliases"])
        self.assertIn("central force", plan["aliases"])
        self.assertIn("internal torque", " ".join(item["text"] for item in plan["queries"]))

    def test_multihop_circuit_alias_query_keeps_kcl_independence_and_node_voltage(self):
        from core.query_understanding import build_query_plan

        plan = build_query_plan(
            "一个有五个节点的电阻网络为什么只需四个线性独立的KCL方程？"
            "已知各节点电压后支路电流又如何求出？"
        )
        query = " ".join(item["text"] for item in plan["queries"])
        self.assertIn("Kirchhoff current law", query)
        self.assertIn("linearly independent", query)
        self.assertIn("node voltage", query)
        self.assertIn("nodal analysis", query)
        self.assertIn("branch currents", query)

    def test_circuit_superposition_question_adds_linearity_boundary_aliases(self):
        from core.query_understanding import build_query_plan

        self.assertIn("nonlinear function", build_query_plan("叠加定理能直接用于功率吗？")["aliases"])

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
