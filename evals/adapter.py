from __future__ import annotations

import json
import shutil
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from core.config import TEMP_DIR
from core.context_builder import ContextBuilder
from core.gardener_graph import run_gardener_graph
from core.learning_memory import LearningMemoryService
from core.retrieval import search_notes
from core.storage import GardenStore


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    cases = []
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        payload = json.loads(line)
        if not payload.get("id") or not payload.get("question"):
            raise ValueError(f"评测数据第 {line_number} 行缺少 id 或 question")
        cases.append(payload)
    return cases


def _title_key(value: str) -> str:
    return "".join(str(value).lower().split())


def retrieval_metrics(retrieved_titles: list[str], reference_titles: list[str], k: int) -> dict[str, float]:
    retrieved = [_title_key(item) for item in retrieved_titles[:k]]
    references = {_title_key(item) for item in reference_titles if str(item).strip()}
    if not references:
        return {
            f"recall_at_{k}": 1.0 if not retrieved else 0.0,
            f"precision_at_{k}": 1.0 if not retrieved else 0.0,
        }
    hits = sum(1 for item in retrieved if item in references)
    return {
        f"recall_at_{k}": hits / len(references),
        f"precision_at_{k}": hits / max(1, len(retrieved)),
    }


def run_retrieval_case(store: GardenStore, case: dict[str, Any], *, limit: int = 10) -> dict[str, Any]:
    started = time.perf_counter()
    hits = search_notes(
        store,
        str(case["question"]),
        kinds={"concept", "moc", "bridge", "knowledge", "course", "textbook"},
        limit=limit,
    )
    titles = [str(item["title"]) for item in hits]
    paths = [str(item["path"]) for item in hits]
    reference_titles = [str(item) for item in case.get("reference_titles", [])]
    normalized_references = {_title_key(item) for item in reference_titles}
    first_relevant_rank = next(
        (index for index, title in enumerate(titles, 1) if _title_key(title) in normalized_references),
        None,
    )
    metrics = {
        **retrieval_metrics(titles, reference_titles, 5),
        **retrieval_metrics(titles, reference_titles, 10),
    }
    should_abstain = bool(case.get("should_abstain", False))
    if should_abstain:
        metrics["retrieval_abstention_correct"] = float(not hits)
    query_plan = hits[0].get("query_plan", {}) if hits else {}
    retrieval_diagnostics = [
        {
            "title": item.get("title", ""),
            "fusion_score": round(float(item.get("fusion_score", 0.0)), 6),
            "reranker_score": item.get("reranker_score"),
            "semantic_score": item.get("semantic_score"),
            "query_matches": item.get("query_matches", []),
        }
        for item in hits
    ]
    return {
        "id": case["id"],
        "category": case.get("category", "unspecified"),
        "discipline": case.get("discipline", "未分类"),
        "difficulty": case.get("difficulty", "未标注"),
        "reasoning_type": case.get("reasoning_type", ""),
        "section": case.get("section", ""),
        "coverage_status": case.get("coverage_status", ""),
        "requires_online_completion": bool(case.get("requires_online_completion", False)),
        "evidence_terms": case.get("evidence_terms", []),
        "question": case["question"],
        "reference": str(case.get("reference", "")),
        "should_abstain": should_abstain,
        "retrieved_titles": titles,
        "retrieved_context_ids": paths,
        "reference_titles": reference_titles,
        "query_plan": query_plan,
        "query_strategy": query_plan.get("strategy", "") if isinstance(query_plan, dict) else "",
        "query_count": len(query_plan.get("queries", [])) if isinstance(query_plan, dict) else 0,
        "retrieval_diagnostics": retrieval_diagnostics,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "first_relevant_rank": first_relevant_rank,
        "reciprocal_rank": round(1.0 / first_relevant_rank, 4) if first_relevant_rank else 0.0,
        "hit_at_5": float(first_relevant_rank is not None and first_relevant_rank <= 5),
        "hit_at_10": float(first_relevant_rank is not None and first_relevant_rank <= 10),
        **metrics,
    }


@contextmanager
def temporary_store(source_db: str | Path) -> Iterator[GardenStore]:
    with tempfile.TemporaryDirectory(
        prefix="knowledge-garden-eval-", dir=TEMP_DIR,
    ) as folder:
        target = Path(folder) / "garden-eval.db"
        shutil.copy2(Path(source_db), target)
        yield GardenStore(target)


def run_graph_case(
    store: GardenStore, case: dict[str, Any], *, force_full_graph: bool = True,
) -> dict[str, Any]:
    question = str(case["question"])
    session_id = f"eval-{case['id']}-{uuid.uuid4()}"
    memory = LearningMemoryService(store)
    turn = memory.begin_turn(question, session_id)
    context = ContextBuilder(store).build(
        question,
        [],
        session_id=turn["session_id"],
        request_id=turn["request_id"],
        message_id=turn["message_id"],
    )
    started = time.perf_counter()
    result = run_gardener_graph(
        store, context, include_evaluation_context=force_full_graph,
    )
    memory.complete_turn(context, result)
    evaluation = result.pop("evaluation_context", {})
    return {
        "id": case["id"],
        "category": case.get("category", "unspecified"),
        "discipline": case.get("discipline", "未分类"),
        "difficulty": case.get("difficulty", "未标注"),
        "reasoning_type": case.get("reasoning_type", ""),
        "section": case.get("section", ""),
        "coverage_status": case.get("coverage_status", ""),
        "requires_online_completion": bool(case.get("requires_online_completion", False)),
        "evidence_terms": case.get("evidence_terms", []),
        "question": question,
        "reference": str(case.get("reference", "")),
        "should_abstain": bool(case.get("should_abstain", False)),
        "answer": result.get("answer", ""),
        "answer_mode": result.get("answer_mode", "garden_enhanced"),
        "route": result.get("route", {}),
        "reasoning": result.get("reasoning", {}),
        "evidence_layer": result.get("evidence_layer", "none"),
        "retrieved_contexts": evaluation.get("retrieved_contexts", []),
        "retrieved_context_ids": evaluation.get("retrieved_context_ids", []),
        "retrieved_titles": evaluation.get("retrieved_titles", []),
        "used_source_ids": result.get("citation_binding", {}).get("used_source_ids", []),
        "agent_trace": result.get("agent_trace", []),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
