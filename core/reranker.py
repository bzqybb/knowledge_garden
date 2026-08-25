from __future__ import annotations

import math
import os
import threading
from typing import Any, Sequence


DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANKER_SCORE_WEIGHT = 0.04
RERANKER_RRF_WEIGHT = 1.0
_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {}


def _model_name() -> str:
    return os.getenv("GARDEN_RERANKER_MODEL", DEFAULT_RERANKER_MODEL).strip() or DEFAULT_RERANKER_MODEL


def _disabled() -> bool:
    return os.getenv("GARDEN_DISABLE_RERANKER", "").strip().lower() in {"1", "true", "yes"}


def _load_model() -> Any:
    model_name = _model_name()
    with _LOCK:
        if _CACHE.get("name") == model_name:
            return _CACHE["model"]
        import torch
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(
            model_name,
            device="cpu",
            max_length=512,
            activation_fn=torch.nn.Sigmoid(),
            local_files_only=True,
        )
        _CACHE.clear()
        _CACHE.update(name=model_name, model=model)
        return model


def apply_reranker_scores(
    candidates: Sequence[dict[str, Any]], scores: Sequence[float], *, model_name: str,
) -> list[dict[str, Any]]:
    """Fuse Cross-Encoder rank with the existing lexical/vector RRF rank."""
    if len(candidates) != len(scores):
        raise ValueError("精排分数数量与候选数量不一致")
    scored = [({**item}, float(score)) for item, score in zip(candidates, scores)]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    for rank, (item, score) in enumerate(scored, 1):
        item["retrieval_fusion_score"] = float(item.get("fusion_score", 0.0))
        item["reranker_score"] = round(score, 4)
        item["reranker_rank"] = rank
        item["reranker_model"] = model_name
        item["fusion_score"] = (
            item["retrieval_fusion_score"]
            + RERANKER_SCORE_WEIGHT * score
            + RERANKER_RRF_WEIGHT / (60 + rank)
        )
    return [item for item, _ in sorted(scored, key=lambda pair: pair[0]["fusion_score"], reverse=True)]


def rerank_candidates(query: str, candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if _disabled() or not query.strip() or not candidates:
        return [dict(item) for item in candidates]
    model = _load_model()
    pairs = []
    for item in candidates:
        passage = str(item.get("semantic_snippet") or item.get("snippet") or "").strip()
        pairs.append((query, f"{item.get('title', '')}\n{passage}"[:4000]))
    scores = model.predict(pairs, batch_size=8, show_progress_bar=False)
    normalized = [float(value) if math.isfinite(float(value)) else 0.0 for value in scores]
    return apply_reranker_scores(candidates, normalized, model_name=_model_name())
