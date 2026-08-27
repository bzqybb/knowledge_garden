from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from core.llm import LLMError, chat_json
from core.pdf_ocr import clean_pdf_text
from core.query_understanding import build_query_plan, normalize_query
from core.storage import GardenStore
from core.learning_memory import note_activation


LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{1,}|\d+(?:\.\d+)?")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]+")
STOPWORDS = {
    "一个", "一种", "这个", "这些", "以及", "可以", "进行", "通过", "我们", "你们", "他们",
    "其中", "对于", "因为", "所以", "如果", "什么", "如何", "为什么", "怎么", "怎样", "请问",
    "解释", "说明", "理解", "觉得", "是否", "关系", "影响", "作用",
}
QUESTION_FRAMES = re.compile(
    r"(?:请问|请解释|请说明|你能否|能不能|为什么|为何|怎么|怎样|如何|是什么意思|是什么|有什么关系)"
)
PLACEHOLDER_MARKERS = (
    "等待后续资料继续充实", "由 ingest 自动建立", "由Ingest自动建立", "待后续资料", "仅占位",
)


_TEXTBOOK_SUBJECTS: dict[str, tuple[str, ...]] = {
    "生物化学": ("生物化学", "biochemistry"),
    "有机化学": ("有机化学", "organic chemistry"),
    "分子生物学": ("分子生物学", "molecular biology"),
    "化学": ("化学", "chemistry", "chemical"),
    "生物学": ("生物学", "biology", "biological"),
    "生物": ("生物", "biology", "biological"),
    "力学": ("力学", "mechanics"),
    "物理": ("物理", "physics", "mechanics"),
    "微积分": ("微积分", "calculus"),
    "电路": ("电路", "circuit"),
}
_FOUNDATION_TEXTBOOK_SIGNALS: dict[str, tuple[str, ...]] = {
    "数学": ("数学", "微积分", "代数", "复变", "calculus", "mathematics", "algebra"),
    "物理": ("物理", "力学", "电路", "电磁", "量子", "physics", "mechanics", "circuit"),
    "化学": ("化学", "chemistry", "chemical"),
    "生物": ("生物", "生命科学", "遗传", "biology", "biological", "genetics"),
    "计算机": ("计算机", "算法", "数据结构", "编程", "computer", "algorithm", "programming"),
    "哲学": ("哲学", "科学史", "philosophy", "history of science"),
}
_DEFINITION_CONCEPT = re.compile(
    r"(?:什么是|何谓|请解释(?:什么是)?|请说明(?:什么是)?)[‘“\"']?"
    r"(?P<concept>[^？?，,；;。‘’“”\"']{2,24})"
)
_EXPLICIT_TEXTBOOK_SUBJECT = re.compile(
    r"(?:本地|已经导入的|已导入的|当前|现有)?"
    r"(?P<subject>生物化学|有机化学|分子生物学|生物学|化学|生物|力学|物理|微积分|电路)"
    r"(?:的)?(?:教材|课本|讲义)"
)


def _requested_textbook_exists(query: str, notes: list[dict[str, Any]]) -> bool:
    match = _EXPLICIT_TEXTBOOK_SUBJECT.search(query)
    if not match:
        return True
    signals = _TEXTBOOK_SUBJECTS[str(match.group("subject"))]
    return any(note.get("kind") in {"textbook", "course"} and any(
        signal.casefold() in str(note.get("title") or "").casefold() for signal in signals
    ) for note in notes)


def _is_foundational_plan(plan: dict[str, Any]) -> bool:
    return str(plan.get("subject_mode") or "").strip().lower() == "foundational"


def _rerank_query(plan: dict[str, Any], fallback: str) -> str:
    """Keep cross-lingual precision matching in the evidence passage language."""
    if _is_foundational_plan(plan) and plan.get("strategy") == "bilingual_expand":
        for variant in plan.get("queries") or []:
            if variant.get("source") == "bilingual_alias":
                focused = str(variant.get("text") or "").strip()
                if focused:
                    return focused
    return str(plan.get("resolved") or fallback)


def _target_kind_weight(kind: str, source_type: str) -> float:
    if kind in {"textbook", "course"} or source_type in {"textbook", "course", "pdf"}:
        return 1.35
    if kind in {"concept", "moc", "bridge", "knowledge", "knowledge_point"}:
        return 1.06
    return 1.0


def _textbook_subject_fields(note: dict[str, Any]) -> set[str]:
    if note.get("kind") not in {"textbook", "course"} and note.get("source") != "pdf":
        return set()
    title = str(note.get("title") or "").casefold()
    return {
        field for field, signals in _FOUNDATION_TEXTBOOK_SIGNALS.items()
        if any(signal.casefold() in title for signal in signals)
    }


def _foundation_domain_weight(note: dict[str, Any], fields: set[str]) -> float:
    if not fields or (note.get("kind") not in {"textbook", "course"} and note.get("source") != "pdf"):
        return 1.0
    textbook_fields = _textbook_subject_fields(note)
    if textbook_fields & fields:
        return 1.22
    return 0.68 if textbook_fields else 0.55


def _missing_foundation_evidence(
    query: str,
    fields: set[str],
    notes: list[dict[str, Any]],
    *,
    aliases: list[str] | None = None,
) -> bool:
    """Reject an absent named concept without hiding rare genuine terminology."""
    match = _DEFINITION_CONCEPT.search(query)
    if match is None:
        if not fields or any(_textbook_subject_fields(note) & fields for note in notes):
            return False
        query_terms = set(tokenize(query))
        return not any(
            note.get("kind") in {"concept", "knowledge"}
            and bool(query_terms & set(tokenize(str(note.get("title") or ""))))
            and relevance_gate(query, str(note.get("title") or ""), str(note.get("content") or ""))["passed"]
            for note in notes
        )
    concept = str(match.group("concept")).strip()
    if len(concept) < 2:
        return False
    grounded_texts = [
        f"{note.get('title', '')}\n{note.get('content', '')}".casefold()
        for note in notes
        if note.get("kind") in {"textbook", "course", "concept", "knowledge"}
    ]
    lowered = concept.casefold()
    if any(lowered in text for text in grounded_texts):
        return False
    if any(
        len(alias.strip()) >= 4 and alias.casefold() in text
        for alias in aliases or [] for text in grounded_texts
    ):
        return False

    # “实数系的完备性” can be grounded by separate textbook passages for
    # 实数系 and 完备性, even when the complete phrase never occurs verbatim.
    components = [part for part in lowered.split("的") if len(part) >= 2]
    if len(components) >= 2 and all(
        any(component in text for text in grounded_texts) for component in components
    ):
        return False

    # OCR may preserve only a rare specialist root: the chemistry textbook
    # says “手性”, not “手性分子”. Keep that page when the remaining suffix is
    # generic; do not mistake two unrelated rare terms for a complete concept.
    if len(lowered) >= 4:
        specialist_root = lowered[:2]
        suffix = lowered[2:]
        rarity_threshold = max(3, len(grounded_texts) // 200)
        specialist_threshold = max(2, len(grounded_texts) // 1_000)
        root_count = sum(specialist_root in text for text in grounded_texts)
        suffix_count = sum(suffix in text for text in grounded_texts)
        if 0 < root_count <= specialist_threshold and suffix_count > rarity_threshold:
            return False
    return True


def _channel_consensus_bonus(matches: list[dict[str, Any]]) -> float:
    """Protect evidence independently confirmed by lexical and semantic search."""
    bonus = 0.0
    for source in ("resolved", "bilingual_alias"):
        lexical = [
            int(item["rank"]) for item in matches
            if item.get("source") == source and item.get("channel") == "lexical"
        ]
        semantic = [
            int(item["rank"]) for item in matches
            if item.get("source") == source and item.get("channel") == "semantic"
        ]
        if not lexical or not semantic:
            continue
        if source == "bilingual_alias" and min(lexical) <= 5 and min(semantic) <= 5:
            bonus += 0.018
        elif source == "bilingual_alias" and min(lexical) <= 2 and min(semantic) <= 10:
            bonus += 0.009
        elif source == "resolved" and min(lexical) <= 2 and min(semantic) <= 2:
            bonus += 0.014
    return min(0.024, bonus)


def _diverse_rerank_candidates(
    ranked: list[dict[str, Any]], *, limit: int, lexical_slots: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep fresh lexical hits visible while their vector index is being built."""
    selected = list(ranked[:limit])
    selected_paths = {str(item.get("path")) for item in selected}
    lexical_leaders = sorted(
        (item for item in ranked if isinstance(item.get("lexical_rank"), int)),
        key=lambda item: (int(item["lexical_rank"]), -float(item.get("fusion_score", 0))),
    )[:lexical_slots]
    for candidate in lexical_leaders:
        candidate_path = str(candidate.get("path"))
        if candidate_path in selected_paths:
            continue
        replace_at = next(
            (index for index in range(len(selected) - 1, -1, -1)
             if not isinstance(selected[index].get("lexical_rank"), int)),
            len(selected) - 1,
        )
        if replace_at < 0:
            continue
        selected_paths.discard(str(selected[replace_at].get("path")))
        selected[replace_at] = candidate
        selected_paths.add(candidate_path)
    remainder = [item for item in ranked if str(item.get("path")) not in selected_paths]
    return selected, remainder


def _textbook_navigation_weight(note: dict[str, Any]) -> float:
    """Prevent textbook indexes/contents from outranking substantive evidence."""
    if note.get("kind") not in {"textbook", "course"} and note.get("source") != "pdf":
        return 1.0
    opening = re.sub(r"\s+", " ", str(note.get("content") or "")[:220]).strip()
    if re.match(
        r"^(?:(?:\d+|[ivxlcdm]+)\s+)?(?:(?:index|contents|table of contents)\b|目录|目\s*录|索引|术语索引)",
        opening,
        re.I,
    ):
        return 0.32
    if re.match(r"^[索引]\s", opening) and len(re.findall(r"\s\d{1,3}(?=\s|$)", opening)) >= 5:
        return 0.32
    return 1.0


def _meaningful_phrases(text: str) -> list[str]:
    """Extract concept-sized Chinese phrases without treating question frames as evidence."""
    cleaned = QUESTION_FRAMES.sub(" ", text)
    phrases: set[str] = set()
    for block in CHINESE_RE.findall(cleaned):
        if 2 <= len(block) <= 8 and block not in STOPWORDS:
            phrases.add(block)
        # Four to six characters is usually concept-sized. Two-character
        # overlaps remain useful for ranking, but may not pass the hard gate.
        for size in range(4, min(6, len(block)) + 1):
            phrases.update(block[index:index + size] for index in range(len(block) - size + 1))
    phrases.update(token.lower() for token in LATIN_RE.findall(cleaned) if len(token) >= 3)
    return sorted(phrases, key=lambda item: (-len(item), item))


def local_knowledge_status(note: dict[str, Any]) -> str:
    """Classify whether a local page can prove facts or only provide navigation."""
    content = str(note.get("content") or "")
    lowered = content.lower()
    if not content.strip() or any(marker.lower() in lowered for marker in PLACEHOLDER_MARKERS):
        return "placeholder"
    if note.get("kind") in {"textbook", "course"} or note.get("source") == "pdf":
        return "grounded"
    if note.get("source_url") or re.search(
        r"(?:https?://|doi\s*[:：]|isbn\s*[:：]|原始资料\s*[:：]|来源(?:证据)?\s*[:：]|参考文献)",
        content,
        re.I,
    ):
        return "grounded"
    return "derived"


def relevance_gate(query: str, title: str, content: str) -> dict[str, Any]:
    """A deterministic floor below the LLM reviewer.

    The gate deliberately ignores generic question words. A candidate must
    match a concept-sized phrase, or several specific smaller terms. This
    prevents activation weight or fluent model judgement from rescuing a
    semantically unrelated page.
    """
    haystack = (title + "\n" + content).lower()
    title_lower = title.lower()
    phrases = _meaningful_phrases(query)
    strong = [phrase for phrase in phrases if len(phrase) >= 4 and phrase.lower() in haystack]
    query_tokens = set(tokenize(QUESTION_FRAMES.sub(" ", query)))
    doc_tokens = set(tokenize(title + " " + content))
    small = sorted(
        token for token in query_tokens & doc_tokens
        if token not in STOPWORDS and (len(token) >= 2 or re.fullmatch(r"[A-Za-z0-9_\-]{3,}", token))
    )
    title_matches = [phrase for phrase in phrases if len(phrase) >= 2 and phrase.lower() in title_lower]
    title_token_matches = sorted(
        token for token in query_tokens & set(tokenize(title)) if len(token) >= 2 and token not in STOPWORDS
    )
    passed = bool(strong or title_matches or title_token_matches or len(small) >= 2)
    score = min(1.0, 0.62 + 0.07 * len(strong) + 0.05 * len(title_matches) + 0.04 * len(small)) if passed else 0.0
    return {
        "passed": passed,
        "score": round(score, 3),
        "matched_terms": list(dict.fromkeys([*strong[:5], *title_matches[:3], *title_token_matches[:3], *small[:5]])),
        "reason": "命中核心概念或多个具体术语" if passed else "只有通用问法或零散弱词重合",
    }


def tokenize(text: str) -> list[str]:
    tokens = [item.lower() for item in LATIN_RE.findall(text)]
    for block in CHINESE_RE.findall(text):
        if len(block) <= 4:
            tokens.append(block)
        tokens.extend(block[i:i + 2] for i in range(len(block) - 1))
        tokens.extend(block[i:i + 3] for i in range(len(block) - 2))
    return [token for token in tokens if token not in STOPWORDS]


def _snippet(text: str, terms: list[str], width: int = 280) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    positions = [clean.lower().find(term.lower()) for term in terms]
    positions = [pos for pos in positions if pos >= 0]
    start = max(0, (min(positions) if positions else 0) - 70)
    snippet = clean[start:start + width]
    return ("…" if start else "") + snippet + ("…" if start + width < len(clean) else "")


def _fallback_queries(plan: dict[str, Any], query: str) -> list[str]:
    fallback: list[str] = []
    concepts = plan.get("concepts", [])
    aliases = plan.get("aliases", [])
    if aliases:
        candidate = normalize_query(f"{query} {aliases[0]}")
        if candidate:
            fallback.append(candidate)
    for value in concepts[:1]:
        normalized = normalize_query(str(value))
        if normalized and normalized not in fallback and normalized != normalize_query(query):
            fallback.append(normalized)
    return list(dict.fromkeys(fallback))[:1]


def _search_notes_lexical(
    store: GardenStore, query: str, *, kinds: set[str] | None = None, limit: int = 5,
    strict_relevance: bool = True,
    notes: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    notes = notes if notes is not None else store.list_notes(limit=10_000)
    if kinds:
        notes = [note for note in notes if note["kind"] in kinds]
    query_counts = Counter(query_tokens)
    definition = _DEFINITION_CONCEPT.search(query)
    concept_tokens = set(tokenize(str(definition.group("concept")))) if definition else set()
    prepared: list[tuple[dict[str, Any], dict[str, Any], Counter[str], set[str], set[str]]] = []
    document_frequencies: Counter[str] = Counter()
    for note in notes:
        doc_tokens = tokenize(note["title"] + " " + note["content"])
        if not doc_tokens:
            continue
        counts = Counter(doc_tokens)
        overlap = set(query_counts) & set(counts)
        if not overlap:
            continue
        document_frequencies.update(overlap)
        relevance = relevance_gate(query, note["title"], note["content"])
        prepared.append((note, relevance, counts, overlap, set(tokenize(note["title"]))))
    rare_concept_tokens = {
        token for token in concept_tokens
        if len(token) >= 2 and 0 < document_frequencies[token] <= max(3, len(notes) // 200)
    }
    scored: list[tuple[float, dict[str, Any]]] = []
    for note, relevance, counts, overlap, title_tokens in prepared:
        rare_concept_match = bool(overlap & rare_concept_tokens)
        if strict_relevance and not relevance["passed"] and not rare_concept_match:
            continue
        score = sum(
            (1 + math.log1p(counts[token]))
            * (1 + math.log1p(len(notes) / max(1, document_frequencies[token])))
            * (2 if token in title_tokens else 1)
            * (1.8 if token in concept_tokens else 1.0)
            for token in overlap
        )
        score /= math.sqrt(max(1, sum(counts.values())))
        if rare_concept_match:
            score *= 4.0
        knowledge_value = note_activation(note)
        # Temporal value reorders genuinely relevant hits; it never makes an
        # unrelated note match and never hides a low-activation note completely.
        score *= 0.75 + 0.5 * knowledge_value
        item = {k: v for k, v in note.items() if k != "content"}
        item["snippet"] = _snippet(note["content"], list(overlap))
        item["score"] = round(score, 4)
        item["knowledge_value"] = knowledge_value
        item["relevance_score"] = relevance["score"] if relevance["passed"] else 0.66
        item["matched_terms"] = relevance["matched_terms"]
        item["relevance_reason"] = (
            relevance["reason"] if relevance["passed"] else "命中教材中低频且具有区分度的专业概念"
        )
        item["knowledge_status"] = local_knowledge_status(note)
        scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:limit]]


def search_notes(
    store: GardenStore, query: str, *, kinds: set[str] | None = None, limit: int = 5,
    strict_relevance: bool = True, query_plan: dict[str, Any] | None = None,
    max_queries: int = 3,
    semantic_enabled: bool | None = None,
    rerank_enabled: bool | None = None,
) -> list[dict[str, Any]]:
    """Run structured multi-query lexical/vector retrieval, then rerank."""
    candidate_limit = max(20, limit * 4)
    plan = query_plan or build_query_plan(query)
    queries = list(plan.get("queries") or [{"text": query, "source": "original", "weight": 1.0}])
    strategy = str(plan.get("strategy") or "")
    strategy_limit = {
        "single_query": 1,
        "resolved_followup": 1,
        "decompose": 1,
        "semantic_rewrite": 2,
        "bilingual_expand": 2,
    }
    query_budget = min(max_queries, strategy_limit.get(strategy, max(1, max_queries)))
    foundational = _is_foundational_plan(plan)
    foundation_fields = {str(field) for field in plan.get("foundation_fields", [])}
    if foundational and query_budget < 2:
        query_budget = min(2, max(1, max_queries))
    queries = queries[:query_budget]
    notes = store.list_notes(limit=10_000)
    if not _requested_textbook_exists(str(plan.get("resolved") or query), notes):
        return []
    if foundational and _missing_foundation_evidence(
        str(plan.get("resolved") or query), foundation_fields, notes,
        aliases=[str(alias) for alias in plan.get("aliases", [])],
    ):
        return []
    note_by_path = {str(note["path"]): note for note in notes if not kinds or note["kind"] in kinds}
    fused: dict[str, dict[str, Any]] = {}
    if semantic_enabled is None:
        semantic_enabled = os.getenv("GARDEN_DISABLE_SEMANTIC", "").strip().lower() not in {"1", "true", "yes"}
    for query_index, variant in enumerate(queries):
        variant_text = str(variant.get("text") or "").strip()
        source = str(variant.get("source") or "rewrite")
        weight = max(0.1, min(1.0, float(variant.get("weight", 1.0))))
        if not variant_text:
            continue
        lexical = _search_notes_lexical(
            store, variant_text, kinds=kinds, limit=candidate_limit,
            strict_relevance=strict_relevance if query_index == 0 else False,
            notes=notes,
        )
        for rank, hit in enumerate(lexical, 1):
            path = str(hit["path"])
            item = fused.setdefault(path, {**hit, "fusion_score": 0.0, "query_matches": []})
            item_kind = str(item.get("kind") or "")
            item_source = str(item.get("source") or "")
            kind_weight = _target_kind_weight(item_kind, item_source) if foundational else 1.0
            if foundational:
                source_note = note_by_path.get(path, item)
                kind_weight *= _textbook_navigation_weight(source_note)
                kind_weight *= _foundation_domain_weight(source_note, foundation_fields)
            item["fusion_score"] += weight * kind_weight / (60 + rank)
            item["query_matches"].append({"source": source, "channel": "lexical", "rank": rank})
            if "lexical_rank" not in item or rank < item["lexical_rank"]:
                item["lexical_rank"] = rank
        semantic: list[dict[str, Any]] = []
        if semantic_enabled:
            try:
                from core.semantic_index import semantic_search

                semantic = semantic_search(
                    variant_text, limit=candidate_limit, kinds=kinds, store_notes=notes,
                )
            except Exception:
                semantic = []
        for rank, hit in enumerate(semantic, 1):
            path = str(hit["path"])
            note = note_by_path.get(path)
            if note is None:
                continue
            item = fused.get(path)
            if item is None:
                item = {key: value for key, value in note.items() if key != "content"}
                note_kind = str(note.get("kind") or "")
                item.update({
                    "snippet": hit["text"], "score": 0.0, "knowledge_value": note_activation(note),
                    "relevance_score": hit["semantic_score"], "matched_terms": ["语义向量匹配"],
                    "relevance_reason": "多语言或同义语义匹配", "knowledge_status": local_knowledge_status(note),
                    "fusion_score": 0.0, "query_matches": [],
                })
                fused[path] = item
                item_kind = note_kind
            else:
                item_kind = str(item.get("kind") or note.get("kind") or "")
            source_type = str(note.get("source") or "")
            if foundational:
                kind_weight = _target_kind_weight(item_kind, source_type)
                kind_weight *= _textbook_navigation_weight(note)
                kind_weight *= _foundation_domain_weight(note, foundation_fields)
            else:
                kind_weight = 1.0
            item["fusion_score"] += 1.15 * weight * kind_weight / (60 + rank)
            item["query_matches"].append({"source": source, "channel": "semantic", "rank": rank})
            old_score = float(item.get("semantic_score", -1.0))
            if hit["semantic_score"] > old_score:
                item["semantic_rank"] = rank
                item["semantic_score"] = hit["semantic_score"]
                item["semantic_snippet"] = hit["text"]
                item["snippet"] = hit["text"]

    if not fused and strategy in {"single_query", "resolved_followup"}:
        for fallback_text in _fallback_queries(plan, query):
            fallback_clean = normalize_query(fallback_text)
            if not fallback_clean:
                continue
            lexical = _search_notes_lexical(
                store, fallback_clean, kinds=kinds, limit=candidate_limit,
                strict_relevance=False,
                notes=notes,
            )
            for rank, hit in enumerate(lexical, 1):
                path = str(hit["path"])
                item = fused.setdefault(path, {**hit, "fusion_score": 0.0, "query_matches": []})
                item_kind = str(item.get("kind") or "")
                item_source = str(item.get("source") or "")
                kind_weight = _target_kind_weight(item_kind, item_source) if foundational else 1.0
                if foundational:
                    source_note = note_by_path.get(path, item)
                    kind_weight *= _textbook_navigation_weight(source_note)
                    kind_weight *= _foundation_domain_weight(source_note, foundation_fields)
                item["fusion_score"] += 0.85 * kind_weight / (80 + rank)
                item["query_matches"].append({
                    "source": "fallback", "channel": "lexical", "rank": rank,
                })
                if "lexical_rank" not in item or rank < item["lexical_rank"]:
                    item["lexical_rank"] = rank
            if semantic_enabled:
                try:
                    from core.semantic_index import semantic_search

                    semantic = semantic_search(fallback_clean, limit=candidate_limit, kinds=kinds, store_notes=notes)
                except Exception:
                    semantic = []
            else:
                semantic = []
            for rank, hit in enumerate(semantic, 1):
                path = str(hit["path"])
                note = note_by_path.get(path)
                if note is None:
                    continue
                item = fused.get(path)
                if item is None:
                    item = {key: value for key, value in note.items() if key != "content"}
                    item.update({
                        "snippet": hit["text"], "score": 0.0, "knowledge_value": note_activation(note),
                        "relevance_score": hit["semantic_score"], "matched_terms": ["语义向量匹配"],
                        "relevance_reason": "多语言或同义语义匹配", "knowledge_status": local_knowledge_status(note),
                        "fusion_score": 0.0, "query_matches": [],
                    })
                    fused[path] = item
                note_kind = str(note.get("kind") or "")
                source_type = str(note.get("source") or "")
                kind_weight = _target_kind_weight(note_kind, source_type) if foundational else 1.0
                if foundational:
                    kind_weight *= _textbook_navigation_weight(note)
                    kind_weight *= _foundation_domain_weight(note, foundation_fields)
                item["fusion_score"] += 1.0 * kind_weight / (80 + rank)
                item["query_matches"].append({"source": "fallback", "channel": "semantic", "rank": rank})
                old_score = float(item.get("semantic_score", -1.0))
                if hit["semantic_score"] > old_score:
                    item["semantic_rank"] = rank
                    item["semantic_score"] = hit["semantic_score"]
                    item["semantic_snippet"] = hit["text"]
                    item["snippet"] = hit["text"]

    if rerank_enabled is None:
        rerank_enabled = os.getenv("GARDEN_DISABLE_RERANKER", "").strip().lower() not in {"1", "true", "yes"}
    if foundational:
        for item in fused.values():
            if item.get("relevance_reason") == "命中教材中低频且具有区分度的专业概念":
                # Cross-encoders favor broad question wording and can bury the
                # only OCR page containing a rare two-character definition.
                # Keep that auditable lexical evidence visible after RRF.
                item["specialist_concept_bonus"] = 0.022
                item["fusion_score"] += item["specialist_concept_bonus"]
            consensus = _channel_consensus_bonus(item.get("query_matches") or [])
            if consensus:
                item["channel_consensus_bonus"] = consensus
                item["fusion_score"] += consensus
    ranked = sorted(fused.values(), key=lambda item: item["fusion_score"], reverse=True)
    if rerank_enabled:
        try:
            from core.reranker import rerank_candidates

            configured = int(os.getenv("GARDEN_RERANK_CANDIDATES", "16"))
            rerank_limit = min(len(ranked), max(limit, max(4, configured)))
            candidates, remainder = _diverse_rerank_candidates(ranked, limit=rerank_limit)
            ranked = rerank_candidates(_rerank_query(plan, query), candidates) + remainder
        except Exception:
            # Reranking is an optional precision layer. Retrieval remains
            # available on machines where its local model is not installed.
            pass
    for item in ranked[:limit]:
        item["query_plan"] = plan
    return ranked[:limit]


def build_semantic_links(store: GardenStore, *, max_notes: int = 180) -> int:
    notes = store.list_notes(limit=max_notes)
    token_sets = {note["id"]: set(tokenize(note["title"] + " " + note["content"][:2500])) for note in notes}
    created = 0
    for index, left in enumerate(notes):
        candidates: list[tuple[float, dict[str, Any], set[str]]] = []
        left_tokens = token_sets[left["id"]]
        if not left_tokens:
            continue
        for right in notes[index + 1:]:
            if left["kind"] == right["kind"] and left["kind"] not in {"frontier", "interest"}:
                continue
            shared = left_tokens & token_sets[right["id"]]
            if len(shared) < 2:
                continue
            similarity = len(shared) / math.sqrt(len(left_tokens) * len(token_sets[right["id"]]))
            if similarity >= 0.08:
                candidates.append((similarity, right, shared))
        for similarity, right, shared in sorted(candidates, reverse=True, key=lambda x: x[0])[:2]:
            keywords = "、".join(sorted(shared, key=len, reverse=True)[:4])
            store.add_semantic_link(
                left["id"], right["id"], right["title"],
                f"候选连接基于共同概念：{keywords}", min(1.0, similarity * 3),
                evidence=[f"{left['title']} 与 {right['title']} 共同出现：{keywords}"], status="proposed",
            )
            created += 1
    return created


def ingest_pdf_directory(pdf_dir: str | Path, store: GardenStore, *, max_pages: int | None = None) -> dict[str, Any]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("导入 PDF 需要安装 pypdf：pip install pypdf") from exc
    root = Path(pdf_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("教材目录不存在")
    previous_manifest = store.setting("textbook_file_manifest", {}) or {}
    next_manifest = dict(previous_manifest)
    files = pages = changed = failed = skipped = processed = 0
    errors: list[dict[str, str]] = []
    for pdf_path in root.rglob("*.pdf"):
        files += 1
        key = str(pdf_path.resolve())
        try:
            stat = pdf_path.stat()
            fingerprint = f"{stat.st_size}:{stat.st_mtime_ns}"
            if previous_manifest.get(key) == fingerprint:
                skipped += 1
                continue
            processed += 1
            reader = PdfReader(str(pdf_path))
            for page_index, page in enumerate(reader.pages):
                if max_pages is not None and pages >= max_pages:
                    break
                text = clean_pdf_text(page.extract_text() or "")
                if len(text) < 30:
                    continue
                pages += 1
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                _, is_changed = store.upsert_note({
                    "path": f"pdf::{pdf_path}#page={page_index + 1}",
                    "title": f"{pdf_path.stem} · 第 {page_index + 1} 页",
                    "kind": "textbook",
                    "content": text,
                    "tags": ["教材", pdf_path.stem],
                    "source": "pdf",
                    "source_url": str(pdf_path),
                    "content_hash": digest,
                })
                changed += int(is_changed)
            next_manifest[key] = fingerprint
        except Exception as exc:
            failed += 1
            errors.append({"file": pdf_path.name, "error": f"{exc.__class__.__name__}: {str(exc)[:160]}"})
    existing = {str(path.resolve()) for path in root.rglob("*.pdf")}
    next_manifest = {key: value for key, value in next_manifest.items() if key in existing or key not in previous_manifest}
    store.set_setting("textbook_file_manifest", next_manifest)
    store.set_setting("textbook_directory", str(root))
    store.add_activity(
        "ingest_textbooks", f"发现 {files} 本；处理 {processed} 本、跳过未变化 {skipped} 本、提取 {pages} 页",
        min(30, changed // 10),
    )
    domains = rebuild_domain_map(store) if processed else 0
    return {
        "files": files, "processed": processed, "skipped": skipped, "pages": pages,
        "changed": changed, "failed": failed, "errors": errors, "domains": domains,
        "directory": str(root),
    }


DOMAIN_TAXONOMY = [
    {
        "root": "数学",
        "branch": "数学分析与微积分",
        "signals": ["微积分", "极限", "导数", "积分", "calculus", "derivative", "integral"],
        "topics": {
            "极限与连续": ["极限", "连续", "lim", "limit", "continuity"],
            "一元微分学": ["导数", "微分", "中值定理", "泰勒", "derivative", "differenti"],
            "一元积分学": ["积分", "原函数", "微积分基本定理", "integral", "antiderivative"],
            "多元微积分": ["多元", "偏导", "梯度", "重积分", "曲面积分", "partial derivative", "gradient"],
            "无穷级数": ["级数", "幂级数", "收敛", "傅里叶级数", "series", "convergence"],
            "常微分方程": ["微分方程", "常微分", "ode", "differential equation"],
        },
    },
    {
        "root": "物理学",
        "branch": "经典力学",
        "signals": ["力学", "牛顿", "动量", "mechanics", "newton", "momentum"],
        "topics": {
            "运动学": ["位移", "速度", "加速度", "kinematic", "velocity", "acceleration"],
            "牛顿动力学": ["牛顿", "合力", "质量", "newton", "force", "mass"],
            "功与能": ["动能", "势能", "机械能", "work", "energy", "potential"],
            "动量与碰撞": ["动量", "冲量", "碰撞", "momentum", "impulse", "collision"],
            "角动量与转动": ["角动量", "转动惯量", "力矩", "angular momentum", "torque", "rotation"],
            "振动与简谐运动": ["振动", "简谐", "oscillation", "harmonic"],
            "刚体力学": ["刚体", "rigid body"],
            "万有引力": ["引力", "开普勒", "gravitation", "kepler"],
        },
    },
]


CURRICULUM_POINTS = {
    "极限与连续": {
        "数列极限": ["数列极限", "sequence limit"], "函数极限": ["函数极限", "limit of a function"],
        "连续与间断": ["连续函数", "间断点", "continuity"], "无穷小与等价量": ["无穷小", "等价无穷小"],
    },
    "一元微分学": {
        "导数与微分": ["导数", "微分", "derivative"], "中值定理": ["中值定理", "mean value theorem"],
        "泰勒公式": ["泰勒", "taylor"], "函数单调性与极值": ["单调", "极值", "extremum"],
    },
    "一元积分学": {
        "不定积分": ["不定积分", "原函数", "antiderivative"], "定积分": ["定积分", "definite integral"],
        "微积分基本定理": ["微积分基本定理", "fundamental theorem"], "反常积分": ["反常积分", "improper integral"],
    },
    "多元微积分": {
        "偏导数": ["偏导", "partial derivative"], "梯度与方向导数": ["梯度", "方向导数", "gradient"],
        "重积分": ["重积分", "double integral", "triple integral"], "曲线与曲面积分": ["曲线积分", "曲面积分"],
    },
    "无穷级数": {
        "数项级数": ["数项级数"], "收敛判别": ["收敛判别", "convergence test"],
        "幂级数": ["幂级数", "power series"], "傅里叶级数": ["傅里叶级数", "fourier series"],
    },
    "常微分方程": {
        "一阶微分方程": ["一阶微分方程", "first-order"], "线性微分方程": ["线性微分方程", "linear differential"],
    },
    "运动学": {
        "位移、速度与加速度": ["位移", "速度", "加速度", "velocity"], "抛体运动": ["抛体", "projectile"],
    },
    "牛顿动力学": {
        "牛顿运动定律": ["牛顿运动定律", "newton's law"], "受力分析": ["受力", "free-body"],
    },
    "功与能": {
        "动能定理": ["动能定理", "kinetic energy"], "势能与能量守恒": ["势能", "能量守恒", "potential energy"],
    },
    "动量与碰撞": {
        "动量守恒": ["动量守恒", "momentum conservation"], "冲量": ["冲量", "impulse"], "碰撞": ["碰撞", "collision"],
    },
    "角动量与转动": {
        "角动量守恒": ["角动量守恒", "angular momentum"], "力矩": ["力矩", "torque"], "转动惯量": ["转动惯量", "moment of inertia"],
    },
    "振动与简谐运动": {"简谐运动": ["简谐", "simple harmonic"], "振动周期": ["振动周期", "period of oscillation"]},
    "刚体力学": {"质心运动": ["质心", "center of mass"], "刚体转动": ["刚体转动", "rigid body rotation"]},
    "万有引力": {"万有引力定律": ["万有引力", "universal gravitation"], "开普勒定律": ["开普勒", "kepler"]},
}


def _term_hits(text: str, terms: list[str]) -> int:
    lowered = text.lower()
    return sum(lowered.count(term.lower()) for term in terms)


GENERIC_DISCIPLINE_HINTS = [
    {
        "signals": ["电路", "电压", "电流", "阻抗", "circuit", "voltage", "current", "kirchhoff"],
        "discipline": "电气与电子工程", "branch": "电路与系统",
        "topics": [
            ("基础概念与电阻电路", ["basic concepts", "resistive circuits", "ohm", "kirchhoff", "series", "parallel"]),
            ("节点、回路与网络定理", ["nodal", "loop analysis", "node-voltage", "mesh", "superposition", "thevenin", "norton"]),
            ("运算放大器", ["operational amplifiers", "inverting", "noninverting", "op amp"]),
            ("电容、电感与暂态", ["capacitance", "inductance", "transient", "first-order", "second-order"]),
            ("交流稳态与功率", ["ac steady-state", "phasor", "impedance", "power analysis", "power factor"]),
            ("耦合网络与多相电路", ["mutual inductance", "transformer", "polyphase", "three-phase", "coupled networks"]),
            ("频率响应与变换分析", ["variable-frequency", "frequency response", "laplace", "fourier", "two-port"]),
        ],
    },
    {
        "signals": ["arduino", "gpio", "pwm", "传感器", "串口", "机器人", "嵌入式", "运动控制"],
        "discipline": "工程与技术", "branch": "机器人与嵌入式系统",
        "topics": [
            ("基础 IO 与逻辑控制", ["gpio", "io", "逻辑控制"]),
            ("感知与通信", ["传感器", "通信协议", "串口", "i2c", "spi"]),
            ("运动控制", ["运动控制", "电机", "pwm"]),
        ],
    },
    {
        "signals": ["心理", "人格", "认知", "行为", "emotion", "cognitive", "psychology"],
        "discipline": "心理学", "branch": "心理过程与行为",
        "topics": [
            ("认知心理学", ["认知", "注意", "记忆", "cognitive"]),
            ("社会与文化心理学", ["社会心理", "文化心理", "群体", "social psychology"]),
        ],
    },
    {
        "signals": ["主客体", "认识论", "本体论", "哲学", "subject-object", "epistemology", "ontology"],
        "discipline": "哲学", "branch": "认识论与主客体关系",
        "topics": [("主客体关系", ["主客体", "主体", "客体"])],
    },
    {
        "signals": ["文学", "诗歌", "散文", "叙事", "landscape", "poetry", "literature", "nature writing"],
        "discipline": "文学与文化研究", "branch": "文学批评与自然书写",
        "topics": [("自然书写", ["自然书写", "风景", "landscape", "nature writing"])],
    },
]


def _valid_taxonomy_label(value: Any) -> str:
    label = re.sub(r"\s+", " ", str(value or "")).strip(" ：:，,。/\\")
    if not 2 <= len(label) <= 32 or re.search(r"第\s*\d+\s*页|\.pdf$", label, re.I):
        return ""
    return label


def _evidence_terms(raw: Any, content: str) -> list[str]:
    terms = [str(item).strip() for item in (raw or []) if 2 <= len(str(item).strip()) <= 60]
    lowered = content.lower()
    return [item for item in terms if item.lower() in lowered][:5]


def _pdf_page_number(note: dict[str, Any]) -> int:
    match = re.search(r"#page=(\d+)$", str(note.get("path") or ""))
    return int(match.group(1)) if match else 1_000_000_000


def classify_textbook_structure(title: str, content: str) -> dict[str, Any]:
    """Build a reusable curriculum tree from one textbook's actual content."""
    result = None
    taxonomy_model_disabled = os.getenv("GARDEN_DISABLE_TAXONOMY_MODEL", "").strip().lower() in {"1", "true", "yes"}
    try:
        if taxonomy_model_disabled:
            raise LLMError("教材结构模型已显式关闭")
        result = chat_json(
            "你是教材知识结构分析专家。目标不是复刻目录，而是回答‘这本教材实际教了什么’，并从粗到细生成 学科→分支→学习方向→知识点 四层知识树。先判断学科，再识别具有独立核心问题和方法的分支，再把跨章节主题组织为学习方向，最后抽取正文明确讲授的概念、定理、方法或模型。学科通常对应独立院系与研究对象；分支是学科内的主要子领域；学习方向有清晰知识边界；知识点是一个可独立讲授和复习的单元。父节点必须比子节点更宽泛，子节点必须确实展开父节点；同层节点应尽量互斥，若‘微分方程’与‘偏微分方程’同层，应把前者收窄为‘常微分方程’。目录、篇章标题可用于定位候选，但不能单独证明教材真正讲授了该节点，必须同时有正文中的定义、首段论述、反复术语或总结句作为证据。每个节点至少提供3个能在给定正文中逐字找到的 evidence_terms；不足3个就让该层为空，禁止凑数。节点名必须是知识性名词短语，不能是教材名、章节号、页码、练习题、案例名、问题或完整句子。导论教材偏向覆盖广度，高级教材或专著偏向精确深度。最多2个学科、每学科3个分支、每分支6个方向、每方向8个知识点；教材没讲的不推测。",
            f"教材：{title}\n目录与代表页：\n{content[:18000]}\n\n"
            "返回 {\"disciplines\":[{\"name\":\"学科\",\"evidence_terms\":[\"至少3个正文术语\"],\"branches\":[{\"name\":\"分支\",\"evidence_terms\":[\"至少3个正文术语\"],\"topics\":[{\"name\":\"方向\",\"evidence_terms\":[\"至少3个正文术语\"],\"knowledge_points\":[{\"name\":\"知识点\",\"evidence_terms\":[\"至少3个正文术语\"]}]}]}]}],\"structure_analysis\":\"200字以内说明层级密度、侧重和证据不足处\"}。",
            timeout=35,
            max_retries=0,
        )
    except LLMError:
        result = None
    disciplines = []
    for raw_discipline in (result or {}).get("disciplines") or []:
        if not isinstance(raw_discipline, dict):
            continue
        discipline = _valid_taxonomy_label(raw_discipline.get("name"))
        d_evidence = _evidence_terms(raw_discipline.get("evidence_terms"), content)
        if not discipline or len(d_evidence) < 3:
            continue
        branches = []
        for raw_branch in (raw_discipline.get("branches") or [])[:3]:
            if not isinstance(raw_branch, dict):
                continue
            branch = _valid_taxonomy_label(raw_branch.get("name"))
            b_evidence = _evidence_terms(raw_branch.get("evidence_terms"), content)
            if not branch or len(b_evidence) < 3:
                continue
            topics = []
            for raw_topic in (raw_branch.get("topics") or [])[:6]:
                if not isinstance(raw_topic, dict):
                    continue
                topic = _valid_taxonomy_label(raw_topic.get("name"))
                t_evidence = _evidence_terms(raw_topic.get("evidence_terms"), content)
                if not topic or len(t_evidence) < 3:
                    continue
                points = []
                for raw_point in (raw_topic.get("knowledge_points") or [])[:8]:
                    if not isinstance(raw_point, dict):
                        continue
                    point = _valid_taxonomy_label(raw_point.get("name"))
                    p_evidence = _evidence_terms(raw_point.get("evidence_terms"), content)
                    if point and len(p_evidence) >= 3:
                        points.append({"name": point, "evidence_terms": p_evidence})
                topics.append({"name": topic, "evidence_terms": t_evidence, "knowledge_points": points})
            branches.append({"name": branch, "evidence_terms": b_evidence, "topics": topics})
        disciplines.append({"name": discipline, "evidence_terms": d_evidence, "branches": branches})
        if len(disciplines) >= 2:
            break
    if disciplines:
        return {"method": "langchain", "disciplines": disciplines}

    lowered = content.lower()
    fallback = []
    for hint in GENERIC_DISCIPLINE_HINTS:
        evidence = [term for term in hint["signals"] if term.lower() in lowered]
        if len(evidence) < 3:
            continue
        topics = []
        for topic, terms in hint["topics"]:
            matches = [term for term in terms if term.lower() in lowered]
            if len(matches) >= 3:
                topics.append({"name": topic, "evidence_terms": matches, "knowledge_points": []})
        fallback.append({
            "name": hint["discipline"], "evidence_terms": evidence[:5],
            "branches": [{"name": hint["branch"], "evidence_terms": evidence[:5], "topics": topics}],
        })
    return {"method": "evidence_fallback" if fallback else "unresolved", "disciplines": fallback[:2]}


def rebuild_domain_map(store: GardenStore, *, force_model: bool = False) -> int:
    """Turn retrieval-only PDF pages into a compact discipline/branch/topic map."""
    pdf_notes = [note for note in store.list_notes(limit=5000) if note["source"] == "pdf"]
    store.clear_notes_by_source("derived_domain")
    if not pdf_notes:
        return 0
    combined = "\n".join(note["content"][:5000] for note in pdf_notes)
    created: dict[str, int] = {}

    def ensure_node(title: str, level: str, content: str, tags: list[str]) -> int:
        note_id, _ = store.upsert_note({
            "path": f"domain::{level}::{title}", "title": title, "kind": "domain",
            "content": content, "tags": tags, "source": "derived_domain",
            "content_hash": hashlib.sha256((title + content).encode("utf-8")).hexdigest(),
        })
        created[title] = note_id
        return note_id

    for taxonomy in DOMAIN_TAXONOMY:
        signal_score = _term_hits(combined, taxonomy["signals"])
        if signal_score == 0:
            continue
        root_id = ensure_node(
            taxonomy["root"], "discipline", f"由已导入教材归纳出的学科主干：{taxonomy['root']}。",
            ["学科", taxonomy["root"]],
        )
        branch_id = ensure_node(
            taxonomy["branch"], "branch", f"{taxonomy['root']}中的主要学习分支，来源于本地教材内容。",
            [taxonomy["root"], "知识分支"],
        )
        store.add_structural_link(root_id, branch_id, taxonomy["branch"], "contains", "学科包含知识分支")
        topic_scores = []
        for topic, terms in taxonomy["topics"].items():
            score = _term_hits(combined, terms)
            if score:
                topic_scores.append((score, topic, terms))
        for score, topic, terms in sorted(topic_scores, reverse=True)[:8]:
            evidence_notes = []
            for note in pdf_notes:
                if any(term.lower() in note["content"].lower() for term in terms):
                    evidence_notes.append(note["title"])
                if len(evidence_notes) == 3:
                    break
            topic_id = ensure_node(
                topic, "topic",
                f"方向：{topic}。教材检索证据：" + "、".join(evidence_notes),
                [taxonomy["root"], taxonomy["branch"], "学习方向"],
            )
            store.add_structural_link(
                branch_id, topic_id, topic, "contains", f"教材内容命中 {score} 次；来源页仅作为检索证据。",
                min(1.0, 0.55 + math.log1p(score) / 10),
            )
            for point, point_terms in CURRICULUM_POINTS.get(topic, {}).items():
                point_score = _term_hits(combined, point_terms)
                if not point_score:
                    continue
                point_evidence = []
                for note in pdf_notes:
                    if any(term.lower() in note["content"].lower() for term in point_terms):
                        point_evidence.append(note["title"])
                    if len(point_evidence) == 2:
                        break
                point_id = ensure_node(
                    point, "knowledge", f"已学知识点：{point}。来源证据：" + "、".join(point_evidence),
                    [taxonomy["root"], taxonomy["branch"], topic, "知识点"],
                )
                # Reclassify the generated leaf while keeping it out of PDF-page clutter.
                with store.connect() as conn:
                    conn.execute("UPDATE notes SET kind='knowledge' WHERE id=?", (point_id,))
                store.add_structural_link(topic_id, point_id, point, "contains", "学习方向包含知识点")

    # The static maps above are a dependable offline baseline for mathematics
    # and mechanics.  Every textbook is also classified independently so future
    # disciplines do not require a code change.
    books: dict[str, list[dict[str, Any]]] = {}
    for note in pdf_notes:
        key = str(note.get("source_url") or note["title"].split(" · 第", 1)[0])
        books.setdefault(key, []).append(note)
    cache = store.setting("textbook_taxonomy_cache_v2", {}) or {}
    next_cache: dict[str, Any] = {}
    for book_key, pages_for_book in books.items():
        # Sorting raw paths puts page 100 before page 11 and caused the table
        # of contents to be skipped. Taxonomy sampling must follow PDF order.
        pages_for_book.sort(key=_pdf_page_number)
        book_title = pages_for_book[0]["title"].split(" · 第", 1)[0]
        # Early pages usually contain the table of contents; a few evenly
        # spaced pages add evidence for later chapters without sending a book.
        indexes = list(range(min(12, len(pages_for_book))))
        if len(pages_for_book) > 12:
            indexes.extend(sorted({len(pages_for_book) // 3, len(pages_for_book) * 2 // 3, len(pages_for_book) - 1}))
        selected_pages = [pages_for_book[index] for index in indexes]
        sample = "\n\n".join(
            f"[{item['title']}]\n{item['content'][:1600]}" for item in selected_pages
        )[:24000]
        signature = hashlib.sha256((book_title + sample).encode("utf-8")).hexdigest()
        cached = cache.get(book_key) if isinstance(cache, dict) else None
        if (
            isinstance(cached, dict) and cached.get("signature") == signature
            and (
                (cached.get("structure") or {}).get("method") == "langchain"
                or not force_model
            )
        ):
            structure = cached.get("structure") or {"method": "unresolved", "disciplines": []}
        else:
            structure = classify_textbook_structure(book_title, sample)
        next_cache[book_key] = {"signature": signature, "structure": structure}

        for discipline in structure.get("disciplines") or []:
            discipline_name = _valid_taxonomy_label(discipline.get("name"))
            if not discipline_name:
                continue
            d_evidence = "、".join(discipline.get("evidence_terms") or [])
            root_id = ensure_node(
                discipline_name, "discipline",
                f"由教材《{book_title}》正文归纳出的学科主干。分类证据：{d_evidence}。",
                ["学科", discipline_name, "教材归纳"],
            )
            for branch in discipline.get("branches") or []:
                branch_name = _valid_taxonomy_label(branch.get("name"))
                if not branch_name:
                    continue
                b_evidence = "、".join(branch.get("evidence_terms") or [])
                branch_id = ensure_node(
                    branch_name, "branch",
                    f"{discipline_name}下的知识分支。教材：《{book_title}》；正文证据：{b_evidence}。",
                    [discipline_name, "知识分支", "教材归纳"],
                )
                store.add_structural_link(
                    root_id, branch_id, branch_name, "contains",
                    f"教材《{book_title}》中的证据词：{b_evidence}", 0.82,
                )
                for topic in branch.get("topics") or []:
                    topic_name = _valid_taxonomy_label(topic.get("name"))
                    if not topic_name:
                        continue
                    t_evidence = "、".join(topic.get("evidence_terms") or [])
                    topic_id = ensure_node(
                        topic_name, "topic",
                        f"学习方向：{topic_name}。教材：《{book_title}》；正文证据：{t_evidence}。",
                        [discipline_name, branch_name, "学习方向", "教材归纳"],
                    )
                    store.add_structural_link(
                        branch_id, topic_id, topic_name, "contains",
                        f"教材《{book_title}》中的证据词：{t_evidence}", 0.8,
                    )
                    for point in topic.get("knowledge_points") or []:
                        point_name = _valid_taxonomy_label(point.get("name"))
                        if not point_name:
                            continue
                        p_evidence = "、".join(point.get("evidence_terms") or [])
                        point_id = ensure_node(
                            point_name, "knowledge",
                            f"已学知识点：{point_name}。教材：《{book_title}》；正文证据：{p_evidence}。",
                            [discipline_name, branch_name, topic_name, "知识点", "教材归纳"],
                        )
                        with store.connect() as conn:
                            conn.execute("UPDATE notes SET kind='knowledge' WHERE id=?", (point_id,))
                        store.add_structural_link(
                            topic_id, point_id, point_name, "contains",
                            f"教材《{book_title}》正文明确涉及：{p_evidence}", 0.78,
                        )
    store.set_setting("textbook_taxonomy_cache_v2", next_cache)
    store.add_activity("rebuild_domain_map", f"归纳 {len(created)} 个学科/分支/方向节点", 5 if created else 0)
    return len(created)
