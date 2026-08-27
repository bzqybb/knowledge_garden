from __future__ import annotations

import hashlib
import json
import re
from difflib import SequenceMatcher
from datetime import datetime, timezone
from typing import Any

from core.llm import LLMError, chat_json
from core.learning_memory import LearningMemoryService
from core.retrieval import search_notes
from core.storage import GardenStore
from core.web_research import fetch_open_access_pdf_text


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip('“”"\'')


def _select_sections(text: str, limit: int = 30_000) -> str:
    """Keep the abstract/opening plus method/result/discussion neighborhoods."""
    clean = re.sub(r"\r\n?", "\n", str(text or ""))
    if len(clean) <= limit:
        return clean
    pieces = [clean[:10_000]]
    lowered = clean.casefold()
    patterns = (
        "introduction", "background", "method", "materials and methods", "results",
        "discussion", "limitations", "conclusion", "摘要", "引言", "方法", "结果", "讨论", "局限", "结论",
    )
    used: list[tuple[int, int]] = [(0, 10_000)]
    for heading in patterns:
        position = lowered.find(heading.casefold(), 500)
        if position < 0:
            continue
        start = max(0, position - 350)
        end = min(len(clean), position + 4_000)
        if any(max(start, old_start) < min(end, old_end) for old_start, old_end in used):
            continue
        pieces.append(clean[start:end])
        used.append((start, end))
        if sum(len(item) for item in pieces) >= limit:
            break
    return "\n\n--- 选段 ---\n\n".join(pieces)[:limit]


def _fallback(title: str, source_text: str, scope: str) -> dict[str, Any]:
    sentence_parts = [
        part.strip() for part in re.split(r"(?<=[。！？.!?])\s+", source_text)
        if len(part.strip()) >= 30
    ]
    first = sentence_parts[0][:500] if sentence_parts else "当前来源没有足够文字可供可靠深读。"
    return {
        "problem": first,
        "novelty": "需要模型结合方法与结果段进一步判断；当前不从标题猜测创新点。",
        "method": "当前来源未形成可验证的方法摘要。",
        "concepts": [],
        "findings": [],
        "limitations": ["当前为离线提取结果，尚未完成模型级深读。"],
        "prerequisites": [],
        "reading_routes": {
            "ten_minutes": ["先读摘要并圈出研究对象、比较基准和结论。"],
            "thirty_minutes": ["再核对方法、结果图表和局限性。"],
        },
        "questions": [f"《{title}》最关键的证据是否足以支持其核心结论？"],
        "confidence": 0.25 if scope != "metadata_only" else 0.05,
    }


def _short_list(value: Any, *, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for raw in value:
        item = re.sub(r"\s+", " ", str(raw or "")).strip()
        if 2 <= len(item) <= 80 and item not in items:
            items.append(item)
        if len(items) >= limit:
            break
    return items


def _mastery_match(store: GardenStore, concept: str) -> dict[str, Any] | None:
    """Return observed mastery evidence; never infer learning from retrieval alone."""
    try:
        mastery = LearningMemoryService(store).overview().get("concept_mastery", [])
    except Exception:
        return None
    needle = _compact(concept).casefold()
    if len(needle) < 2:
        return None
    for item in mastery if isinstance(mastery, list) else []:
        key = str(item.get("concept_key") or "").strip()
        compact_key = _compact(key).casefold()
        if len(compact_key) < 2 or (needle not in compact_key and compact_key not in needle):
            continue
        return {
            "concept": key,
            "stage": str(item.get("stage") or "exposed"),
            "confidence": round(float(item.get("confidence") or 0.0), 3),
            "evidence": "来自复习/作答记录中的概念掌握证据",
        }
    return None


_GENERIC_CONNECTION_TERMS = {
    "analysis", "learning", "effect", "principle", "relation", "path", "force", "deep",
    "regions", "generation", "design", "model", "data", "method", "system", "network",
    "neural", "fixed", "parameters", "of", "implicit", "representations", "fields", "super",
    "zero", "shot", "resolution", "分辨", "分辨率", "辨率", "神经",
}
_SAFE_SINGLE_CONNECTION_TERMS = {
    "fourier", "convolution", "transform", "spectrum", "operator", "integral", "derivative",
    "gradient", "matrix", "eigenvalue", "interference", "entropy", "傅里叶", "卷积", "变换",
    "频谱", "算子", "积分", "导数", "梯度", "矩阵", "特征值", "干涉", "熵",
}


def _rank_connection_concepts(concepts: list[str], limit: int = 6) -> list[str]:
    def specificity(item: str) -> tuple[float, int]:
        latin = re.findall(r"[A-Za-z][A-Za-z0-9'-]+", item)
        acronyms = sum(token.isupper() and 2 <= len(token) <= 8 for token in latin)
        non_generic = sum(token.casefold() not in _GENERIC_CONNECTION_TERMS for token in latin)
        chinese = sum(len(block) >= 3 for block in re.findall(r"[\u4e00-\u9fff]+", item))
        return (acronyms * 4 + non_generic * 0.8 + chinese * 1.5 + min(len(item), 60) / 60, -concepts.index(item))

    return sorted(concepts, key=specificity, reverse=True)[:limit]


def _supported_bridge_terms(concept: str, hit: dict[str, Any], snippet: str) -> list[str]:
    haystack = f"{hit.get('title', '')} {snippet}".casefold()
    matched = {
        str(item).casefold() for item in (hit.get("matched_terms") or [])
        if str(item).strip()
    }
    return sorted(
        term for term in matched - _GENERIC_CONNECTION_TERMS
        if term in concept.casefold() and term in haystack
    )


def _specific_connection_hit(concept: str, hit: dict[str, Any], snippet: str) -> bool:
    compact_concept = _compact(re.sub(r"\([^)]*\)", "", concept)).casefold()
    compact_haystack = _compact(f"{hit.get('title', '')} {snippet}").casefold()
    if len(compact_concept) >= 8 and compact_concept in compact_haystack:
        return True
    terms = _supported_bridge_terms(concept, hit, snippet)
    return len(terms) >= 2 or any(term in _SAFE_SINGLE_CONNECTION_TERMS for term in terms)


def _local_candidates(store: GardenStore, title: str, analysis: dict[str, Any]) -> list[dict[str, Any]]:
    concepts = _short_list(analysis.get("concepts"), limit=8)
    if not concepts:
        # The title is a discovery query, not proof that the user has learned it.
        concepts = [title[:80]] if len(_compact(title)) >= 2 else []
    concepts = _rank_connection_concepts(concepts, limit=6)
    allowed_kinds = {"textbook", "course", "concept", "knowledge", "knowledge_point", "bridge"}
    mastery = {concept: _mastery_match(store, concept) for concept in concepts}
    candidates: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for concept_index, concept in enumerate(concepts):
        try:
            hits = search_notes(
                store, concept, kinds=allowed_kinds, limit=2, strict_relevance=True,
                max_queries=1, semantic_enabled=concept_index < 2, rerank_enabled=False,
            )
        except Exception:
            hits = []
        for hit in hits:
            path = str(hit.get("path") or "")
            if not path or path in seen_paths or hit.get("knowledge_status") == "placeholder":
                continue
            snippet = re.sub(r"\s+", " ", str(hit.get("snippet") or "")).strip()[:520]
            if len(_compact(snippet)) < 20 or not _specific_connection_hit(concept, hit, snippet):
                continue
            seen_paths.add(path)
            candidates.append({
                "index": len(candidates) + 1,
                "paper_concept": concept,
                "title": str(hit.get("title") or "未命名本地资料"),
                "kind": str(hit.get("kind") or "knowledge"),
                "path": path,
                "source_url": str(hit.get("source_url") or ""),
                "snippet": snippet,
                "matched_terms": _short_list(hit.get("matched_terms"), limit=6),
                "retrieval_score": round(float(hit.get("relevance_score") or 0.0), 4),
                "mastery": mastery.get(concept),
            })
            if len(candidates) >= 8:
                return candidates
    return candidates


def _connect_local_knowledge(
    store: GardenStore, title: str, analysis: dict[str, Any], *, allow_model: bool = False,
) -> dict[str, Any]:
    """Knowledge Garden connector: retrieve locally, then validate every bridge."""
    candidates = _local_candidates(store, title, analysis)
    if not candidates:
        return {
            "connections": [], "candidate_count": 0,
            "note": "没有检索到足够相关的本地知识，因此没有强行关联教材。",
        }

    proposed: dict[str, Any] | None = None
    if allow_model:
        compact_a = {
            key: analysis.get(key) for key in ("problem", "novelty", "method", "concepts", "findings")
        }
        try:
            proposed = chat_json(
                "你是知识花园现有主链中的本地关联器。输入包含论文解读器生成的结构化论文卡和本地检索候选。"
                "只允许连接候选资料；资料片段不能直接支持关系时必须舍弃。不得把‘存在于本地库’写成‘用户已经掌握’；"
                "只有候选 mastery 非空时才可提到用户学过。只返回JSON。",
                "论文卡：\n" + json.dumps(compact_a, ensure_ascii=False) +
                "\n\n本地候选：\n" + json.dumps(candidates, ensure_ascii=False) +
                "\n\n返回 connections 数组（最多4项），每项包含 source_index、relation_type"
                "（prerequisite/analogy/method/application之一）、bridge、why_useful、confidence（0到1）。"
                "宁缺毋滥；没有直接关系就返回空数组。",
                timeout=65,
                max_retries=0,
            )
        except LLMError:
            proposed = None

    by_index = {item["index"]: item for item in candidates}
    accepted: list[dict[str, Any]] = []
    raw_connections = proposed.get("connections", []) if isinstance(proposed, dict) else []
    for raw in raw_connections if isinstance(raw_connections, list) else []:
        if not isinstance(raw, dict):
            continue
        try:
            source_index = int(raw.get("source_index"))
            confidence = max(0.0, min(1.0, float(raw.get("confidence") or 0.0)))
        except (TypeError, ValueError):
            continue
        source = by_index.get(source_index)
        bridge = re.sub(r"\s+", " ", str(raw.get("bridge") or "")).strip()
        why_useful = re.sub(r"\s+", " ", str(raw.get("why_useful") or "")).strip()
        relation_type = str(raw.get("relation_type") or "prerequisite").strip().lower()
        if source is None or confidence < 0.5 or len(_compact(bridge)) < 8:
            continue
        if relation_type not in {"prerequisite", "analogy", "method", "application"}:
            relation_type = "prerequisite"
        accepted.append({
            **source,
            "relation_type": relation_type,
            "bridge": bridge[:360],
            "why_useful": why_useful[:260],
            "confidence": round(confidence, 3),
        })
        if len(accepted) >= 4:
            break
    if not accepted and not allow_model:
        for source in candidates:
            matched = _supported_bridge_terms(
                str(source.get("paper_concept") or ""), source, str(source.get("snippet") or "")
            )
            if not matched:
                compact_concept = _compact(
                    re.sub(r"\([^)]*\)", "", str(source.get("paper_concept") or ""))
                ).casefold()
                compact_source = _compact(
                    f"{source.get('title', '')} {source.get('snippet', '')}"
                ).casefold()
                if len(compact_concept) < 8 or compact_concept not in compact_source:
                    continue
                matched = [str(source.get("paper_concept") or "")]
            keywords = "、".join(str(term) for term in matched[:3])
            accepted.append({
                **source,
                "relation_type": "prerequisite",
                "bridge": (
                    f"论文概念“{source['paper_concept']}”与本地《{source['title']}》"
                    f"共同涉及“{keywords}”，可把该资料作为前置概念或数学工具的复习入口。"
                ),
                "why_useful": "这里只确认共同概念和可追溯片段，不推断两者的结论等价。",
                "confidence": round(min(0.84, 0.7 + 0.06 * len(matched)), 3),
            })
            if len(accepted) >= 3:
                break
    return {
        "connections": accepted,
        "candidate_count": len(candidates),
        "note": (
            f"从 {len(candidates)} 条本地候选中保留 {len(accepted)} 条可解释连接。"
            if accepted else "检索到了本地候选，但没有关系通过知识花园现有证据门槛，因此未展示。"
        ),
    }


def _aligned_evidence(evidence: str, source_text: str) -> str:
    """Recover a source sentence from a close paraphrase without relaxing facts."""
    evidence_clean = re.sub(r"\s+", " ", evidence).strip()
    evidence_words = set(re.findall(r"[A-Za-z][A-Za-z0-9'-]+|\d+(?:\.\d+)?", evidence_clean.casefold()))
    evidence_numbers = set(re.findall(r"\d+(?:\.\d+)?", evidence_clean))
    if len(evidence_words) < 5:
        return ""
    normalized_source = re.sub(r"\s+", " ", source_text)
    sentences = [
        item.strip() for item in re.split(r"(?<=[.!?。！？])\s+", normalized_source)
        if 30 <= len(item.strip()) <= 700
    ]
    best = ""
    best_score = 0.0
    for sentence in sentences:
        if evidence_numbers and not evidence_numbers.issubset(set(re.findall(r"\d+(?:\.\d+)?", sentence))):
            continue
        words = set(re.findall(r"[A-Za-z][A-Za-z0-9'-]+|\d+(?:\.\d+)?", sentence.casefold()))
        overlap = len(evidence_words & words) / max(1, len(evidence_words))
        if overlap < 0.62:
            continue
        ratio = SequenceMatcher(None, evidence_clean.casefold(), sentence.casefold()).ratio()
        score = 0.68 * overlap + 0.32 * ratio
        if score > best_score:
            best, best_score = sentence, score
    return best[:700] if best_score >= 0.72 else ""


def _validated_findings(items: Any, source_text: str) -> list[dict[str, Any]]:
    compact_source = _compact(source_text)
    findings: list[dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim") or "").strip()
        evidence = str(item.get("evidence") or "").strip().strip('“”"')
        # Keep concise scientific evidence snippets in the audit result.  A
        # short unsupported quote is still useful: the UI can mark it as
        # ungrounded instead of silently hiding the model's overclaim.
        if not claim or len(_compact(evidence)) < 6:
            continue
        evidence_compact = _compact(evidence)
        exact = evidence_compact[:160] in compact_source
        aligned = "" if exact else _aligned_evidence(evidence, source_text)
        grounded = exact or bool(aligned)
        findings.append({
            "claim": claim,
            "evidence": evidence if exact else aligned,
            "grounded": grounded,
            "validation": "exact_quote" if exact else ("aligned_quote" if aligned else "rejected"),
        })
        if len(findings) >= 6:
            break
    return findings


def deep_read_paper(store: GardenStore, article: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    title = str(article.get("title") or "未命名论文").strip()
    url = str(article.get("url") or "").strip()
    pdf_url = str(article.get("pdf_url") or "").strip()
    abstract = str(article.get("abstract") or "").strip()
    fingerprint = hashlib.sha256(
        f"v7|{url}|{pdf_url}|{title}|{abstract[:400]}".encode("utf-8")
    ).hexdigest()
    cache = store.setting("paper_deep_read_cache_v7", {}) or {}
    if not force and isinstance(cache.get(fingerprint), dict):
        return {**cache[fingerprint], "cached": True}

    scope = "metadata_only"
    source_text = ""
    source_note = "只有论文元数据，不能可靠判断方法、结果或局限。"
    fulltext_error = ""
    if pdf_url:
        try:
            extracted = fetch_open_access_pdf_text(pdf_url)
            if len(_compact(extracted)) >= 800:
                source_text = extracted
                scope = "open_fulltext"
                source_note = "已读取开放获取 PDF 正文；关键主张仍应回到论文图表和原文核对。"
            else:
                fulltext_error = "开放PDF可提取文字过少"
        except Exception as exc:
            fulltext_error = str(exc) or exc.__class__.__name__
    if not source_text and abstract:
        source_text = abstract
        scope = "abstract"
        source_note = "本次仅依据摘要深读，无法验证完整方法、实验细节和作者全部局限说明。"

    selected = _select_sections(source_text)
    fallback = _fallback(title, selected, scope)
    result: dict[str, Any] | None = None
    if len(_compact(selected)) >= 120:
        try:
            result = chat_json(
                "你是基础学科前沿论文深读导师。只根据所给论文文本工作，禁止用常识补写论文没有陈述的实验、数字、结论或局限。只返回JSON。findings中每个claim必须配一条来自输入文本、可逐字核对的短evidence；如果当前只是摘要，明确降低置信度并说明哪些问题无法判断。解释要兼顾本科生理解和学术准确性。",
                f"学习水平：{store.setting('learning_level', '本科入门')}\n"
                f"用户兴趣：{'、'.join(store.setting('interests', []) or []) or '未设置'}\n"
                f"来源范围：{scope}\n标题：{title}\n期刊：{article.get('venue') or '未知'}\n"
                f"作者：{'、'.join(article.get('authors') or [])}\n论文选段：\n{selected}\n\n"
                "返回 problem、novelty、method、concepts（3~8个适合本地检索的具体术语）、"
                "findings（3~6项，每项含claim和逐字evidence）、"
                "limitations、prerequisites、reading_routes（含ten_minutes与thirty_minutes步骤数组）、"
                "questions（2~4项）、confidence（0到1）。",
                timeout=100,
                max_retries=1,
            )
        except LLMError:
            result = None

    analysis = dict(fallback)
    if isinstance(result, dict):
        for key in ("problem", "novelty", "method", "limitations", "prerequisites", "reading_routes", "questions"):
            if result.get(key):
                analysis[key] = result[key]
        analysis["concepts"] = _short_list(result.get("concepts"), limit=8)
        analysis["findings"] = _validated_findings(result.get("findings"), selected)
        try:
            confidence = max(0.0, min(1.0, float(result.get("confidence", fallback["confidence"]))))
        except (TypeError, ValueError):
            confidence = fallback["confidence"]
        if scope == "abstract":
            confidence = min(confidence, 0.68)
        elif scope == "metadata_only":
            confidence = min(confidence, 0.15)
        grounded = [item for item in analysis["findings"] if item.get("grounded")]
        if analysis["findings"] and len(grounded) < max(1, len(analysis["findings"]) // 2):
            confidence = min(confidence, 0.45)
        analysis["confidence"] = confidence

    local_bridge = _connect_local_knowledge(
        store, title, analysis,
        allow_model=bool(store.setting("paper_local_connector_remote_consent", False)),
    )
    analysis["local_connections"] = local_bridge["connections"]
    analysis["local_connection_note"] = local_bridge["note"]
    payload = {
        "title": title,
        "url": url,
        "pdf_url": pdf_url,
        "scope": scope,
        "scope_label": {
            "open_fulltext": "开放全文深读",
            "abstract": "摘要级深读",
            "metadata_only": "仅元数据",
        }[scope],
        "source_note": source_note,
        "fulltext_error": fulltext_error,
        "extracted_chars": len(_compact(source_text)),
        "analysis": analysis,
        "reading_pipeline": {
            "paper_reader": "论文正文/摘要 → 受证据约束的结构化论文卡",
            "garden_connector": "论文卡 → 现有本地混合检索与记忆主链 → 经来源编号校验的个性化连接",
            "local_candidate_count": local_bridge["candidate_count"],
            "accepted_connection_count": len(local_bridge["connections"]),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cached": False,
    }
    cache[fingerprint] = payload
    if len(cache) > 60:
        oldest = sorted(
            cache,
            key=lambda key: str((cache.get(key) or {}).get("generated_at") or ""),
        )[:-60]
        for key in oldest:
            cache.pop(key, None)
    store.set_setting("paper_deep_read_cache_v7", cache)
    store.add_activity("paper_deep_read", title[:100], 5)
    return payload
