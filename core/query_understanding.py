from __future__ import annotations

import os
import re
from typing import Any, Iterable


QUESTION_PREFIX = re.compile(r"^(?:请问|请解释|请说明|能否|可以帮我|我想知道)\s*")
SPACE_RE = re.compile(r"\s+")
LIGHT_QUERY_MAX_LEN = 26

# Auditable fallback terminology. It expands retrieval only; textbook evidence
# is still required before any alias may support a factual answer.
ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("基尔霍夫电流定律", "KCL", "Kirchhoff current law", "charge conservation"),
    ("基尔霍夫电压定律", "KVL", "Kirchhoff voltage law", "energy conservation"),
    ("节点电压", "node voltage", "node-voltage analysis", "reference node"),
    ("线性独立", "linearly independent", "independent equations", "N-1"),
    ("电容", "capacitor", "capacitance", "energy storage"),
    ("电感", "inductor", "inductance", "energy storage"),
    ("相机闪光灯", "camera flash", "camera's flash circuit", "xenon lamp"),
    ("能量储存", "energy storage", "stored energy", "capacitor"),
    ("单位正电荷", "unit positive charge", "give up energy", "less energy"),
    ("参考方向", "reference direction", "higher potential", "opposite reference direction"),
    ("矩阵的秩", "矩阵秩", "matrix rank", "rank of a matrix", "linearly independent rows"),
    ("运算放大器", "运放", "operational amplifier", "op amp"),
    ("相量", "phasor", "AC steady-state"),
    ("阻抗", "impedance", "AC circuit"),
)


FOUNDATION_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("数学", ("数学", "数学分析", "高等数学", "高数", "微积分", "线性代数", "概率", "统计", "几何",
           "集合", "函数", "导数", "积分", "极限", "矩阵", "方程", "向量", "张量", "证明", "定理")),
    ("物理", ("物理", "力学", "电路", "电磁", "热力学", "光学", "量子", "振动", "波", "场", "能量", "功", "动量",
           "电压", "电流", "电阻", "电荷", "电场")),
    ("化学", ("化学", "原子", "分子", "化学反应", "有机", "无机", "化学平衡")),
    ("生物", ("生物", "生理", "细胞", "遗传", "代谢", "进化", "生态")),
    ("计算机", ("数据结构", "算法", "离散", "复杂度", "数据库", "网络", "操作系统", "编程")),
    ("哲学", ("逻辑", "哲学", "认识论", "本体论", "伦理")),
)


def normalize_query(text: str) -> str:
    cleaned = SPACE_RE.sub(" ", str(text or "")).strip()
    cleaned = QUESTION_PREFIX.sub("", cleaned)
    return cleaned.strip(" ，,。")


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = normalize_query(value)
        key = clean.casefold()
        if len(clean) < 2 or key in seen:
            continue
        seen.add(key)
        result.append(clean)
    return result


def _matched_aliases(text: str) -> list[str]:
    lowered = text.casefold()
    matches: list[str] = []
    for group in ALIAS_GROUPS:
        if any(alias.casefold() in lowered for alias in group):
            matches.extend(group)
    if re.search(r"(?:节点|node).*(?:独立|independent).*(?:方程|equation)|(?:独立|independent).*(?:KCL|节点)", text, re.I):
        matches.extend(ALIAS_GROUPS[0])
        matches.extend(ALIAS_GROUPS[3])
    if re.search(r"(?:节点|node).{0,18}(?:电压|voltage)|(?:电压|voltage).{0,18}(?:节点|node)", text, re.I):
        matches.extend(ALIAS_GROUPS[2])
    return _dedupe(matches)


def _question_type(text: str) -> str:
    if re.search(r"区别|比较|异同|versus|\bvs\.?\b", text, re.I):
        return "compare"
    if re.search(r"为什么|为何|原理|机制|why|mechanism", text, re.I):
        return "mechanism"
    if re.search(r"怎么|如何|步骤|求解|calculate|how to", text, re.I):
        return "procedure"
    if re.search(r"是否|对不对|能否|判断|verify|whether", text, re.I):
        return "verification"
    return "fact_or_definition"


def _is_light_path(
    resolved: str,
    question_type: str,
    followup_resolved: bool,
    subquestions: list[str],
    aliases: list[str],
    constraints: list[str],
) -> bool:
    """Return True for short, direct questions where aggressive rewrite is likely noise."""
    if followup_resolved or subquestions:
        return False
    if question_type != "fact_or_definition" and question_type != "mechanism":
        return False
    if constraints:
        return False
    if len(resolved) > LIGHT_QUERY_MAX_LEN:
        return False
    if re.search(r"(?:并且|以及|分别|过程|步骤|公式|证明|应用|例子|演化|发展|起源|比较|区别|为何|为什么)", resolved, re.I):
        return False
    # Keep one-pass deterministic query for direct definition pattern.
    if re.search(r"^(?:什么是|何为|是什么意思|是什么)\s*[\u4e00-\u9fffA-Za-z]", resolved):
        return True
    if len(aliases) <= 1:
        return True
    return False


def _subquestions(text: str) -> list[str]:
    if len(text) < 28:
        return []
    parts = re.split(r"[；;]|(?:同时|另外|并且|以及|然后)", text)
    return [item for item in _dedupe(parts) if 6 <= len(item) < len(text)][:2]


def _detect_foundation_subject(text: str, concepts: list[str]) -> list[str]:
    lowered = text.casefold()
    fields: list[str] = []
    for field, signals in FOUNDATION_FIELDS:
        for signal in signals:
            if signal.casefold() in lowered:
                fields.append(field)
                break
    for concept in concepts:
        concept_lower = str(concept).strip().casefold()
        if len(concept_lower) <= 1:
            continue
        for field, signals in FOUNDATION_FIELDS:
            if any(signal.casefold() == concept_lower or concept_lower in signal.casefold() for signal in signals):
                fields.append(field)
                break
    return list(dict.fromkeys(fields))


def _critical_constraints(text: str) -> list[str]:
    """Return values an agent rewrite may not silently drop."""
    values = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?\s*(?:V|A|Ω|F|H)?|(?<![A-Za-z])[A-Z]{2,8}(?![A-Za-z])", text)
    return _dedupe(values)


def _preserves_constraints(candidate: str, constraints: Iterable[str]) -> bool:
    compact = re.sub(r"\s+", "", candidate).casefold()
    return all(re.sub(r"\s+", "", value).casefold() in compact for value in constraints)


def _has_explicit_latin_term(text: str) -> bool:
    return bool(re.search(r"(?<![A-Za-z])[A-Z]{2,8}(?![A-Za-z])|[A-Za-z]{4,}(?:[- ][A-Za-z]{3,})*", text))


def build_query_plan(
    question: str,
    *,
    resolved_question: str = "",
    concepts: Iterable[str] = (),
    suggested_queries: Iterable[str] = (),
    max_queries: int = 3,
) -> dict[str, Any]:
    original = normalize_query(question)
    resolved = normalize_query(resolved_question) or original
    concept_list = _dedupe(concepts)
    question_type = _question_type(resolved)
    constraints = _critical_constraints(resolved)
    if os.getenv("GARDEN_DISABLE_QUERY_REWRITE", "").strip().lower() in {"1", "true", "yes"}:
        return {
            "original": original, "resolved": resolved, "question_type": question_type,
            "concepts": concept_list, "aliases": [],
            "queries": [{"text": resolved, "source": "resolved", "weight": 1.0}],
            "subquestions": [], "constraints": constraints, "strategy": "single_query",
            "routing_reason": "查询改写已显式关闭", "method": "disabled",
        }

    aliases = _matched_aliases(" ".join([resolved, *concept_list]))
    subquestions = _subquestions(resolved)
    followup_resolved = original.casefold() != resolved.casefold()
    simple_exact = (
        len(resolved) <= 24 and _has_explicit_latin_term(resolved)
        and question_type == "fact_or_definition" and not subquestions
    )
    foundation_fields = _detect_foundation_subject(resolved, concept_list)
    is_foundation = bool(foundation_fields)
    bilingual_needed = bool(aliases and re.search(r"[\u4e00-\u9fff]", resolved) and not simple_exact)
    valid_agent_queries = [
        item for item in _dedupe(suggested_queries)
        if _preserves_constraints(item, constraints)
    ]

    if subquestions:
        strategy = "decompose"
        routing_reason = "问题包含多个可独立检索的目标"
    elif followup_resolved:
        strategy = "resolved_followup"
        routing_reason = "当前问题包含多轮指代，优先检索消解后的完整问题"
    elif _is_light_path(
        resolved=resolved,
        question_type=question_type,
        followup_resolved=followup_resolved,
        subquestions=subquestions,
        aliases=aliases,
        constraints=constraints,
    ):
        strategy = "single_query"
        routing_reason = "问题短小且直接，优先单路径检索减少上下文"
    elif bilingual_needed:
        strategy = "bilingual_expand"
        routing_reason = "中文问题命中可审计的中英专业术语组"
    elif is_foundation and question_type in {"fact_or_definition", "mechanism"} and not constraints:
        strategy = "single_query"
        routing_reason = "基础学科定义与关系问题保守走单路径，优先教材链式证据"
    elif valid_agent_queries and not simple_exact:
        strategy = "semantic_rewrite"
        routing_reason = "问题理解 Agent 提供了保留关键约束的非重复表达"
    else:
        strategy = "single_query"
        routing_reason = "问题简短明确，额外改写预计收益低"

    candidates: list[tuple[str, str, float]] = [(resolved, "resolved", 1.0)]
    # An unresolved pronoun query such as “它为什么” is noise, so do not add
    # the raw follow-up after a complete contextual resolution.
    if strategy == "semantic_rewrite":
        candidates.extend((item, "understanding_agent", 0.92) for item in valid_agent_queries[:1])
    if strategy == "bilingual_expand":
        # A foundational question can match more than one terminology group,
        # e.g. KCL + linear independence. Keep enough aliases to preserve both
        # the domain term and the mathematical relation in the expanded query.
        alias_query = " ".join(aliases[:8]).strip()
        if alias_query:
            candidates.append((f"{resolved} {alias_query}", "bilingual_alias", 0.88))
    if strategy == "decompose":
        candidates.extend((item, "decomposition", 0.78) for item in subquestions[:1])

    strategy_limits = {
        "resolved_followup": 1,
        "single_query": 1,
        "decompose": 1,
        "bilingual_expand": 2,
        "semantic_rewrite": 2,
    }
    query_limit = strategy_limits.get(strategy, max(1, max_queries))

    queries = []
    seen: set[str] = set()
    for text, source, weight in candidates:
        clean = normalize_query(text)
        key = clean.casefold()
        if not clean or key in seen:
            continue
        seen.add(key)
        queries.append({"text": clean, "source": source, "weight": weight})
        if len(queries) >= query_limit:
            break
    return {
        "original": original, "resolved": resolved, "question_type": question_type,
        "concepts": concept_list, "aliases": aliases, "queries": queries,
        "subquestions": subquestions, "constraints": constraints,
        "strategy": strategy, "routing_reason": routing_reason,
        "subject_mode": "foundational" if is_foundation else "general",
        "foundation_fields": foundation_fields,
        "method": "adaptive_structured_query",
    }
