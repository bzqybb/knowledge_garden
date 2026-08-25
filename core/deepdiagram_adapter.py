from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field


class DiagramNode(BaseModel):
    id: str
    label: str
    role: Literal[
        "anchor", "concept", "comparison_dimension", "step", "boundary", "evidence", "unknown"
    ] = "concept"
    evidence_ids: list[str] = Field(default_factory=list)


class DiagramEdge(BaseModel):
    source: str
    target: str
    label: str = ""


class DiagramSpec(BaseModel):
    status: Literal["ready", "unavailable", "suppressed"] = "unavailable"
    provider: str = "deepdiagram-compatible"
    kind: Literal["none", "mindmap", "flowchart", "timeline", "comparison", "concept"] = "none"
    title: str = ""
    design_concept: str = ""
    nodes: list[DiagramNode] = Field(default_factory=list)
    edges: list[DiagramEdge] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    warning: str = ""


_UNSAFE_LABEL = re.compile(r"<\s*(?:script|iframe|object)|javascript:|data:text/html", re.I)


def unavailable_diagram(kind: str, reason: str) -> dict[str, Any]:
    safe_kind = kind if kind in {"mindmap", "flowchart", "timeline", "comparison", "concept"} else "none"
    return DiagramSpec(
        status="unavailable",
        kind=safe_kind,
        warning=reason,
    ).model_dump()


def validate_diagram(
    payload: dict[str, Any] | None,
    *,
    requested_kind: str,
    allowed_source_ids: set[str],
    provider: str = "deepdiagram-compatible",
) -> dict[str, Any]:
    """Validate the model-produced DeepDiagram-compatible intermediate format.

    The browser never receives arbitrary HTML, JavaScript, Mermaid directives or
    executable diagram code. It renders this small node/edge representation as
    SVG. Evidence identifiers are also intersected with the audited source set.
    """
    if not isinstance(payload, dict):
        return unavailable_diagram(requested_kind, "图解生成器没有返回可解析的结构。")
    candidate = dict(payload)
    candidate["provider"] = provider
    candidate["kind"] = requested_kind if requested_kind in {
        "mindmap", "flowchart", "timeline", "comparison", "concept"
    } else "none"
    candidate["status"] = "ready"
    try:
        spec = DiagramSpec.model_validate(candidate)
    except Exception:
        return unavailable_diagram(requested_kind, "图解结构未通过类型校验。")

    clean_nodes: list[DiagramNode] = []
    seen: set[str] = set()
    for raw in spec.nodes[:18]:
        node_id = re.sub(r"[^A-Za-z0-9_-]", "", raw.id)[:36]
        label = re.sub(r"\s+", " ", raw.label).strip()[:80]
        if not node_id or node_id in seen or not label or _UNSAFE_LABEL.search(label):
            continue
        seen.add(node_id)
        clean_nodes.append(DiagramNode(
            id=node_id,
            label=label,
            role=raw.role,
            evidence_ids=[item for item in raw.evidence_ids if item in allowed_source_ids][:4],
        ))
    if len(clean_nodes) < 2:
        return unavailable_diagram(requested_kind, "可核查的图解节点不足，已回退为纯文字回答。")

    node_ids = {item.id for item in clean_nodes}
    clean_edges: list[DiagramEdge] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for raw in spec.edges[:28]:
        label = re.sub(r"\s+", " ", raw.label).strip()[:32]
        key = (raw.source, raw.target, label)
        if raw.source not in node_ids or raw.target not in node_ids or raw.source == raw.target or key in seen_edges:
            continue
        seen_edges.add(key)
        clean_edges.append(DiagramEdge(source=raw.source, target=raw.target, label=label))
    if not clean_edges:
        return unavailable_diagram(requested_kind, "图解节点之间没有形成有效关系，已回退为纯文字回答。")

    used_sources = sorted({source for node in clean_nodes for source in node.evidence_ids})
    return DiagramSpec(
        status="ready",
        provider=provider,
        kind=spec.kind,
        title=re.sub(r"\s+", " ", spec.title).strip()[:80] or "知识图解",
        design_concept=re.sub(r"\s+", " ", spec.design_concept).strip()[:180],
        nodes=clean_nodes,
        edges=clean_edges,
        source_ids=used_sources,
        warning=("图中未标注来源的节点只表达回答结构，不作为新的事实依据。" if not used_sources else ""),
    ).model_dump()


def _clean_knowledge_text(value: Any) -> str:
    """Turn note prose into a diagram label without leaking vault markup.

    Local fallback diagrams must never expose Obsidian navigation, Markdown
    headings, source boilerplate, or file paths as if they were concepts.
    """
    text = str(value or "")
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"(?m)^\s*(?:#{1,6}|>|[-*+]\s+)\s*", "", text)
    text = re.sub(r"[*_`~]", "", text)
    text = re.sub(r"\s*[|>]\s*", " · ", text)
    text = re.sub(r"\s+", " ", text).strip(" ·：:；;，,。")
    # These are provenance/navigation strings, not learnable knowledge nodes.
    if re.search(r"(?:^|\s)(?:来源|原始资料|文件路径|降维对照|主题索引)\s*[:：]", text):
        return ""
    if re.search(r"(?:^|[\\/])wiki[\\/]|\.md(?:\s|$)|第\s*\d+\s*页", text, re.I):
        return ""
    return text


def _short_label(value: Any, limit: int = 28) -> str:
    text = _clean_knowledge_text(value)
    text = re.sub(r"^(?:为什么|如何|怎么|请|能否|是否)", "", text).strip("？?。；;：: ")
    for separator in ("；", ";", "。", "，", ","):
        if separator in text and len(text) > limit:
            text = text.split(separator, 1)[0].strip()
    return text[:limit].strip()


def _comparison_subjects(blueprint: dict[str, Any]) -> list[str]:
    declared = [
        _short_label(item, 22)
        for item in blueprint.get("comparison_subjects", [])
        if _short_label(item, 22)
    ]
    if len(declared) >= 2:
        return declared[:2]
    value = str(blueprint.get("research_object") or blueprint.get("core_question") or "")
    patterns = (
        r"(.+?)\s*(?:与|和|跟|vs\.?|VS\.?)\s*(.+?)(?:的(?:核心)?(?:区别|差异|关系)|有什么(?:区别|差异)|相比|$)",
        r"比较\s*(.+?)\s*(?:与|和|跟)\s*(.+)",
    )
    for pattern in patterns:
        match = re.search(pattern, value, re.I)
        if match:
            subjects = [_short_label(match.group(1), 22), _short_label(match.group(2), 22)]
            if all(subjects):
                return subjects
    return declared


def _dimension_for_claim(claim: str) -> str:
    rules = (
        ("核心关注", r"研究|关注|对象|核心|问题"),
        ("目标与价值", r"目标|目的|价值|为了|旨在"),
        ("方法与判断", r"方法|实验|分析|评估|判断|测量"),
        ("应用场景", r"应用|设计|实践|场景|系统|产品"),
        ("边界与风险", r"边界|风险|伦理|责任|规范|限制|条件"),
    )
    for label, pattern in rules:
        if re.search(pattern, claim):
            return label
    return "关键差异"


def build_local_diagram(
    blueprint: dict[str, Any],
    *,
    requested_kind: str,
    allowed_source_ids: set[str],
    fallback_reason: str,
) -> dict[str, Any]:
    """Build a deterministic, evidence-bounded diagram when full DeepDiagram fails.

    This adapter performs no LLM call. It is intentionally less decorative than
    DeepDiagram, but it always returns a usable graph from the audited blueprint.
    """
    kind = requested_kind if requested_kind in {
        "mindmap", "flowchart", "timeline", "comparison", "concept"
    } else "mindmap"
    anchor = _short_label(blueprint.get("research_object") or blueprint.get("core_question"), 32) or "本轮知识"
    direct_ids = [
        str(item) for item in blueprint.get("direct_source_ids", [])
        if str(item) in allowed_source_ids
    ][:4]
    evidence_items = [item for item in blueprint.get("evidence_items", []) if isinstance(item, dict)]
    claims_with_sources = [
        (_short_label(item.get("excerpt"), 38), [str(item.get("source_id"))])
        for item in evidence_items
        if str(item.get("source_id")) in allowed_source_ids
    ]
    if not claims_with_sources:
        claims_with_sources = [(_short_label(item, 38), direct_ids) for item in blueprint.get("usable_claims", [])]
    claims_with_sources = [(label, sources) for label, sources in claims_with_sources if label][:6]
    claims = [item[0] for item in claims_with_sources]
    if not claims:
        claims = [_short_label(item) for item in blueprint.get("explanation_order", [])]
        claims = [item for item in claims if item][:5]
        claims_with_sources = [(item, []) for item in claims]
    if not claims:
        return unavailable_diagram(kind, fallback_reason + "；蓝图中没有可呈现的审计内容。")

    comparison_subjects = _comparison_subjects(blueprint) if kind == "comparison" else []
    if kind == "comparison" and len(comparison_subjects) >= 2:
        nodes = [
            {"id": "left", "label": comparison_subjects[0], "role": "anchor", "evidence_ids": []},
            {"id": "right", "label": comparison_subjects[1], "role": "anchor", "evidence_ids": []},
        ]
        for index, (claim, claim_sources) in enumerate(claims_with_sources, 1):
            nodes.append({
                "id": f"n{index}", "label": claim, "role": "comparison_dimension",
                "evidence_ids": [item for item in claim_sources if item in allowed_source_ids],
            })
    else:
        nodes = [{"id": "anchor", "label": anchor, "role": "anchor", "evidence_ids": []}]
        for index, (claim, claim_sources) in enumerate(claims_with_sources, 1):
            nodes.append({
                "id": f"n{index}", "label": claim, "role": "concept",
                "evidence_ids": [item for item in claim_sources if item in allowed_source_ids],
            })
    gaps = [_short_label(item) for item in blueprint.get("gaps", [])]
    gaps = [item for item in gaps if item][:1]
    if gaps:
        nodes.append({"id": "boundary", "label": gaps[0], "role": "boundary", "evidence_ids": []})

    edges: list[dict[str, str]] = []
    if kind in {"flowchart", "timeline"}:
        previous = "anchor"
        for index in range(1, len(claims) + 1):
            current = f"n{index}"
            edges.append({"source": previous, "target": current, "label": "下一步" if kind == "flowchart" else "演进"})
            previous = current
        if gaps:
            edges.append({"source": previous, "target": "boundary", "label": "边界"})
    elif kind == "comparison" and len(comparison_subjects) >= 2:
        for index, (claim, _) in enumerate(claims_with_sources, 1):
            dimension = _dimension_for_claim(claim)
            normalized_claim = re.sub(r"\s+", "", claim).lower()
            left_hit = re.sub(r"\s+", "", comparison_subjects[0]).lower() in normalized_claim
            right_hit = re.sub(r"\s+", "", comparison_subjects[1]).lower() in normalized_claim
            # A claim that does not name either side is presented as an audited
            # comparison dimension, not falsely assigned to one concept.
            if left_hit or not right_hit:
                edges.append({"source": "left", "target": f"n{index}", "label": dimension})
            if right_hit or not left_hit:
                edges.append({"source": "right", "target": f"n{index}", "label": dimension})
        if gaps:
            edges.extend([
                {"source": "left", "target": "boundary", "label": "证据边界"},
                {"source": "right", "target": "boundary", "label": "证据边界"},
            ])
    else:
        relation = "空间关系" if kind == "concept" else "包含"
        edges.extend({"source": "anchor", "target": f"n{index}", "label": relation} for index in range(1, len(claims) + 1))
        if gaps:
            edges.append({"source": "anchor", "target": "boundary", "label": "适用边界"})

    result = validate_diagram(
        {
            "title": (
                f"{comparison_subjects[0]}与{comparison_subjects[1]} · 比较图解"
                if kind == "comparison" and len(comparison_subjects) >= 2 else
                f"{anchor} · 知识图解"
            ),
            "design_concept": "完整 DeepDiagram 不可用，已由本地确定性适配器从审计蓝图生成。",
            "nodes": nodes,
            "edges": edges,
        },
        requested_kind=kind,
        allowed_source_ids=allowed_source_ids,
        provider="local-deterministic-adapter",
    )
    if result.get("status") == "ready":
        result["warning"] = fallback_reason
    return result


def diagram_is_grounded(spec: dict[str, Any], allowed_source_ids: set[str]) -> bool:
    if spec.get("status") != "ready":
        return True
    for node in spec.get("nodes", []):
        if any(source not in allowed_source_ids for source in node.get("evidence_ids", [])):
            return False
    node_ids = {str(item.get("id")) for item in spec.get("nodes", [])}
    return bool(node_ids) and all(
        edge.get("source") in node_ids and edge.get("target") in node_ids
        for edge in spec.get("edges", [])
    )


def diagram_has_teaching_value(spec: dict[str, Any], requested_kind: str) -> bool:
    """Semantic hard gate used by the Reflector, beyond schema validity."""
    if spec.get("status") != "ready":
        return requested_kind in {"none", ""}
    nodes = [item for item in spec.get("nodes", []) if isinstance(item, dict)]
    labels = [str(item.get("label") or "") for item in nodes]
    dirty = re.compile(r"(?:^\s*#|\[\[|\.md(?:\s|$)|降维对照\s*>|(?:来源|原始资料)\s*[:：]|\|)", re.I)
    if any(not label.strip() or dirty.search(label) for label in labels):
        return False
    if requested_kind == "comparison":
        anchors = [item for item in nodes if item.get("role") == "anchor"]
        dimensions = [item for item in nodes if item.get("role") == "comparison_dimension"]
        edge_labels = {str(item.get("label") or "") for item in spec.get("edges", [])}
        return len(anchors) >= 2 and len(dimensions) >= 2 and len(edge_labels - {"", "对照"}) >= 1
    return len(nodes) >= 2
