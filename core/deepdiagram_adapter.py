from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field


class DiagramNode(BaseModel):
    id: str
    label: str
    role: Literal["anchor", "concept", "step", "boundary", "evidence", "unknown"] = "concept"
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


def _short_label(value: Any, limit: int = 28) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"^(?:为什么|如何|怎么|请|能否|是否)", "", text).strip("？?。；;：: ")
    for separator in ("；", ";", "。", "，", ","):
        if separator in text and len(text) > limit:
            text = text.split(separator, 1)[0].strip()
    return text[:limit].strip()


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
        (_short_label(item.get("excerpt")), [str(item.get("source_id"))])
        for item in evidence_items
        if str(item.get("source_id")) in allowed_source_ids
    ]
    if not claims_with_sources:
        claims_with_sources = [(_short_label(item), direct_ids) for item in blueprint.get("usable_claims", [])]
    claims_with_sources = [(label, sources) for label, sources in claims_with_sources if label][:6]
    claims = [item[0] for item in claims_with_sources]
    if not claims:
        claims = [_short_label(item) for item in blueprint.get("explanation_order", [])]
        claims = [item for item in claims if item][:5]
        claims_with_sources = [(item, []) for item in claims]
    if not claims:
        return unavailable_diagram(kind, fallback_reason + "；蓝图中没有可呈现的审计内容。")

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
    else:
        relation = "对照" if kind == "comparison" else "空间关系" if kind == "concept" else "包含"
        edges.extend({"source": "anchor", "target": f"n{index}", "label": relation} for index in range(1, len(claims) + 1))
        if gaps:
            edges.append({"source": "anchor", "target": "boundary", "label": "适用边界"})

    result = validate_diagram(
        {
            "title": f"{anchor} · 知识图解",
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
