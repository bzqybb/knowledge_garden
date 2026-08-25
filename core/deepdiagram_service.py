from __future__ import annotations

import json
import os
import re
from html import unescape
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from core.config import llm_config
from core.deepdiagram_adapter import validate_diagram


class DeepDiagramServiceError(RuntimeError):
    pass


_AGENT_PREFIX = {
    "mindmap": "@mindmap",
    "flowchart": "@flow",
    "timeline": "@mermaid",
    "comparison": "@mermaid",
    "concept": "@drawio",
}


def _service_root() -> str:
    return os.getenv("DEEPDIAGRAM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def _validate_service_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise DeepDiagramServiceError("DeepDiagram 地址格式无效")
    loopback = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    if not loopback and os.getenv("DEEPDIAGRAM_ALLOW_REMOTE", "").lower() not in {"1", "true", "yes"}:
        raise DeepDiagramServiceError("为保护学习资料，默认只允许本机 DeepDiagram 服务")


def _request_json(url: str, *, timeout: float) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "KnowledgeGarden/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read(64_000).decode("utf-8"))


def full_service_available(timeout: float = 1.2) -> bool:
    root = _service_root()
    try:
        _validate_service_url(root)
        payload = _request_json(root + "/", timeout=timeout)
        return "deepdiagram" in str(payload.get("message", "")).lower()
    except Exception:
        return False


def _clean_label(text: Any) -> tuple[str, list[str]]:
    value = unescape(str(text or ""))
    value = re.sub(r"<br\s*/?>", " · ", value, flags=re.I)
    value = re.sub(r"<[^>]+>", "", value)
    evidence = re.findall(r"\[((?:M|L|W|A|T)\d+)\]", value)
    value = re.sub(r"\[((?:M|L|W|A|T)\d+)\]", "", value)
    return re.sub(r"\s+", " ", value).strip(" `#*-\t"), evidence


def _parse_mindmap(code: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    stack: list[tuple[int, str]] = []
    for raw in code.splitlines():
        if not raw.strip() or raw.lstrip().startswith("```"):
            continue
        heading = re.match(r"^\s*(#{1,6})\s+(.+)$", raw)
        bullet = re.match(r"^(\s*)[-*+]\s+(.+)$", raw)
        if heading:
            depth, text = len(heading.group(1)), heading.group(2)
        elif bullet:
            depth, text = 2 + len(bullet.group(1).replace("\t", "  ")) // 2, bullet.group(2)
        else:
            continue
        label, evidence = _clean_label(text)
        if not label:
            continue
        node_id = f"n{len(nodes) + 1}"
        nodes.append({"id": node_id, "label": label, "role": "anchor" if not nodes else "concept", "evidence_ids": evidence})
        while stack and stack[-1][0] >= depth:
            stack.pop()
        if stack:
            edges.append({"source": stack[-1][1], "target": node_id, "label": "包含"})
        stack.append((depth, node_id))
    return nodes, edges


def _parse_flow_json(code: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    match = re.search(r"\{[\s\S]*\}", code)
    payload = json.loads(match.group(0) if match else code)
    raw_nodes = payload.get("nodes") or payload.get("elements", {}).get("nodes") or []
    raw_edges = payload.get("edges") or payload.get("elements", {}).get("edges") or []
    nodes = []
    for item in raw_nodes:
        data = item.get("data") if isinstance(item.get("data"), dict) else {}
        label, evidence = _clean_label(data.get("label") or item.get("label") or item.get("text"))
        if label:
            nodes.append({"id": str(item.get("id", "")), "label": label, "role": "concept", "evidence_ids": evidence})
    edges = [
        {"source": str(item.get("source", "")), "target": str(item.get("target", "")), "label": _clean_label(item.get("label") or (item.get("data") or {}).get("label"))[0]}
        for item in raw_edges
    ]
    return nodes, edges


def _parse_mermaid(code: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    labels: dict[str, tuple[str, list[str]]] = {}
    edges: list[dict[str, str]] = []
    # DeepDiagram usually declares nodes on their own lines and connects them
    # later (e.g. `A["label"]` followed by `ROOT --> A`).  The old parser only
    # understood inline declarations on edge lines, reducing real diagrams to
    # opaque IDs such as A/B/CQ.  Capture standalone declarations first.
    quoted_node_re = re.compile(
        r"([A-Za-z][A-Za-z0-9_-]*)\s*(?:\[\[|\[\(|\[|\(\(|\(|\{\{|\{)\s*"
        r"[\"'](.*?)[\"']\s*(?:\]\]|\]\)|\]|\)\)|\)|\}\}|\})"
    )
    node_re = re.compile(
        r"^\s*([A-Za-z][A-Za-z0-9_-]*)\s*"
        r"(?:\[\[|\[\(|\[|\(\(|\(|\{\{|\{)\s*"
        r"(.*?)\s*"
        r"(?:\]\]|\]\)|\]|\)\)|\)|\}\}|\})\s*$"
    )
    for raw in code.splitlines():
        line = raw.strip()
        if not line or line.startswith(("flowchart ", "graph ", "subgraph ", "class", "style ", "end", "</")):
            continue
        quoted = list(quoted_node_re.finditer(raw))
        for match in quoted:
            node_id, node_label = match.groups()
            labels[node_id] = _clean_label(node_label)
        if not quoted:
            match = node_re.match(raw)
            if match:
                node_id, node_label = match.groups()
                labels[node_id] = _clean_label(node_label.strip(" \"'"))

    edge_re = re.compile(
        r"\b([A-Za-z][A-Za-z0-9_-]*)\b\s*"
        r"(?:-->|---|==>|-\.->)(?:\|([^|]+)\|)?\s*"
        r"\b([A-Za-z][A-Za-z0-9_-]*)\b"
    )
    for raw in code.splitlines():
        # Replace inline node declarations with their IDs before reading the
        # relation. This preserves nested evidence markers such as [W1] inside
        # quoted labels instead of mistaking their closing bracket for a shape.
        relation_line = quoted_node_re.sub(lambda match: match.group(1), raw)
        relation_line = re.sub(
            r"([A-Za-z][A-Za-z0-9_-]*)\s*\[([^\[\]]+)\]",
            lambda match: match.group(1), relation_line,
        )
        match = edge_re.search(relation_line)
        if not match:
            continue
        source, edge_label, target = match.groups()
        labels.setdefault(source, _clean_label(source))
        labels.setdefault(target, _clean_label(target))
        edges.append({"source": source, "target": target, "label": _clean_label(edge_label)[0]})
    nodes = [
        {"id": node_id, "label": value[0], "role": "concept", "evidence_ids": value[1]}
        for node_id, value in labels.items() if value[0]
    ]
    return nodes, edges


def _structure_comparison_graph(
    nodes: list[dict[str, Any]], edges: list[dict[str, str]], blueprint: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Add explicit comparison anchors to a full-service Mermaid result.

    Mermaid subgraph headings are visual containers rather than graph nodes, so
    they disappear when converted into Knowledge Garden's safe SVG format.  We
    reconstruct the two audited subjects and bind evidence-bearing claims to
    the matching subject without asking another model or adding facts.
    """
    subjects = [str(item).strip() for item in blueprint.get("comparison_subjects", []) if str(item).strip()][:2]
    if len(subjects) < 2:
        return nodes, edges

    normalized_question = re.sub(r"\s+", "", str(blueprint.get("core_question") or "")).lower()
    subject_normalized = [re.sub(r"\s+", "", item).lower() for item in subjects]
    filtered: list[dict[str, Any]] = []
    for item in nodes:
        label = str(item.get("label") or "").strip()
        normalized = re.sub(r"\s+", "", label).lower()
        is_unbound_comparison_heading = (
            not item.get("evidence_ids")
            and normalized not in subject_normalized
            and all(subject in normalized for subject in subject_normalized)
        )
        if "corequestion" in normalized or (normalized_question and normalized == normalized_question) or is_unbound_comparison_heading:
            continue
        item = dict(item)
        item["role"] = "boundary" if re.match(r"^(?:gap|boundary|边界|缺口)\s*[:：]", label, re.I) else "comparison_dimension"
        filtered.append(item)

    existing_ids = {str(item.get("id")) for item in filtered}
    subject_nodes: list[dict[str, Any]] = []
    for index, subject in enumerate(subjects, 1):
        existing = next((
            item for item in filtered
            if re.sub(r"\s+", "", str(item.get("label") or "")).lower() == subject_normalized[index - 1]
        ), None)
        if existing:
            filtered.remove(existing)
            existing = dict(existing)
            existing["role"] = "anchor"
            subject_nodes.append(existing)
        else:
            node_id = f"comparison_subject_{index}"
            while node_id in existing_ids:
                node_id += "_"
            existing_ids.add(node_id)
            subject_nodes.append({"id": node_id, "label": subject, "role": "anchor", "evidence_ids": []})

    evidence_subjects: dict[str, set[int]] = {}
    for item in blueprint.get("evidence_items", []):
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id") or "")
        excerpt = re.sub(r"\s+", "", str(item.get("excerpt") or "")).lower()
        for index, subject in enumerate(subjects):
            if re.sub(r"\s+", "", subject).lower() in excerpt:
                evidence_subjects.setdefault(source_id, set()).add(index)

    clean_ids = {str(item.get("id")) for item in filtered}
    clean_edges = [
        dict(item) for item in edges
        if str(item.get("source")) in clean_ids and str(item.get("target")) in clean_ids
    ]
    for item in filtered:
        if item.get("role") != "comparison_dimension":
            continue
        matched: set[int] = set()
        for source_id in item.get("evidence_ids", []):
            matched.update(evidence_subjects.get(str(source_id), set()))
        targets = sorted(matched) if matched else list(range(len(subject_nodes)))
        for index in targets:
            source_id, target_id = subject_nodes[index]["id"], str(item.get("id"))
            existing_edge = next((
                edge for edge in clean_edges
                if str(edge.get("source")) == source_id and str(edge.get("target")) == target_id
            ), None)
            if existing_edge:
                existing_edge["label"] = "比较维度"
            else:
                clean_edges.append({"source": source_id, "target": target_id, "label": "比较维度"})
            subject_nodes[index]["evidence_ids"] = sorted(set(
                subject_nodes[index]["evidence_ids"] + [str(value) for value in item.get("evidence_ids", [])]
            ))
    return subject_nodes + filtered, clean_edges


def _parse_drawio(code: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    start = code.find("<mxfile")
    if start < 0:
        start = code.find("<mxGraphModel")
    if start < 0:
        raise ValueError("没有 mxGraph XML")
    root = ElementTree.fromstring(code[start:].strip())
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    for cell in root.iter("mxCell"):
        if cell.get("vertex") == "1":
            label, evidence = _clean_label(cell.get("value"))
            if label:
                nodes.append({"id": str(cell.get("id", "")), "label": label, "role": "concept", "evidence_ids": evidence})
        elif cell.get("edge") == "1":
            edges.append({
                "source": str(cell.get("source", "")), "target": str(cell.get("target", "")),
                "label": _clean_label(cell.get("value"))[0],
            })
    return nodes, edges


def _parse_code(code: str, agent: str, kind: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if agent == "mindmap" or kind == "mindmap":
        return _parse_mindmap(code)
    if agent == "flow" or kind == "flowchart":
        return _parse_flow_json(code)
    if agent == "drawio" or kind == "concept":
        return _parse_drawio(code)
    return _parse_mermaid(code)


def generate_with_full_service(
    *, user_request: str, kind: str, blueprint: dict[str, Any],
    allowed_source_ids: set[str], timeout: float | None = None,
) -> dict[str, Any]:
    timeout = timeout or float(os.getenv("DEEPDIAGRAM_TIMEOUT_SECONDS", "45"))
    root = _service_root()
    _validate_service_url(root)
    if not full_service_available():
        raise DeepDiagramServiceError("完整 DeepDiagram 服务未启动")
    config = llm_config()
    prefix = _AGENT_PREFIX.get(kind, "@mindmap")
    prompt = (
        f"{prefix} {user_request}\n\n"
        "以下蓝图已经过知识花园的来源审查。只能重组其中的知识，不得添加新事实；"
        "事实节点末尾保留对应来源ID，例如 [W1]。输出小而清晰的图。\n"
        f"AUDITED_BLUEPRINT={json.dumps(blueprint, ensure_ascii=False)[:12000]}"
    )
    body: dict[str, Any] = {"prompt": prompt, "history": [], "context": {"source": "knowledge-garden"}}
    # The official local API accepts per-request OpenAI-compatible model config.
    # Credentials are forwarded only to a loopback service and never persisted here.
    if urlparse(root).hostname in {"127.0.0.1", "localhost", "::1"} and config.enabled:
        body.update({"model_id": config.model, "api_key": config.api_key, "base_url": config.base_url})
    request = Request(
        root + "/api/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream", "User-Agent": "KnowledgeGarden/1.0"},
        method="POST",
    )
    selected_agent = ""
    design_parts: list[str] = []
    code = ""
    event = ""
    with urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                try:
                    data = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                if event == "agent_selected":
                    selected_agent = str(data.get("agent") or "")
                elif event == "design_concept":
                    design_parts.append(str(data.get("content") or ""))
                elif event == "tool_end":
                    code = str(data.get("output") or "")
                elif event == "error":
                    raise DeepDiagramServiceError(str(data.get("message") or "生成失败"))
            if len(code) > 500_000:
                raise DeepDiagramServiceError("DeepDiagram 输出过大")
    if not code:
        raise DeepDiagramServiceError("完整 DeepDiagram 没有返回图形代码")
    try:
        nodes, edges = _parse_code(code, selected_agent, kind)
    except Exception as exc:
        raise DeepDiagramServiceError("完整 DeepDiagram 产物无法安全解析") from exc
    if kind == "comparison":
        nodes, edges = _structure_comparison_graph(nodes, edges, blueprint)
    return validate_diagram(
        {
            "title": str(blueprint.get("research_object") or blueprint.get("core_question") or "知识图解"),
            "design_concept": "".join(design_parts)[:180],
            "nodes": nodes,
            "edges": edges,
        },
        requested_kind=kind,
        allowed_source_ids=allowed_source_ids,
        provider="deepdiagram-full",
    )
