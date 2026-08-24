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
    value = re.sub(r"<[^>]+>", "", unescape(str(text or "")))
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
    token = r"([A-Za-z][A-Za-z0-9_-]*)(?:\s*[\[({]+\s*[\"']?([^\]\)}\"']+)[\"']?\s*[\])}]+)?"
    edge_re = re.compile(token + r"\s*(?:-->|---|==>|-.->)(?:\|([^|]+)\|)?\s*" + token)
    for raw in code.splitlines():
        match = edge_re.search(raw)
        if not match:
            continue
        source, source_label, edge_label, target, target_label = match.groups()
        labels.setdefault(source, _clean_label(source_label or source))
        labels.setdefault(target, _clean_label(target_label or target))
        edges.append({"source": source, "target": target, "label": _clean_label(edge_label)[0]})
    nodes = [
        {"id": node_id, "label": value[0], "role": "concept", "evidence_ids": value[1]}
        for node_id, value in labels.items() if value[0]
    ]
    return nodes, edges


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
    allowed_source_ids: set[str], timeout: float = 14,
) -> dict[str, Any]:
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
