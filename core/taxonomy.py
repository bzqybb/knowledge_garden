from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from typing import Any

from core.llm import LLMError, chat_json
from core.storage import GardenStore


ALIASES = {
    "attention": "注意力机制", "snn": "脉冲神经网络",
    "surrogategradient": "代理梯度", "backpropagation": "反向传播",
    "heaviside": "阶跃函数",
}


def _key(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    value = re.sub(r"\s*[（(][^（）()]{1,80}[）)]\s*$", "", value)
    compact = re.sub(r"[\s·•_—–\-:：]+", "", value)
    return ALIASES.get(compact, compact)


def _fallback_relations(topic: str, concepts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Conservative offline fallback: add one level only with visible naming evidence."""
    if len(concepts) < 3:
        return []
    topic_surface = re.sub(r"\s+", "", unicodedata.normalize("NFKC", topic).lower())
    scored = []
    for item in concepts:
        title = item["title"]
        aliases = [part for part in re.split(r"[（）()]", unicodedata.normalize("NFKC", title).lower()) if part]
        direct = any(len(part.strip()) >= 3 and re.sub(r"\s+", "", part) in topic_surface for part in aliases)
        broad = bool(re.search(r"(机制|网络|模型|理论|系统|框架)$", re.sub(r"[（(].*$", "", title)))
        score = (10 if direct else 0) + (3 if broad else 0)
        if score:
            scored.append((score, -len(title), title, item))
    if not scored:
        return []
    parent = max(scored)[3]
    return [
        {
            "parent_id": parent["id"], "child_id": child["id"], "child_title": child["title"],
            "confidence": 0.66, "reason": "离线保守规则：主题名称或宽泛概念名提供了父子证据",
        }
        for child in concepts if child["id"] != parent["id"]
    ]


def _acyclic(parent_of: dict[int, int], parent: int, child: int) -> bool:
    cursor = parent
    seen = {child}
    while cursor in parent_of:
        if cursor in seen:
            return False
        seen.add(cursor)
        cursor = parent_of[cursor]
    return cursor not in seen


def rebuild_concept_hierarchy(store: GardenStore, force: bool = False) -> dict[str, Any]:
    """Classify every MOC's concepts and persist a reusable, model-assisted tree."""
    with store.connect() as conn:
        notes = {
            row["id"]: dict(row) for row in conn.execute(
                "SELECT id,title,kind,content FROM notes WHERE kind IN ('moc','concept') ORDER BY title"
            )
        }
        edges = [dict(row) for row in conn.execute(
            "SELECT source_id,target_id FROM links WHERE relation='wikilink' AND target_id IS NOT NULL"
        )]
    if not notes:
        return {"changed": False, "relations": 0, "method": "empty", "topics": 0}

    concept_groups: dict[str, list[int]] = defaultdict(list)
    for note_id, note in notes.items():
        if note["kind"] == "concept":
            concept_groups[_key(note["title"])].append(note_id)
    representative: dict[int, int] = {}
    for group in concept_groups.values():
        chosen = max(group, key=lambda note_id: len(notes[note_id]["content"]))
        representative.update({note_id: chosen for note_id in group})

    topics: dict[int, set[int]] = defaultdict(set)
    for edge in edges:
        left, right = edge["source_id"], edge["target_id"]
        if left in notes and right in notes:
            if notes[left]["kind"] == "moc" and notes[right]["kind"] == "concept":
                topics[left].add(representative.get(right, right))
            elif notes[right]["kind"] == "moc" and notes[left]["kind"] == "concept":
                topics[right].add(representative.get(left, left))

    signature_payload = [
        (
            notes[moc_id]["title"],
            hashlib.sha256(notes[moc_id]["content"].encode("utf-8")).hexdigest(),
            sorted(
                (
                    _key(notes[item]["title"]),
                    hashlib.sha256(notes[item]["content"].encode("utf-8")).hexdigest(),
                )
                for item in concept_ids
            ),
        )
        for moc_id, concept_ids in sorted(topics.items())
    ]
    signature = hashlib.sha256(json.dumps(signature_payload, ensure_ascii=False).encode("utf-8")).hexdigest()
    if not force and store.setting("mindmap_hierarchy_signature", "") == signature:
        return {"changed": False, "relations": 0, "method": "cached", "topics": len(topics)}

    all_relations: list[dict[str, Any]] = []
    methods = set()
    globally_parented: set[int] = set()
    for moc_id, concept_ids in topics.items():
        concepts = [
            {"id": item, "title": notes[item]["title"], "excerpt": re.sub(r"\s+", " ", notes[item]["content"])[:360]}
            for item in sorted(concept_ids, key=lambda item: notes[item]["title"])
        ]
        if len(concepts) < 2:
            continue
        result = None
        try:
            result = chat_json(
                "你是知识图谱分类导师。根据概念角色建立有教学意义的父子层级，而不是把所有相关词都并列或强行串成链。只使用给出的原始标题。父节点必须比子节点更基础、更宽泛，或是确实包含该组成/方法的对象；互为补充、对比或仅仅相关的概念保持同级。证据不足就放 unresolved。每个子节点最多一个父节点，禁止环。",
                "主题：" + notes[moc_id]["title"] + "\n概念：\n" +
                "\n".join(f"- {item['title']}：{item['excerpt']}" for item in concepts) +
                "\n返回 JSON：{\"relations\":[{\"parent\":\"原始标题\",\"child\":\"原始标题\",\"reason\":\"层级依据\",\"confidence\":0到1}],\"unresolved\":[\"原始标题\"]}。仅保留 confidence>=0.65 的关系。",
            )
        except LLMError:
            result = None

        by_key = {_key(item["title"]): item for item in concepts}
        candidates = []
        if isinstance(result, dict) and isinstance(result.get("relations"), list):
            for relation in result["relations"]:
                if not isinstance(relation, dict):
                    continue
                parent = by_key.get(_key(str(relation.get("parent", ""))))
                child = by_key.get(_key(str(relation.get("child", ""))))
                try:
                    confidence = float(relation.get("confidence", 0))
                except (TypeError, ValueError):
                    confidence = 0
                if parent and child and parent["id"] != child["id"] and confidence >= 0.65:
                    candidates.append({
                        "parent_id": parent["id"], "child_id": child["id"], "child_title": child["title"],
                        "confidence": min(1.0, confidence), "reason": str(relation.get("reason") or "模型判断的概念层级"),
                    })
            if candidates:
                methods.add("langchain")
        if not candidates:
            candidates = _fallback_relations(notes[moc_id]["title"], concepts)
            if candidates:
                methods.add("offline")

        parent_of: dict[int, int] = {}
        for relation in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
            child = relation["child_id"]
            if child in globally_parented or child in parent_of:
                continue
            if not _acyclic(parent_of, relation["parent_id"], child):
                continue
            parent_of[child] = relation["parent_id"]
            globally_parented.add(child)
            all_relations.append(relation)

    store.replace_agent_taxonomy_links(all_relations)
    store.set_setting("mindmap_hierarchy_signature", signature)
    if all_relations:
        store.add_activity("agent_taxonomy", f"整理 {len(topics)} 个主题、{len(all_relations)} 条父子关系", 5)
    return {
        "changed": True, "relations": len(all_relations),
        "method": "+".join(sorted(methods)) or "unresolved", "topics": len(topics),
    }


def classify_unmounted_concepts(store: GardenStore, limit: int = 24) -> dict[str, Any]:
    """Give future concept notes an explicit classification lifecycle.

    Only an exact existing MOC may be selected automatically. A quoted body
    excerpt and confidence >= .70 are required; otherwise the note remains in a
    visible review queue instead of being assigned by keyword coincidence.
    """
    with store.connect() as conn:
        concepts = [dict(row) for row in conn.execute(
            "SELECT id,title,content,updated_at FROM notes WHERE kind='concept' ORDER BY updated_at DESC"
        )]
        mocs = [dict(row) for row in conn.execute(
            "SELECT id,title,content FROM notes WHERE kind='moc' ORDER BY title"
        )]
        mounted = {
            int(row["concept_id"]) for row in conn.execute(
                """SELECT CASE WHEN s.kind='concept' THEN s.id ELSE t.id END concept_id
                   FROM links l JOIN notes s ON s.id=l.source_id JOIN notes t ON t.id=l.target_id
                   WHERE l.status!='rejected' AND (
                     (s.kind='moc' AND t.kind='concept') OR (s.kind='concept' AND t.kind='moc')
                   ) AND l.relation IN ('wikilink','contains')"""
            )
        }
    pending = [item for item in concepts if item["id"] not in mounted][:limit]
    previous = store.setting("classification_queue_v1", {}) or {}
    queue = dict(previous) if isinstance(previous, dict) else {}
    if not pending:
        store.set_setting("classification_queue_v1", {})
        return {"examined": 0, "classified": 0, "needs_review": 0, "items": []}

    moc_by_key = {_key(item["title"]): item for item in mocs}
    classified = 0
    for note in pending:
        content = re.sub(r"\s+", " ", str(note.get("content") or "")).strip()
        status = "needs_review"
        detail: dict[str, Any] = {
            "note_id": note["id"], "title": note["title"], "status": status,
            "target_moc": "", "confidence": 0.0, "evidence": "", "reason": "",
        }
        if len(content) < 80 or re.search(r"等待后续资料|等待继续充实|自动建立.*占位|此页由.*自动建立", content):
            detail["reason"] = "正文不足或仍是占位页，不能进行有依据的自动分类。"
            queue[str(note["id"])] = detail
            continue
        if not mocs:
            detail["reason"] = "知识库中尚无可作为父节点的主题 MOC。"
            queue[str(note["id"])] = detail
            continue

        result = None
        try:
            result = chat_json(
                "你是待分类新知 Agent。只能把概念挂到给定的一个现有主题 MOC，不能新造分类名。判断必须理解正文主张、研究对象和机制，不能只凭标题关键词。evidence 必须逐字引用正文中能证明领域归属的一小句；若没有充分证据，target_moc 留空、status=needs_review。confidence<0.70 不得自动挂载。",
                f"待分类概念：{note['title']}\n正文：{content[:2200]}\n"
                "可选主题MOC：\n" + "\n".join(f"- {item['title']}" for item in mocs) +
                "\n输出 status(classified/needs_review)、target_moc、confidence、evidence、reason。",
            )
        except LLMError:
            result = None
        if isinstance(result, dict):
            target = moc_by_key.get(_key(str(result.get("target_moc") or "")))
            evidence = re.sub(r"\s+", " ", str(result.get("evidence") or "")).strip()
            try:
                confidence = float(result.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            normalized_body = re.sub(r"\s+", "", content)
            evidence_ok = len(evidence) >= 8 and re.sub(r"\s+", "", evidence) in normalized_body
            if target and confidence >= 0.70 and evidence_ok:
                reason = str(result.get("reason") or "正文证据支持该主题归属")[:180]
                store.add_structural_link(
                    int(target["id"]), int(note["id"]), str(note["title"]),
                    "contains", f"classification_agent:{reason}｜依据：{evidence[:140]}", confidence,
                )
                detail.update({
                    "status": "classified", "target_moc": target["title"],
                    "confidence": round(min(1.0, confidence), 3), "evidence": evidence[:180],
                    "reason": reason,
                })
                classified += 1
            else:
                detail.update({
                    "confidence": round(max(0.0, min(1.0, confidence)), 3),
                    "evidence": evidence[:180],
                    "reason": str(result.get("reason") or "分类证据或置信度未达到自动写入门槛")[:180],
                })
        else:
            detail["reason"] = "理解 API 暂不可用，本轮保留待复核，不使用关键词硬分类。"
        queue[str(note["id"])] = detail

    active_ids = {str(item["id"]) for item in pending}
    queue = {key: value for key, value in queue.items() if key in active_ids and value.get("status") != "classified"}
    store.set_setting("classification_queue_v1", queue)
    if classified:
        store.add_activity("classification_agent", f"自动归类 {classified} 条新知，{len(queue)} 条等待复核", 4)
    return {
        "examined": len(pending), "classified": classified,
        "needs_review": len(queue), "items": list(queue.values()),
    }
