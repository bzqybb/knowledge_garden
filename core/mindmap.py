from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from typing import Any

from core.storage import GardenStore
from core.learning_memory import note_activation


VISIBLE_KINDS = {"domain", "knowledge", "moc", "concept"}

TITLE_ALIASES = {
    "attention": "注意力机制",
    "snn": "脉冲神经网络",
    "surrogategradient": "代理梯度",
    "backpropagation": "反向传播",
    "heaviside": "阶跃函数",
    "计算机科学与技术": "计算机科学",
    "电气与电子工程": "电子工程",
    "传媒学": "传播学",
}

MOC_PARENT_HINTS = {
    "人工智能": "计算机科学",
    "神经网络": "人工智能",
    "类脑计算与snn": "神经网络",
    "神经形态计算": "神经网络",
    "神经形态计算与脉冲神经网络": "神经网络",
}

NON_KNOWLEDGE_TITLES = {
    "com", "phone", "robot", "code", "word", "skill", "cloud", "frontier",
    "bilibili", "pdf", "8大用法",
}


def _knowledge_eligibility(note: dict[str, Any]) -> tuple[bool, str]:
    """Keep extraction residue and staging notes out of the canonical map.

    This is intentionally a presentation quarantine, not destructive cleanup:
    the original note remains available for later review or reclassification.
    """
    title = str(note.get("title") or "").strip()
    tags = {str(item).strip().casefold() for item in note.get("tags", [])}
    if note.get("kind") in {"domain", "knowledge"} and str(note.get("path", "")).startswith("domain::"):
        return True, "curated_domain_taxonomy"
    if "待归类的新知".casefold() in tags or title == "待归类的新知":
        return False, "staging_area"
    if re.fullmatch(r"BV[0-9A-Za-z]{8,14}", title):
        return False, "source_identifier"
    if re.fullmatch(r"https?://\S+|www\.\S+", title, re.I):
        return False, "url_identifier"
    if title.casefold() in NON_KNOWLEDGE_TITLES:
        return False, "generic_extraction_token"
    if re.fullmatch(r"\d+\s*(?:大|种|个)?(?:用法|方法|技巧|步骤)", title):
        return False, "contextless_list_label"
    if re.match(r"^(?:为什么|怎么|如何|能不能|是否|请问|帮我)", title) or title.endswith(("?", "？")):
        return False, "uncompiled_question"
    if not title or len(title) > 90:
        return False, "invalid_title"
    return True, "eligible"


def _canonical_title(title: str) -> str:
    normalized = unicodedata.normalize("NFKC", title).lower()
    normalized = re.sub(r"\s*[（(][^（）()]{1,80}[）)]\s*$", "", normalized)
    compact = re.sub(r"[\s·•_—–\-:：]+", "", normalized)
    return TITLE_ALIASES.get(compact, compact)


def _surface_terms(title: str) -> set[str]:
    """Return visible Chinese/English aliases for matching a branch name."""
    normalized = unicodedata.normalize("NFKC", title).lower()
    parts = re.split(r"[（）()]", normalized)
    return {
        re.sub(r"[\s·•_—–\-:：]+", "", part)
        for part in parts if re.sub(r"[\s·•_—–\-:：]+", "", part)
    }


def _summary(content: str, title: str) -> str:
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "---", "|", "```", "- [[", ">", "<!--")):
            continue
        line = re.sub(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", r"\1", line)
        line = re.sub(r"[*_`>]", "", line).strip()
        if line and line != title:
            return line[:140]
    return title


def branch_diagram_blueprint(tree: dict[str, Any], node_id: int) -> dict[str, Any] | None:
    """Create a bounded, auditable visual brief from one canonical subtree."""
    target: dict[str, Any] | None = None

    def find(node: dict[str, Any]) -> None:
        nonlocal target
        if target is not None:
            return
        if node.get("id") == node_id:
            target = node
            return
        for child in node.get("children", []):
            find(child)

    find(tree)
    if target is None:
        return None
    relations: list[str] = []
    queue = [target]
    node_count = 0
    while queue and node_count < 16:
        parent = queue.pop(0)
        node_count += 1
        for child in parent.get("children", []):
            if node_count + len(queue) >= 16:
                break
            relations.append(f"{parent.get('title', '')} 包含 {child.get('title', '')}")
            queue.append(child)
    return {
        "research_object": str(target.get("title") or "知识分支"),
        "core_question": f"展开 {target.get('title', '知识分支')} 的已学知识结构",
        "usable_claims": relations[:15],
        "explanation_order": [
            str(child.get("title")) for child in target.get("children", [])[:8]
        ],
        "direct_source_ids": [],
        "evidence_items": [],
        "gaps": [],
        "canonical_subtree": target,
        "visible_node_count": node_count,
    }


def build_mindmap(store: GardenStore) -> dict[str, Any]:
    """Compile the graph into one canonical discipline → branch → topic → knowledge tree."""
    with store.connect() as conn:
        rows = conn.execute(
            """SELECT id,path,title,kind,content,tags_json,updated_at,
                      base_importance,activation_score,access_count,last_accessed_at,stability_days
               FROM notes
               WHERE kind IN ('domain','knowledge','moc','concept')
               ORDER BY title"""
        ).fetchall()
        notes = {}
        quarantined: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["tags"] = json.loads(item.pop("tags_json") or "[]")
            item["summary"] = _summary(item["content"], item["title"])
            item["knowledge_value"] = note_activation(item)
            eligible, reason = _knowledge_eligibility(item)
            if not eligible:
                quarantined.append({"id": item["id"], "title": item["title"], "reason": reason})
                continue
            notes[item["id"]] = item
        if not notes:
            return {
                "tree": {"id": "root", "title": "我的知识花园", "kind": "root", "children": []},
                "cross_links": [],
                "quality": {
                    "quarantined_count": len(quarantined),
                    "quarantined": quarantined[:40],
                    "policy": "knowledge-admission-v2",
                },
            }
        marks = ",".join("?" for _ in notes)
        edges = [dict(row) for row in conn.execute(
            f"""SELECT id,source_id,target_id,relation,strength,explanation,status
                FROM links WHERE source_id IN ({marks}) AND target_id IN ({marks}) AND status!='rejected'""",
            (*notes.keys(), *notes.keys()),
        )]

    contains: dict[int, list[int]] = defaultdict(list)
    parents: set[int] = set()
    for edge in edges:
        if edge["relation"] == "contains":
            contains[edge["source_id"]].append(edge["target_id"])
            parents.add(edge["target_id"])

    discipline_ids = [
        note_id for note_id, note in notes.items()
        if note["kind"] == "domain" and note["path"].startswith("domain::discipline::")
    ]
    raw_moc_ids = [note_id for note_id, note in notes.items() if note["kind"] == "moc"]
    moc_groups: dict[str, list[int]] = defaultdict(list)
    for note_id in raw_moc_ids:
        moc_groups[_canonical_title(notes[note_id]["title"])].append(note_id)
    moc_alias: dict[int, int] = {}
    moc_ids = []
    for group in moc_groups.values():
        representative = max(group, key=lambda item: len(notes[item]["content"]))
        moc_ids.append(representative)
        moc_alias.update({item: representative for item in group})
    raw_concept_ids = [note_id for note_id, note in notes.items() if note["kind"] == "concept"]
    concept_groups: dict[str, list[int]] = defaultdict(list)
    for note_id in raw_concept_ids:
        concept_groups[_canonical_title(notes[note_id]["title"])].append(note_id)
    concept_alias: dict[int, int] = {}
    concept_group_titles: dict[int, set[str]] = {}
    concept_ids = []
    for group in concept_groups.values():
        representative = max(group, key=lambda item: len(notes[item]["content"]))
        display = min(group, key=lambda item: (len(notes[item]["title"]), notes[item]["title"]))
        notes[representative]["title"] = notes[display]["title"]
        concept_ids.append(representative)
        concept_alias.update({item: representative for item in group})
        concept_group_titles[representative] = {notes[item]["title"] for item in group}
    attached: set[int] = set()

    def make_node(note_id: int, trail: set[int] | None = None) -> dict[str, Any]:
        trail = set(trail or ())
        if note_id in trail:
            return {"id": note_id, "title": notes[note_id]["title"], "kind": notes[note_id]["kind"], "children": []}
        trail.add(note_id)
        note = notes[note_id]
        child_ids = sorted(set(contains.get(note_id, [])), key=lambda child_id: notes[child_id]["title"])
        attached.update(child_ids)
        return {
            "id": note_id,
            "title": note["title"],
            "kind": note["kind"],
            "summary": note["summary"],
            "tags": note["tags"],
            "knowledge_value": note["knowledge_value"],
            "children": [make_node(child_id, trail) for child_id in child_ids],
        }

    top_nodes = [make_node(note_id) for note_id in sorted(discipline_ids, key=lambda item: notes[item]["title"])]

    # MOC-to-MOC links define the visible discipline → branch hierarchy. Only
    # declarative concept pages become leaves; cards, sources and questions stay
    # in node details.
    moc_set = set(moc_ids)
    moc_children: dict[int, set[int]] = defaultdict(set)
    child_mocs: set[int] = set()
    explicit_moc_parent: dict[int, int] = {}
    for child in moc_ids:
        child_key = _canonical_title(notes[child]["title"])
        candidates = [
            parent for parent in moc_ids if parent != child
            and _canonical_title(notes[parent]["title"]) in {
                _canonical_title(tag) for tag in notes[child].get("tags", [])
            }
            and _canonical_title(notes[parent]["title"]) != child_key
        ]
        if candidates:
            explicit_moc_parent[child] = max(
                candidates, key=lambda item: len(_canonical_title(notes[item]["title"]))
            )
    for child, parent in explicit_moc_parent.items():
        moc_children[parent].add(child)
        child_mocs.add(child)

    moc_by_key = {_canonical_title(notes[item]["title"]): item for item in moc_ids}
    for child_key, parent_key in MOC_PARENT_HINTS.items():
        child = moc_by_key.get(_canonical_title(child_key))
        parent = moc_by_key.get(_canonical_title(parent_key))
        if child is not None and parent is not None and child != parent:
            old_parent = explicit_moc_parent.get(child)
            if old_parent is not None:
                moc_children[old_parent].discard(child)
            explicit_moc_parent[child] = parent
            moc_children[parent].add(child)
            child_mocs.add(child)

    for edge in edges:
        if edge["relation"] != "wikilink":
            continue
        left = moc_alias.get(edge["source_id"])
        right = moc_alias.get(edge["target_id"])
        if left not in moc_set or right not in moc_set or left == right:
            continue
        if left in explicit_moc_parent or right in explicit_moc_parent:
            continue
        left_key = _canonical_title(notes[left]["title"])
        right_key = _canonical_title(notes[right]["title"])
        # A cross-reference is not automatically a hierarchy. Only infer a
        # parent when one visible label genuinely specializes the other.
        if left_key not in right_key and right_key not in left_key:
            continue
        parent, child = (left, right) if len(left_key) <= len(right_key) else (right, left)
        moc_children[parent].add(child)
        child_mocs.add(child)

    # Infer obvious subject hierarchy even when the Wiki links were authored in
    # the opposite direction or not yet present: “社会与文化心理学” belongs under
    # the less specific “心理学”, rather than becoming a parallel discipline.
    for child in moc_ids:
        child_key = _canonical_title(notes[child]["title"])
        candidates = [
            parent for parent in moc_ids if parent != child
            and child_key.endswith(_canonical_title(notes[parent]["title"]))
            and len(_canonical_title(notes[parent]["title"])) < len(child_key)
        ]
        if candidates and child not in child_mocs:
            parent = max(candidates, key=lambda item: len(_canonical_title(notes[item]["title"])))
            moc_children[parent].add(child)
            child_mocs.add(child)

    # One branch gets one structural parent. Extra relationships remain semantic
    # cross-links instead of duplicating the same branch in several places.
    parent_candidates: dict[int, list[int]] = defaultdict(list)
    for parent, children in moc_children.items():
        for child in children:
            parent_candidates[child].append(parent)
    moc_children = defaultdict(set)
    child_mocs = set()
    for child, candidates in parent_candidates.items():
        parent = max(candidates, key=lambda item: len(_canonical_title(notes[item]["title"])))
        moc_children[parent].add(child)
        child_mocs.add(child)

    claimed_concepts: set[int] = set()
    claimed_concept_titles = {_canonical_title(notes[item]["title"]) for item in moc_ids}

    def concept_anchor(moc_id: int, candidates: set[int]) -> int | None:
        """Choose a broad concept only when the labels provide real hierarchy evidence."""
        if len(candidates) < 3:
            return None
        moc_surface = "".join(_surface_terms(notes[moc_id]["title"]))
        scored = []
        for concept_id in candidates:
            aliases = set()
            for title in concept_group_titles.get(concept_id, {notes[concept_id]["title"]}):
                aliases.update(_surface_terms(title))
            direct_match = any(len(term) >= 3 and term in moc_surface for term in aliases)
            title = notes[concept_id]["title"]
            broad_name = bool(re.search(r"(机制|网络|模型|理论|系统|框架)$", re.sub(r"[（(].*$", "", title)))
            score = (10 if direct_match else 0) + (3 if broad_name else 0)
            if score:
                scored.append((score, -len(title), concept_id))
        return max(scored)[2] if scored else None

    def make_moc_node(moc_id: int, trail: set[int] | None = None) -> dict[str, Any]:
        trail = set(trail or ())
        if moc_id in trail:
            return {"id": moc_id, "title": notes[moc_id]["title"], "kind": "branch", "children": []}
        trail.add(moc_id)
        concept_ids = set()
        for edge in edges:
            if edge["relation"] == "contains" and edge["source_id"] == moc_id:
                other = concept_alias.get(edge["target_id"], edge["target_id"])
                if other in notes and notes[other]["kind"] == "concept" and _canonical_title(notes[other]["title"]) not in claimed_concept_titles:
                    concept_ids.add(other)
                continue
            if edge["relation"] != "wikilink":
                continue
            source_id = moc_alias.get(edge["source_id"], edge["source_id"])
            target_id = moc_alias.get(edge["target_id"], edge["target_id"])
            other = None
            if source_id == moc_id:
                other = target_id
            elif target_id == moc_id:
                other = source_id
            other = concept_alias.get(other, other)
            if other in notes and notes[other]["kind"] == "concept" and _canonical_title(notes[other]["title"]) not in claimed_concept_titles:
                concept_ids.add(other)
        concept_ids.difference_update(claimed_concepts)
        claimed_concepts.update(concept_ids)
        claimed_concept_titles.update(_canonical_title(notes[item]["title"]) for item in concept_ids)
        attached.update(concept_ids)
        moc_note = notes[moc_id]
        branch_nodes = [
            make_moc_node(item, trail)
            for item in sorted(moc_children.get(moc_id, set()), key=lambda item: notes[item]["title"])
        ]
        local_children: dict[int, set[int]] = defaultdict(set)
        local_parented: set[int] = set()
        for edge in edges:
            if edge["relation"] != "contains":
                continue
            parent = concept_alias.get(edge["source_id"], edge["source_id"])
            child = concept_alias.get(edge["target_id"], edge["target_id"])
            if parent in concept_ids and child in concept_ids and parent != child:
                local_children[parent].add(child)
                local_parented.add(child)

        def make_concept_node(concept_id: int, concept_trail: set[int] | None = None) -> dict[str, Any]:
            concept_trail = set(concept_trail or ())
            if concept_id in concept_trail:
                return {"id": concept_id, "title": notes[concept_id]["title"], "kind": "concept", "children": []}
            concept_trail.add(concept_id)
            note = notes[concept_id]
            return {
                "id": concept_id, "title": note["title"], "kind": "concept",
                "summary": note["summary"], "tags": note["tags"],
                "knowledge_value": note["knowledge_value"],
                "children": [
                    make_concept_node(item, concept_trail)
                    for item in sorted(local_children.get(concept_id, set()), key=lambda item: notes[item]["title"])
                ],
            }

        if local_children:
            roots = concept_ids - local_parented
            concept_nodes = [make_concept_node(item) for item in sorted(roots, key=lambda item: notes[item]["title"])]
        else:
            anchor = concept_anchor(moc_id, concept_ids)
            if anchor is not None:
                anchor_node = make_concept_node(anchor)
                anchor_node["children"] = [
                    make_concept_node(item) for item in sorted(concept_ids - {anchor}, key=lambda item: notes[item]["title"])
                ]
                concept_nodes = [anchor_node]
            else:
                concept_nodes = [make_concept_node(item) for item in sorted(concept_ids, key=lambda item: notes[item]["title"])]
        return {
            "id": moc_id,
            "title": moc_note["title"],
            "kind": "discipline" if moc_id not in child_mocs else "branch",
            "summary": moc_note["summary"],
            "tags": moc_note["tags"],
            "knowledge_value": moc_note["knowledge_value"],
            "children": branch_nodes + concept_nodes,
        }

    root_mocs = [item for item in moc_ids if item not in child_mocs]
    for moc_id in sorted(root_mocs, key=lambda item: notes[item]["title"]):
        top_nodes.append(make_moc_node(moc_id))

    rooted = set(discipline_ids) | set(moc_ids) | attached | parents
    remaining = [
        note_id for note_id in concept_ids
        if note_id not in rooted and note_id not in claimed_concepts
    ]
    # Unmounted concepts stay in the review queue; they are not promoted to a
    # fake discipline in the canonical learning map.
    unmounted_count = len(remaining)

    def merge_nodes(target: dict[str, Any], incoming: dict[str, Any]) -> None:
        existing = {_canonical_title(item["title"]): item for item in target.get("children", [])}
        for child in incoming.get("children", []):
            key = _canonical_title(child["title"])
            if key in existing:
                merge_nodes(existing[key], child)
            else:
                target.setdefault("children", []).append(child)

    merged_top: list[dict[str, Any]] = []
    top_by_key: dict[str, dict[str, Any]] = {}
    for node in top_nodes:
        key = _canonical_title(node["title"])
        if key in top_by_key:
            merge_nodes(top_by_key[key], node)
        else:
            top_by_key[key] = node
            merged_top.append(node)
    top_nodes = merged_top

    included_ids: set[int] = set()
    discipline_by_id: dict[int, str] = {}
    def collect(node: dict[str, Any], discipline: str) -> None:
        if isinstance(node.get("id"), int):
            included_ids.add(node["id"])
            discipline_by_id[node["id"]] = discipline
        for child in node.get("children", []):
            collect(child, discipline)
    for node in top_nodes:
        collect(node, node["title"])
    cross_candidates = [
        edge for edge in edges
        if edge["relation"] == "semantic"
        and edge["source_id"] in included_ids and edge["target_id"] in included_ids
        and discipline_by_id.get(edge["source_id"]) != discipline_by_id.get(edge["target_id"])
    ]
    cross_links = sorted(
        cross_candidates,
        key=lambda edge: (edge["status"] == "accepted", float(edge["strength"])),
        reverse=True,
    )[:16]
    return {
        "tree": {"id": "root", "title": "我的知识花园", "kind": "root", "children": top_nodes},
        "cross_links": cross_links,
        "quality": {
            "quarantined_count": len(quarantined),
            "unmounted_count": unmounted_count,
            "quarantined": quarantined[:40],
            "policy": "knowledge-admission-v2",
        },
    }
