from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from core.learning_memory import LearningMemoryService
from core.llm import LLMError, chat_json
from core.obsidian import parse_markdown, write_wiki_asset
from core.retrieval import search_notes
from core.storage import GardenStore, utc_now


def _fallback_type(text: str) -> tuple[str, bool]:
    if re.search(r"如果|假如|假设|会怎样", text):
        return "counterfactual", bool(re.search(r"人类|大脑|物理|生物|心理|社会|算法", text))
    if re.search(r"是不是|有没有可能|能否.*解释|跨学科|像不像", text):
        return "cross_disciplinary_hypothesis", True
    if re.search(r"我感觉|我总觉得|直觉", text):
        return "intuition_record", True
    return "open_exploration", True


def explore_inspiration(
    store: GardenStore, message: str, history: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    message = message.strip()
    if not message:
        raise ValueError("先写下你想探索的直觉")
    history = [
        {"role": str(item.get("role", "")), "content": str(item.get("content", ""))[:2500]}
        for item in (history or [])[-12:] if item.get("role") in {"user", "assistant"}
    ]
    kind, needs_anchor = _fallback_type(message)
    retrieval_text = "\n".join([*(item["content"] for item in history if item["role"] == "user"), message])
    hits = search_notes(
        store, retrieval_text, kinds={"textbook", "course", "concept", "knowledge"}, limit=4,
    ) if needs_anchor else []
    anchors = [item for item in hits if item.get("knowledge_status") == "grounded"][:2]
    anchor_text = "\n".join(
        f"[{index}]《{item['title']}》：{item['snippet']}" for index, item in enumerate(anchors, 1)
    ) or "本轮没有取得可核验的事实锚点"
    payload = None
    try:
        payload = chat_json(
            "你是知识花园的灵感孵化 Agent，不是答案裁判。先准确接住用户的直觉，再开放探索。"
            "输出 JSON：primary_type、secondary_types、acknowledgement、assumptions、claims、counter_view、branches。"
            "claims 每项含 status(fact/inference/imagination/uncertain)、text、anchor_index；fact 只能引用给定锚点，"
            "无锚点的现实断言必须标 uncertain。branches 给 3 个真正不同的探索方向，每项含 title、question。"
            "不要替用户选择下一步，不要把分支写成必须二选一。",
            f"历史：{history or '无'}\n当前输入：{message}\n初步类型：{kind}\n事实锚点：\n{anchor_text}",
        )
    except LLMError:
        payload = None
    payload = payload or {
        "primary_type": kind, "secondary_types": [],
        "acknowledgement": "我先把你的想法当作一个值得展开的假设，而不是急着判断它对不对。",
        "assumptions": [message],
        "claims": [
            {"status": "uncertain", "text": "这个想法目前还缺少足够的事实锚点，但可以先拆解它依赖的条件。", "anchor_index": None},
            {"status": "imagination", "text": "如果暂时接受这个设定，可以观察它会改变哪些机制、边界与日常经验。", "anchor_index": None},
        ],
        "counter_view": "也可以反过来问：有没有一种更简单的解释，会产生同样的直觉？",
        "branches": [
            {"title": "机制", "question": "这个想法若成立，最小的作用机制可能是什么？"},
            {"title": "反例", "question": "哪一种情形最可能让这个想法失效？"},
            {"title": "应用", "question": "把它放进一个具体场景，会产生什么可观察结果？"},
        ],
    }
    valid_anchor_indexes = set(range(1, len(anchors) + 1))
    claims = []
    for raw in payload.get("claims") or []:
        status = str(raw.get("status", "uncertain"))
        anchor_index = raw.get("anchor_index")
        if status == "fact" and anchor_index not in valid_anchor_indexes:
            status = "uncertain"
            anchor_index = None
        if status not in {"fact", "inference", "imagination", "uncertain"}:
            status = "uncertain"
        claims.append({"status": status, "text": str(raw.get("text", "")).strip(), "anchor_index": anchor_index})
    memory = LearningMemoryService(store)
    turn = memory.begin_turn(message, session_id, capability="inspiration")
    session_id = turn["session_id"]
    event_id = memory.record_event(
        surface="inspiration", event_type="inspiration_turn", source_kind="observed",
        session_id=session_id, message_id=turn["message_id"], concepts=[], payload={
            "l1_only": True, "eligible_for_reflection": False,
            "primary_type": payload.get("primary_type", kind), "message": message[:1000],
        },
    )
    return {
        "session_id": session_id, "event_id": event_id, "mode": "inspiration",
        "primary_type": payload.get("primary_type", kind),
        "secondary_types": payload.get("secondary_types") or [],
        "acknowledgement": str(payload.get("acknowledgement") or "我认真接住了这个想法。"),
        "assumptions": [str(item) for item in (payload.get("assumptions") or [])][:4],
        "claims": claims[:8], "counter_view": str(payload.get("counter_view") or ""),
        "branches": (payload.get("branches") or [])[:4],
        "anchors": [{"id": item["id"], "title": item["title"], "path": item["path"]} for item in anchors],
        "notice": "[推测]、[灵感]与[待核验]不会更新掌握度，也不会自动形成长期画像。",
    }


def save_inspiration_seed(
    store: GardenStore, title: str, messages: list[dict[str, Any]], latest: dict[str, Any], tags: list[str] | None = None,
) -> dict[str, Any]:
    title = title.strip() or "未命名灵感"
    transcript = "\n\n".join(
        f"**{'你' if item.get('role') == 'user' else '园丁'}**：{str(item.get('content', '')).strip()}"
        for item in messages if item.get("role") in {"user", "assistant"}
    )
    body = (
        "---\nasset_type: inspiration_seed\nverification_status: unverified\n"
        "eligible_for_factual_retrieval: false\n---\n\n"
        "# " + title + "\n\n> 这是未核验的灵感种子，不作为事实依据或掌握度证据。\n\n"
        "## 当前假设\n" + "\n".join(f"- {item}" for item in latest.get("assumptions", [])) +
        "\n\n## 探索对话\n" + transcript
    )
    vault = str(store.setting("vault_path", "") or "")
    if vault:
        output = write_wiki_asset(vault, "03-交叉火花/灵感种子", title, body, ["灵感种子", "未核验", *(tags or [])])
        note = parse_markdown(output, Path(vault).expanduser().resolve())
        note_id, _ = store.upsert_note(note)
        path = str(output)
    else:
        note_id, _ = store.upsert_note({
            "path": f"inspiration::{utc_now()}::{title}", "title": title, "kind": "interest",
            "content": body, "tags": ["灵感种子", "未核验", *(tags or [])],
            "source": "inspiration_seed", "content_hash": str(hash(body)),
        })
        path = ""
    store.add_activity("save_inspiration_seed", title, 5)
    return {"note_id": note_id, "path": path, "status": "unverified", "title": title}
