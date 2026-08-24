from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from core.engine import CONCEPT_PROFILES, analyze_material_structure
from core.obsidian import WIKILINK_RE, append_backlink, write_wiki_asset
from core.storage import GardenStore


TITLE_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
CATEGORY_RE = re.compile(r"^-\s*Category\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE)


def _safe_name(value: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "-", value).strip(". ")[:100]


def _find_topic(vault: Path, source_stem: str, category: str) -> str:
    moc_dir = vault / "wiki" / "04-主题索引"
    if moc_dir.is_dir():
        for path in moc_dir.glob("*.md"):
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            if source_stem in text:
                return path.stem
    parts = [part.strip() for part in re.split(r"[/、,，]", category) if part.strip()]
    return parts[-1] if parts else "待分类主题"


def _concept_path(vault: Path, concept: str) -> Path:
    return vault / "wiki" / "01-概念底座" / f"{_safe_name(concept)}.md"


COMPILED_START = "<!-- knowledge-gardener:compiled:start -->"
COMPILED_END = "<!-- knowledge-gardener:compiled:end -->"

TOPIC_DESCRIPTIONS = {
    "心理学": "心理学研究行为与心理过程，并关注个体、情境、社会关系和文化如何共同塑造人的经验。",
    "社会与文化心理学": "社会与文化心理学研究社会情境、群体规范与文化系统如何影响自我、人格判断和行为。这里的重点不是给文化贴标签，而是分析变量之间的机制与测量边界。",
    "类脑计算与 SNN": "类脑计算尝试借鉴生物神经系统的信息表示与学习方式；脉冲神经网络以离散脉冲和时间动态为核心。",
}


def _source_excerpt(text: str, concept: str, width: int = 260) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    variants = {concept, concept.replace("-", "—"), concept.replace("—", "-")}
    positions = [clean.find(item) for item in variants if clean.find(item) >= 0]
    center = min(positions) if positions else 0
    start = max(0, center - 80)
    excerpt = clean[start:start + width].strip()
    return ("……" if start else "") + excerpt + ("……" if start + width < len(clean) else "")


def _compiled_concept_body(
    concept: str, source_stem: str, bridge_title: str, topic: str,
    source_text: str, related: list[str],
) -> str:
    profile = CONCEPT_PROFILES.get(concept, {})
    evidence = _source_excerpt(source_text, concept)
    definition = profile.get("definition") or f"在当前资料中，**{concept}**用于解释以下现象：{evidence}"
    mechanism = profile.get("mechanism") or "先识别它改变的对象、作用条件和结果变量，再判断这种关系是因果机制、统计关联还是解释性类比。"
    example = profile.get("example") or f"可以先回到来源中的具体情境，检查“{concept}”出现前后的条件与结果是否发生了可观察变化。"
    boundary = profile.get("boundary") or "当前页面只依据一份来源建立；在获得独立研究、反例或教材定义前，不应把它视为无条件成立的结论。"
    questions = profile.get("questions") or [
        f"如果要用一个可观察变量检验“{concept}”，你会测量什么？",
        "哪个反例最可能迫使我们修改上面的机制解释？",
    ]
    related_md = "\n".join(f"- [[{item}]]：与本概念共同解释当前材料。" for item in related if item != concept)
    return (
        f"{COMPILED_START}\n"
        f"> **知识状态：已从来源编译，可继续被新证据修订。**\n\n"
        f"## 核心定义\n\n{definition}\n\n"
        f"## 它如何起作用\n\n{mechanism}\n\n"
        f"## 来源中的证据\n\n> {evidence}\n\n"
        f"## 具体例子\n\n{example}\n\n"
        f"## 适用边界与常见误解\n\n{boundary}\n\n"
        f"## 在知识树中的关系\n\n- 主题：[[{topic}]]\n- 对照卡：[[{bridge_title}]]\n"
        f"{related_md}\n\n"
        f"## 来源\n\n- [[sources/{source_stem}]]\n\n"
        f"## 主动回忆\n\n- {questions[0]}\n- {questions[1]}\n\n"
        f"## 苏格拉底追问\n\n> 在什么条件下，上面的解释会失效？你需要什么证据才能判断？\n"
        f"{COMPILED_END}"
    )


def _ensure_concept(
    vault: Path, concept: str, source_stem: str, bridge_title: str, topic: str,
    source_text: str, related: list[str],
) -> tuple[Path, bool]:
    path = _concept_path(vault, concept)
    compiled = _compiled_concept_body(concept, source_stem, bridge_title, topic, source_text, related)
    if path.exists():
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        if "此页由 Ingest 自动建立" in text:
            write_wiki_asset(vault, "01-概念底座", concept, compiled, ["概念底座", concept, topic])
        elif COMPILED_START in text and COMPILED_END in text:
            updated = re.sub(
                re.escape(COMPILED_START) + r".*?" + re.escape(COMPILED_END),
                compiled,
                text,
                flags=re.DOTALL,
            )
            path.write_text(updated, encoding="utf-8")
        append_backlink(path, bridge_title)
        append_backlink(path, topic)
        return path, False
    path = write_wiki_asset(vault, "01-概念底座", concept, compiled, ["概念底座", concept, topic])
    return path, True


def _ensure_moc(
    vault: Path, topic: str, concepts: list[str], bridge_title: str, source_stem: str,
    discipline: str = "",
) -> tuple[Path, bool]:
    path = vault / "wiki" / "04-主题索引" / f"{_safe_name(topic)}.md"
    topic_block = (
        "<!-- knowledge-gardener:topic:start -->\n"
        f"## 主题说明\n\n{TOPIC_DESCRIPTIONS.get(topic, f'{topic}汇集该方向已经学过的概念、来源证据和对照卡，并随着新资料持续修订。')}\n"
        "<!-- knowledge-gardener:topic:end -->"
    )
    if path.exists():
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        if "<!-- knowledge-gardener:topic:start -->" in text:
            text = re.sub(
                r"<!-- knowledge-gardener:topic:start -->.*?<!-- knowledge-gardener:topic:end -->",
                topic_block, text, flags=re.DOTALL,
            )
        else:
            heading_end = text.find("\n", text.find("# "))
            text = text[:heading_end + 1] + "\n" + topic_block + "\n" + text[heading_end + 1:]
        additions = []
        for title in [*concepts, bridge_title, f"sources/{source_stem}"]:
            if f"[[{title}]]" not in text:
                additions.append(f"- [[{title}]]")
        if additions:
            text = text.rstrip() + "\n\n## 自动挂载\n\n" + "\n".join(additions) + "\n"
        path.write_text(text, encoding="utf-8")
        return path, False
    body = (
        "> **主题 MOC**（Map of Content）：由知识编译器持续维护。\n\n" + topic_block + "\n\n"
        "## 概念底座\n\n" + "\n".join(f"- [[{item}]]" for item in concepts) +
        f"\n\n## 降维对照\n\n- [[{bridge_title}]]\n\n## 来源\n\n- [[sources/{source_stem}]]\n\n"
        "## 苏格拉底追问\n\n> 这个主题里目前缺失的关键反例或竞争性解释是什么？补上它会改变哪条已有连接？"
    )
    return write_wiki_asset(vault, "04-主题索引", topic, body, ["MOC", topic, *([discipline] if discipline else [])]), True


def _ensure_discipline_moc(vault: Path, discipline: str, topic: str, source_stem: str) -> tuple[Path, bool]:
    """Create the top discipline and mount a topic MOC beneath it."""
    path = vault / "wiki" / "04-主题索引" / f"{_safe_name(discipline)}.md"
    topic_link = f"[[{topic}]]"
    source_link = f"[[sources/{source_stem}]]"
    description = TOPIC_DESCRIPTIONS.get(
        discipline, f"{discipline}是知识花园中的学科主干；它只承载已形成内容的分支和知识点。"
    )
    discipline_block = (
        "<!-- knowledge-gardener:topic:start -->\n"
        f"## 学科说明\n\n{description}\n"
        "<!-- knowledge-gardener:topic:end -->"
    )
    if path.exists():
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        if "<!-- knowledge-gardener:topic:start -->" in text:
            text = re.sub(
                r"<!-- knowledge-gardener:topic:start -->.*?<!-- knowledge-gardener:topic:end -->",
                discipline_block, text, flags=re.DOTALL,
            )
        else:
            heading_end = text.find("\n", text.find("# "))
            text = text[:heading_end + 1] + "\n" + discipline_block + "\n" + text[heading_end + 1:]
        additions = [link for link in (topic_link, source_link) if link not in text]
        if additions:
            text = text.rstrip() + "\n\n## 自动生长的分支\n\n" + "\n".join(f"- {link}" for link in additions) + "\n"
        path.write_text(text, encoding="utf-8")
        return path, False
    body = (
        "> **学科主干 MOC**：从该学科的分支向外生长已学知识点。\n\n"
        f"{discipline_block}\n\n"
        f"## 学科分支\n\n- {topic_link}\n\n## 来源\n\n- {source_link}\n\n"
        "## 园丁提示\n\n问题与复习任务不会作为主图节点；它们保留在知识点详情中。"
    )
    return write_wiki_asset(vault, "04-主题索引", discipline, body, ["MOC", "学科", discipline]), True


def validate_links(vault_path: str | Path) -> dict[str, Any]:
    vault = Path(vault_path).expanduser().resolve()
    files = list((vault / "wiki").rglob("*.md"))
    raw_files = list((vault / "raw").rglob("*.md")) if (vault / "raw").is_dir() else []
    stems = {path.stem for path in [*files, *raw_files]}
    relative_targets = {str(path.relative_to(vault / "wiki").with_suffix("")).replace("\\", "/") for path in files}
    unresolved: dict[str, list[str]] = {}
    for path in files:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        missing = []
        for target in WIKILINK_RE.findall(text):
            target = target.strip()
            if target not in stems and target not in relative_targets and not (vault / f"{target}.md").exists():
                missing.append(target)
        if missing:
            unresolved[str(path.relative_to(vault)).replace("\\", "/")] = sorted(set(missing))
    return {"files": len(files), "unresolved_count": sum(map(len, unresolved.values())), "unresolved": unresolved}


def ingest_raw(vault_path: str | Path, raw_relative: str, store: GardenStore | None = None) -> dict[str, Any]:
    vault = Path(vault_path).expanduser().resolve()
    raw_path = (vault / raw_relative).resolve()
    if vault not in raw_path.parents or not raw_path.is_file():
        raise ValueError("raw 文件不存在或不在当前 Vault 中")
    text = raw_path.read_text(encoding="utf-8-sig", errors="replace")
    title_match = TITLE_RE.search(text)
    title = title_match.group(1).strip() if title_match else raw_path.stem
    category_match = CATEGORY_RE.search(text)
    category = category_match.group(1).strip() if category_match else "待分类"
    structure = analyze_material_structure(
        title, text,
        str(store.setting("learning_level", "本科入门")) if store else "本科入门",
        category,
    )
    inferred_discipline = str(structure.get("discipline") or "跨学科探索")
    inferred_topic = str(structure.get("topic") or "待归类的新知")
    topic = inferred_topic if category == "待分类" else _find_topic(vault, raw_path.stem, category)
    discipline = inferred_discipline
    extracted_concepts = [str(item) for item in (structure.get("concepts") or []) if str(item).strip()]
    # Explicit [[wikilinks]] are author intent. AGENTS.md requires every missing
    # concept target to receive a substantive compiled page even if the model did
    # not rank it among the top extracted concepts.
    explicit_concepts = [
        target.strip() for target in WIKILINK_RE.findall(text)
        if target.strip() and "/" not in target and not target.strip().startswith("sources")
    ]
    existing_concepts = []
    concept_dir = vault / "wiki" / "01-概念底座"
    if concept_dir.is_dir():
        for path in concept_dir.glob("*.md"):
            if path.stem in text:
                existing_concepts.append(path.stem)
    concepts = list(dict.fromkeys([*existing_concepts, *explicit_concepts, *extracted_concepts]))[:12]
    bridge_title = f"{title}｜降维对照"
    existing_bridge = next((p for p in (vault / "wiki" / "02-降维对照").glob("*.md") if raw_path.stem in p.read_text(encoding="utf-8-sig", errors="replace")), None)
    if existing_bridge:
        bridge_title = existing_bridge.stem

    created: list[str] = []
    for concept in concepts:
        own_bridge = vault / "wiki" / "02-降维对照" / f"{_safe_name(concept + '｜教材—前沿对照')}.md"
        concept_bridge_title = own_bridge.stem if own_bridge.exists() else bridge_title
        path, is_new = _ensure_concept(
            vault, concept, raw_path.stem, concept_bridge_title, topic, text, concepts
        )
        if is_new:
            created.append(str(path.relative_to(vault)))

    bridge_path = vault / "wiki" / "02-降维对照" / f"{_safe_name(bridge_title)}.md"
    if not bridge_path.exists():
        body = (
            f"> **降维对照卡**：将“{title}”映射回可复用的概念底座。\n\n"
            f"## 原始问题\n\n{text[:900].strip()}\n\n## 底层映射\n\n" +
            "\n".join(f"- [[{concept}]]：它解释了原始材料中的一个关键机制或假设。" for concept in concepts) +
            f"\n\n## 来源\n\n- [[sources/{raw_path.stem}]]\n\n## 相关链接\n\n- [[{topic}]]\n\n"
            "## 苏格拉底追问\n\n> 如果移除其中一个底层概念，原材料的结论会在哪个环节首先失效？\n\n"
            "> 哪个映射目前只是类比而非严格推导？需要什么证据才能把它升级为机制解释？"
        )
        bridge_path = write_wiki_asset(vault, "02-降维对照", bridge_title, body, ["降维对照", topic])
        created.append(str(bridge_path.relative_to(vault)))

    moc_path, moc_new = _ensure_moc(vault, topic, concepts, bridge_title, raw_path.stem, discipline)
    if moc_new:
        created.append(str(moc_path.relative_to(vault)))
    if discipline and discipline != topic:
        discipline_path, discipline_new = _ensure_discipline_moc(vault, discipline, topic, raw_path.stem)
        if discipline_new:
            created.append(str(discipline_path.relative_to(vault)))

    source_path = vault / "wiki" / "sources" / f"{raw_path.stem}.md"
    if not source_path.exists():
        classification_evidence = structure.get("classification_evidence") or []
        source_body = (
            f"- **原始资料**：[[{raw_path.stem}]]（`{raw_relative}`）\n- **分类**：{category}\n\n"
            f"- **知识位置**：[[{discipline}]] → [[{topic}]]\n"
            f"- **分类方法**：{structure.get('method', 'fallback')} · 置信度 {float(structure.get('confidence', 0)):.0%}\n\n"
            "## 分类依据\n\n" + (
                "\n".join(f"> {item}" for item in classification_evidence)
                if classification_evidence else "> 当前没有取得足以自动确认分类的正文证据；该位置应视为待复核。"
            ) + "\n\n"
            f"## 核心概念\n\n" + "、".join(f"[[{item}]]" for item in concepts) +
            f"\n\n## 编译产物\n\n- [[{bridge_title}]]\n- [[{topic}]]\n\n"
            "## 苏格拉底追问\n\n> 这份资料最强的隐含假设是什么？若该假设不成立，哪些编译产物需要修订？"
        )
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(
            "---\ngarden_generated: true\ntags:\n  - 来源\n  - " + topic +
            f"\n---\n\n# 来源：{title}\n\n{source_body.strip()}\n",
            encoding="utf-8",
        )
        created.append(str(source_path.relative_to(vault)))

    for concept in concepts:
        append_backlink(_concept_path(vault, concept), bridge_title)
    if store:
        store.add_activity("ingest_raw", raw_path.name, 20 if created else 3)
    return {
        "source": raw_relative, "discipline": discipline, "topic": topic, "concepts": concepts,
        "bridge": bridge_title, "created": created, "links": validate_links(vault),
        "classification": {
            "method": structure.get("method", "fallback"),
            "confidence": structure.get("confidence", 0),
            "evidence": structure.get("classification_evidence", []),
        },
    }
