from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.llm import LLMError, chat_json
from core.obsidian import parse_markdown, write_raw_material, write_wiki_asset
from core.retrieval import search_notes, tokenize
from core.storage import GardenStore, utc_now


TECH_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9+\-]{2,}|[\u4e00-\u9fff]{2,12}(?:算法|模型|网络|定理|原理|效应|方法|机制|理论|系统|结构|函数|学习)")

# Offline mode must still extract noun-like knowledge points instead of arbitrary
# sentence tails that merely happen to end in “效应/理论/机制”. Specific concepts
# are deliberately ordered before broad umbrella terms.
KNOWN_CONCEPTS = [
    "参照群体效应", "特质-情境匹配", "特质—情境匹配", "主动情境选择", "人格三角",
    "跨文化心理学", "文化心理学", "测量不变性", "文化调节效应", "心理普遍性",
    "代理梯度", "脉冲神经网络", "替代梯度", "反向传播", "注意力机制", "梯度下降",
    "集体主义", "个人主义",
]

CONCEPT_PROFILES: dict[str, dict[str, Any]] = {
    "参照群体效应": {
        "definition": "人们进行自我评价时，常以所属文化或群体中的典型成员作为比较基准；因此不同群体得到相同问卷分数，也未必代表相同的心理水平。",
        "mechanism": "文化内部的比较标准改变了量表刻度的实际含义，使群体间均值差异同时混入了‘真实差异’与‘参照标准差异’。",
        "example": "两个文化群体都给自己的外向性打 7 分，其中一组可能是在更外向的群体规范下完成比较，7 分的行为含义便不完全相同。",
        "boundary": "它提醒我们谨慎比较自陈量表，但不能据此断言所有跨文化差异都是测量幻觉；仍需结合测量不变性、行为数据等证据。",
        "keypoints": ["参照", "群体", "文化", "问卷", "比较", "含义"],
        "questions": ["为什么两个文化群体得到相同问卷分数时，仍可能具有不同含义？", "你会增加哪一种行为指标来检验参照群体效应？"],
        "quiz": {"question": "参照群体效应最直接威胁哪一种推断？", "options": ["直接比较不同文化群体的问卷均值", "记录同一个人的反应时间", "计算同一量表的题目数量", "描述单个被试的访谈内容"], "answer": 0},
    },
    "特质-情境匹配": {
        "definition": "个体更可能进入与自身稳定特质相匹配的情境，因此观察到的行为既受特质影响，也受其选择了什么环境影响。",
        "mechanism": "特质影响情境选择，情境再影响行为；这形成‘特质 → 情境暴露 → 行为’的间接路径。",
        "example": "偏内向的人持续避开高社交场景，会表现出更高的行为一致性，但这种一致性部分来自环境筛选。",
        "boundary": "匹配不意味着情境失去作用，也不意味着每次选择都由人格决定；机会、规范和资源仍会限制可选情境。",
        "keypoints": ["特质", "情境", "选择", "行为", "间接"],
        "questions": ["为什么行为一致性不能被直接当作特质作用的纯证据？", "如何区分‘特质直接影响行为’与‘特质先影响情境选择’？"],
        "quiz": {"question": "哪条路径最能表达特质—情境匹配？", "options": ["特质影响情境选择，情境进一步影响行为", "情境完全消除人格差异", "问卷分数自动等于真实行为", "文化为每个人创造不同心理机制"], "answer": 0},
    },
    "主动情境选择": {
        "definition": "个体并非被动接受环境，而会依据偏好、能力与人格主动进入、维持或避开某些情境。",
        "mechanism": "选择行为改变了一个人长期接触的环境分布，进而放大或稳定某些行为模式。",
        "example": "不喜欢高强度社交的人反复选择安静活动，之后观察到的低社交行为不能只归因于当下情境。",
        "boundary": "主动选择受现实机会约束；当制度、家庭或经济条件限制选择时，观察到的情境暴露未必代表真实偏好。",
        "keypoints": ["主动", "选择", "进入", "避开", "环境", "行为"],
        "questions": ["主动情境选择为什么会让人格看起来比实际更稳定？", "在无法自由选择环境时，这个机制会怎样变化？"],
        "quiz": {"question": "主动情境选择强调的核心是什么？", "options": ["个体会改变自己长期接触的环境分布", "行为只由当前环境决定", "人格与环境彼此独立", "所有文化中的选择机会完全相同"], "answer": 0},
    },
}


def infer_taxonomy(title: str, text: str, category: str = "") -> tuple[str, str]:
    """Return a stable discipline and branch for Wiki/Mindmap placement."""
    explicit = [part.strip() for part in re.split(r"[/、,，]", category) if part.strip() and part.strip() != "待分类"]
    if len(explicit) >= 2:
        return explicit[0], explicit[-1]
    haystack = f"{title}\n{text}".lower()
    psychology_terms = [
        "心理", "人格", "特质", "情境", "集体主义", "个人主义", "文化差异",
        "参照群体", "问卷", "行为状态", "跨文化",
    ]
    if sum(haystack.count(term) for term in psychology_terms) >= 3:
        return "心理学", "社会与文化心理学"
    if any(term in haystack for term in ["脉冲神经网络", "snn", "代理梯度", "类脑计算"]):
        return "计算机科学与人工智能", "类脑计算与 SNN"
    if explicit:
        return explicit[0], explicit[-1]
    return "跨学科探索", "待归类的新知"


def _valid_concept(value: str) -> bool:
    value = value.strip(" \t\r\n：:，,。；;、-—")
    if not (2 <= len(value) <= 24):
        return False
    sentence_prefixes = ("而非", "意义", "可能", "因此", "一个", "这种", "该发现", "这意味着")
    return not value.startswith(sentence_prefixes)


def extract_concepts(text: str, level: str = "本科入门") -> list[str]:
    try:
        result = chat_json(
            "你是严谨的学术编辑，只返回 JSON。",
            f"读者水平：{level}\n从材料中提取 1~3 个最值得学习的核心技术概念。"
            f"返回 {{\"concepts\":[\"概念\"]}}。\n材料：{text[:6000]}",
        )
        if result and isinstance(result.get("concepts"), list):
            concepts = [str(item).strip() for item in result["concepts"] if _valid_concept(str(item))]
            if concepts:
                return concepts[:3]
    except LLMError:
        pass
    known = []
    normalized = text.replace("—", "-")
    for term in KNOWN_CONCEPTS:
        canonical = term.replace("—", "-")
        if canonical in normalized and canonical not in [item.replace("—", "-") for item in known]:
            known.append(term.replace("—", "-"))
    if known:
        return known[:3]
    candidates = [term for term in TECH_TERM_RE.findall(text) if _valid_concept(term)]
    counts: dict[str, int] = {}
    for term in candidates:
        counts[term] = counts.get(term, 0) + 1
    ranked = sorted(counts, key=lambda term: (counts[term], len(term)), reverse=True)
    if ranked:
        return ranked[:3]
    tokens = [token for token in tokenize(text) if len(token) >= 3]
    return list(dict.fromkeys(tokens))[:2] or [text.strip()[:16] or "新概念"]


ARTICLE_DOMAIN_SIGNALS: list[tuple[str, tuple[str, ...]]] = [
    ("心理学", ("心理", "认知", "情绪", "人格", "行为", "脑", "神经")),
    ("历史与文化", ("历史", "宫廷", "古代", "王朝", "文物", "考古", "传统", "文化遗产")),
    ("文学与语言", ("文学", "诗歌", "小说", "写作", "语言", "叙事", "修辞", "翻译")),
    ("哲学与思想", ("哲学", "认识论", "伦理", "存在", "意识", "思想史", "价值")),
    ("社会学", ("社会", "群体", "阶层", "性别", "人口", "制度", "传播")),
    ("经济与商业", ("经济", "金融", "市场", "企业", "商业", "投资", "消费", "产业")),
    ("计算机与人工智能", ("人工智能", "AI", "算法", "模型", "机器学习", "神经网络", "大模型", "编程")),
    ("电子信息与工程", ("电路", "电子", "芯片", "通信", "控制", "机器人", "嵌入式", "工程")),
    ("生命科学与医学", ("生物", "医学", "疾病", "临床", "细胞", "基因", "药物", "健康")),
    ("数学与自然科学", ("数学", "物理", "化学", "天文", "定理", "方程", "实验", "科学")),
    ("艺术与审美", ("艺术", "审美", "绘画", "音乐", "舞蹈", "电影", "建筑", "设计")),
]


def article_preview_metadata(title: str, text: str, description: str = "") -> dict[str, Any]:
    """Build honest first-glance metadata from an article's fetched body.

    This path is deliberately deterministic: loading an inbox must not spend
    model quota or turn an uncertain card title into a confident taxonomy.
    """
    body = re.sub(r"\s+", " ", text or "").strip()
    card_text = re.sub(r"\s+", " ", f"{title} {description} {body}").strip()
    scored = []
    for domain, signals in ARTICLE_DOMAIN_SIGNALS:
        matched = [term for term in signals if term.lower() in card_text.lower()]
        if matched:
            scored.append((sum(card_text.lower().count(term.lower()) for term in matched), domain, matched))
    scored.sort(key=lambda item: (-item[0], item[1]))
    domain = scored[0][1] if scored else "待读正文判断"

    knowledge: list[str] = []
    for term in KNOWN_CONCEPTS:
        if term.replace("—", "-") in card_text.replace("—", "-"):
            knowledge.append(term.replace("—", "-"))
    if scored:
        knowledge.extend(scored[0][2])
    for term in TECH_TERM_RE.findall(card_text):
        if _valid_concept(term):
            knowledge.append(term)
    knowledge = list(dict.fromkeys(knowledge))[:4]

    supplied_summary = re.sub(r"\s+", " ", description or "").strip()
    if supplied_summary:
        summary = supplied_summary[:220] + ("…" if len(supplied_summary) > 220 else "")
        summary_source = "公众号卡片摘要"
    else:
        sentences = [part.strip() for part in re.split(r"(?<=[。！？!?])", body) if len(part.strip()) >= 18]
        summary = "".join(sentences[:2])[:220]
        if len("".join(sentences[:2])) > 220:
            summary += "…"
        summary_source = "正文开篇提要"
    if not summary:
        summary = "正文已取得，但暂时没有可稳定提取的摘要。"
        summary_source = "待人工确认"

    readable_chars = len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", body))
    reading_minutes = max(1, (readable_chars + 449) // 450) if readable_chars else 0
    return {
        "summary": summary,
        "summary_source": summary_source,
        "domain": domain,
        "knowledge": knowledge,
        "reading_minutes": reading_minutes,
        "body_chars": readable_chars,
        "classification_status": "正文初步识别" if scored else "正文证据不足",
    }


def analyze_material_structure(
    title: str, text: str, level: str = "本科入门", category: str = "",
) -> dict[str, Any]:
    """Read a source before deciding what it is and where it belongs.

    Classification must be supported by excerpts from the material itself.  This
    prevents a chat/group name or a vaguely similar textbook page from deciding
    the article's discipline.
    """
    fallback_discipline, fallback_topic = infer_taxonomy(title, text, category)
    fallback_concepts = extract_concepts(text, level)
    fallback = {
        "main_claim": "",
        "discipline": fallback_discipline,
        "topic": fallback_topic,
        "classification_evidence": [],
        "concepts": fallback_concepts,
        "concept_evidence": {},
        "method": "fallback",
        "confidence": 0.45 if fallback_topic != "待归类的新知" else 0.0,
    }
    if len(re.sub(r"\s+", "", text)) < 100:
        return {**fallback, "method": "insufficient_text", "confidence": 0.0}
    try:
        result = chat_json(
            "你是知识来源理解与分类 Agent。必须先读懂材料的研究对象、核心主张和证据，再决定知识位置；禁止根据公众号名、群名、文件名或偶然关键词归类。只返回 JSON。discipline 是稳定学科主干，topic 是该学科下能容纳材料核心概念的具体分支；不要用文章标题充当分类。提取 1~5 个名词性、可教学的核心概念，不能截取半句话。每个分类和概念都必须给出材料中的短原句作为证据；找不到原句就不要输出。跨学科材料可以选主要学科，同时在 secondary_disciplines 保留其他明确涉及的学科。",
            f"读者水平：{level}\n用户显式分类：{category or '无'}\n标题：{title}\n材料正文：\n{text[:16000]}\n\n"
            "返回 main_claim、discipline、topic、secondary_disciplines、classification_evidence（1~3条正文原句）、concepts（每项含 name、evidence、role；role 为 foundation/mechanism/method/finding/boundary）、confidence（0到1）。",
        )
    except LLMError:
        return fallback
    if not isinstance(result, dict):
        return fallback

    compact_source = re.sub(r"\s+", "", text)

    def quoted(value: Any) -> bool:
        excerpt = re.sub(r"\s+", "", str(value or "").strip().strip('“”\"'))
        return len(excerpt) >= 6 and excerpt[:120] in compact_source

    classification_evidence = [
        str(item).strip() for item in (result.get("classification_evidence") or [])
        if quoted(item)
    ][:3]
    concepts: list[str] = []
    concept_evidence: dict[str, str] = {}
    for item in result.get("concepts") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        evidence = str(item.get("evidence") or "").strip()
        if _valid_concept(name) and quoted(evidence) and name not in concepts:
            concepts.append(name)
            concept_evidence[name] = evidence
        if len(concepts) >= 5:
            break
    try:
        confidence = max(0.0, min(1.0, float(result.get("confidence", 0))))
    except (TypeError, ValueError):
        confidence = 0.0
    discipline = str(result.get("discipline") or "").strip()
    topic = str(result.get("topic") or "").strip()
    if not classification_evidence or not discipline or not topic or confidence < 0.6:
        discipline, topic, confidence = fallback_discipline, fallback_topic, fallback["confidence"]
    return {
        "main_claim": str(result.get("main_claim") or "").strip(),
        "discipline": discipline,
        "topic": topic,
        "secondary_disciplines": [str(item).strip() for item in (result.get("secondary_disciplines") or []) if str(item).strip()][:3],
        "classification_evidence": classification_evidence,
        "concepts": concepts or fallback_concepts,
        "concept_evidence": concept_evidence,
        "method": "langchain" if classification_evidence and concepts else "mixed_fallback",
        "confidence": confidence,
    }


def _fallback_bridge(concept: str, refs: list[dict[str, Any]], level: str) -> dict[str, Any]:
    profile = CONCEPT_PROFILES.get(concept, {})
    if refs:
        ref = refs[0]
        mapping = f"最接近的教材入口是《{ref['title']}》：{ref['snippet']}"
    else:
        mapping = "当前教材库没有直接命中；建议先补充课程大纲或包含该主题的教材章节。"
    return {
        "analogy": profile.get("example") or f"把“{concept}”想成一株嫁接的新枝：它不是凭空出现，而是在熟悉的基础规律上增加了新的约束、尺度或用途。",
        "textbook_mapping": mapping,
        "evolution": profile.get("mechanism") or f"先确认教材中的基本对象和适用条件，再比较“{concept}”改变了哪个假设、增加了哪种能力。",
        "learning_position": "这是一颗“待嫁接种子”：先从命中的教材节点复习定义与假设，再阅读前沿材料中的扩展部分。" if refs else "这是一颗“悬浮种子”：当前知识库缺少可靠前置节点，建议先补入课程大纲或基础章节，再决定学习顺序。",
        "prerequisites": [ref["title"] for ref in refs[:2]],
        "confidence": 0.72 if refs else 0.35,
        "questions": profile.get("questions") or [
            f"如果去掉“{concept}”最关键的新条件，它会退化成教材里的什么？",
            "这项扩展解决了旧方法的哪个失败场景？",
        ],
        "quiz": profile.get("quiz") or {"question": f"理解“{concept}”时最应该先做什么？", "options": ["找到教材中的基础对象与假设", "只背结论", "忽略适用条件", "只看术语翻译"], "answer": 0},
    }


def generate_bridge(
    store: GardenStore, concept: str, frontier_title: str, frontier_text: str, frontier_url: str = "",
    raw_link: str = "",
) -> dict[str, Any]:
    level = str(store.setting("learning_level", "本科入门"))
    refs = search_notes(store, concept + " " + frontier_text[:1200], kinds={"textbook", "course", "concept"}, limit=3)
    context = "\n\n".join(f"[{ref['title']}] {ref['snippet']}" for ref in refs) or "暂无教材命中"
    result: dict[str, Any] | None = None
    try:
        result = chat_json(
            "你是一位循循善诱、拒绝编造引用的大学导师。只返回 JSON。",
            f"学生水平：{level}\n前沿材料：{frontier_title}\n核心概念：{concept}\n"
            f"材料摘录：{frontier_text[:3500]}\n教材检索结果：\n{context}\n"
            "生成个性化教材—前沿对照卡。返回字段 analogy、textbook_mapping、evolution、learning_position、"
            "prerequisites（仅从教材结果选）、confidence（0~1）、questions(2条)、"
            "quiz（含 question、options 四项、answer 为 0-3）。只允许引用给出的教材结果。",
        )
    except LLMError:
        result = None
    payload = _fallback_bridge(concept, refs, level)
    if result:
        for key in ["analogy", "textbook_mapping", "evolution", "learning_position", "prerequisites", "confidence", "questions", "quiz"]:
            if result.get(key):
                payload[key] = result[key]
    explanation = (
        f"## 在你的知识树中的位置\n{payload['learning_position']}\n\n"
        f"**关联置信度：{float(payload['confidence']):.0%}**\n\n"
        f"## 一句话类比\n{payload['analogy']}\n\n"
        f"## 教材映射\n{payload['textbook_mapping']}\n\n"
        f"## 从教材到前沿\n{payload['evolution']}"
    )
    card = {
        "concept": concept,
        "frontier_title": frontier_title,
        "frontier_url": frontier_url,
        "textbook_refs": refs,
        "explanation": explanation,
        "questions": payload["questions"],
        "quiz": payload["quiz"],
    }
    card["id"] = store.add_card(card)
    due = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(timespec="seconds")
    store.add_task(f"回忆：{concept}", "recall", concept, {"card_id": card["id"], "question": payload["questions"][0]}, due, 15)
    quiz_payload = dict(payload["quiz"])
    quiz_payload["card_id"] = card["id"]
    store.add_task(f"小测：{concept}", "quiz", concept, quiz_payload, due, 20)
    store.add_activity("grow_concept", concept, 12)
    vault = store.setting("vault_path", "")
    if vault:
        refs_md = "\n".join(f"- [[{ref['title']}]]" for ref in refs) or "- 暂无直接教材命中"
        body = f"> 来源：{frontier_title}" + (f" · {frontier_url}" if frontier_url else "")
        if raw_link:
            body += f"\n\n原始资料：[[{raw_link}]]"
        body += f"\n\n{explanation}\n\n## 教材节点\n{refs_md}\n\n## 思考题\n" + "\n".join(f"- {q}" for q in payload["questions"])
        try:
            card["obsidian_path"] = str(write_wiki_asset(vault, "02-降维对照", f"{concept}｜教材—前沿对照", body, ["知识花园", "降维对照", concept]))
        except OSError:
            card["obsidian_path"] = ""
    return card


def analyze_frontier(store: GardenStore, title: str, text: str, url: str = "") -> dict[str, Any]:
    if not text.strip():
        raise ValueError("请粘贴需要分析的文章摘要或正文")
    material_title = title or "前沿材料"
    raw_path = None
    vault = store.setting("vault_path", "")
    removed = store.purge_frontier(material_title)
    archived: list[str] = []
    if vault:
        bridge_dir = Path(vault) / "wiki" / "02-降维对照"
        trash_dir = Path(vault) / ".garden-trash" / datetime.now().strftime("%Y%m%d-%H%M%S")
        if bridge_dir.is_dir():
            for path in bridge_dir.glob("*.md"):
                try:
                    existing = path.read_text(encoding="utf-8-sig", errors="replace")
                    if "garden_generated: true" in existing and f"> 来源：{material_title}" in existing:
                        trash_dir.mkdir(parents=True, exist_ok=True)
                        destination = trash_dir / path.name
                        shutil.move(str(path), str(destination))
                        archived.append(str(destination))
                except OSError:
                    continue
    if vault:
        raw_path = write_raw_material(vault, material_title, text, url, ["前沿", "网页输入"])
    concepts = extract_concepts(text, str(store.setting("learning_level", "本科入门")))
    cards = [
        generate_bridge(store, concept, material_title, text, url, raw_path.stem if raw_path else "")
        for concept in concepts
    ]
    discipline, branch = infer_taxonomy(material_title, text)
    return {
        "concepts": concepts, "cards": cards, "raw_path": str(raw_path) if raw_path else "",
        "discipline": discipline, "branch": branch, "removed": removed, "archived": archived,
    }


def add_interest(store: GardenStore, title: str, content: str, tags: list[str]) -> dict[str, Any]:
    if not content.strip():
        raise ValueError("碎片内容不能为空")
    # Interest sparks should connect to stable knowledge assets, never a raw PDF
    # page title or another generated bridge card.
    professional = search_notes(
        store, content, kinds={"course", "frontier", "concept", "knowledge", "domain"}, limit=2,
        strict_relevance=False,
    )
    link_result = None
    if professional:
        target = professional[0]
        shared = sorted(set(tokenize(content)) & set(tokenize(target["title"] + target["snippet"])), key=len, reverse=True)
        metaphor = f"“{title or content[:12]}”与“{target['title']}”都在处理「{'、'.join(shared[:3]) or '结构与变化'}」：一个来自兴趣直觉，一个来自专业语言。"
        questions = ["如果把二者的关键变量互换，会出现什么新解释？", "这条联系在哪些条件下会失效？"]
        evidence = [f"兴趣碎片：{content[:120]}", f"知识节点：{target['snippet'][:160]}"]
        confidence = 0.62 if shared else 0.4
        try:
            generated = chat_json(
                "你是知识园丁，擅长寻找有依据的跨领域同构，只返回 JSON。",
                f"兴趣碎片：{content[:1800]}\n专业节点：{target['title']}\n{target['snippet']}\n"
                "返回 metaphor、questions(2条)、evidence(2条可核查证据)、confidence(0~1)，避免空泛套话。",
            )
            if generated:
                metaphor = str(generated.get("metaphor") or metaphor)
                questions = generated.get("questions") or questions
                evidence = generated.get("evidence") or evidence
                confidence = float(generated.get("confidence") or confidence)
        except LLMError:
            pass
        link_result = {"target": target, "metaphor": metaphor, "questions": questions, "evidence": evidence, "confidence": confidence, "status": "proposed"}
    vault = store.setting("vault_path", "")
    display_title = title or content[:20]
    body = content
    if link_result:
        body += f"\n\n## 知识园丁发现\n{link_result['metaphor']}\n\n关联：[[{link_result['target']['title']}]]\n\n" + "\n".join(f"- {q}" for q in link_result["questions"])
    if vault:
        try:
            output_path = write_wiki_asset(vault, "03-交叉火花", display_title, body, ["知识花园", "交叉火花", *(tags or ["灵感碎片"])])
            note = parse_markdown(output_path, Path(vault).expanduser().resolve())
            note_id, _ = store.upsert_note(note)
            store.replace_wikilinks(note_id, note["wikilinks"])
        except OSError:
            note_id, _ = store.upsert_note({"path": f"capture::{utc_now()}::{display_title}", "title": display_title, "kind": "interest", "content": body, "tags": tags or ["灵感碎片"], "source": "quick_capture", "content_hash": str(hash(body))})
    else:
        note_id, _ = store.upsert_note({"path": f"capture::{utc_now()}::{display_title}", "title": display_title, "kind": "interest", "content": body, "tags": tags or ["灵感碎片"], "source": "quick_capture", "content_hash": str(hash(body))})
    if link_result:
        target = link_result["target"]
        store.add_semantic_link(note_id, target["id"], target["title"], link_result["metaphor"], link_result["confidence"], evidence=link_result["evidence"], status="proposed")
        store.add_task(f"跨界追问：{display_title}", "socratic", target["title"], {"questions": link_result["questions"]}, utc_now(), 12)
    store.add_activity("capture_interest", display_title, 5)
    return {"note_id": note_id, "link": link_result}


def weekly_report(store: GardenStore) -> dict[str, Any]:
    stats = store.stats()
    cards = store.list_cards(5)
    tasks = store.list_tasks(include_done=True, limit=30)
    recent_done = [task for task in tasks if task["status"] == "done"]
    new_concepts = [card["concept"] for card in cards]
    insight = "本周的花园正在扎根。先完成一项到期任务，再为最感兴趣的概念补一条自己的解释。"
    if new_concepts:
        insight = f"本周新枝集中在“{'、'.join(new_concepts[:3])}”。试着用一个共同问题串起它们，知识会从列表长成网络。"
    return {
        "title": f"知识花园周报 · {datetime.now().strftime('%m月%d日')}",
        "stats": stats,
        "new_concepts": new_concepts,
        "completed": len(recent_done),
        "insight": insight,
        "next_actions": [task["title"] for task in store.list_tasks(limit=3)],
    }


def evaluate_review(
    task: dict[str, Any], answer: Any, self_rating: int = 2,
    history: list[dict[str, Any]] | None = None, context: str = "", profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Understand accumulated reasoning first, then decide whether to continue tutoring."""
    payload = task.get("payload", {})
    if task["task_type"] == "quiz" and not history:
        try:
            selected = int(answer)
        except (TypeError, ValueError):
            raise ValueError("请选择一个答案")
        correct_index = int(payload.get("answer", -1))
        options = payload.get("options", [])
        correct = selected == correct_index
        correct_text = options[correct_index] if 0 <= correct_index < len(options) else "参考答案"
        return {
            "quality": 3 if correct else 0,
            "correct": correct,
            "feedback": "你的选择抓住了这一概念的关键入口。" if correct else "我先保留你的选择；它可能反映了一个常见混淆，而不只是‘没记住答案’。",
            "understood": "能够识别关键入口" if correct else "已经作出明确判断，可以继续检查判断依据",
            "followup": "请用一句话说明为什么这个选项成立。" if correct else f"你为什么选择它？请比较它与“{correct_text}”在作用对象或适用条件上的差别。",
            "needs_followup": not correct,
        }
    response = str(answer or "").strip()
    if len(response) < 8:
        raise ValueError("请先写下至少一句自己的回答，再提交复习")
    question = payload.get("question") or "；".join(payload.get("questions", [])) or task["title"]
    fallback_quality = max(0, min(3, int(self_rating)))
    fallback = {
        "quality": fallback_quality,
        "correct": None,
        "understood": "你已经用自己的语言作出了回答，而不是只点击完成。",
        "followup": f"能否给“{task['concept']}”补一个具体例子或失效条件？",
        "needs_followup": fallback_quality < 2,
        "feedback": {
            0: "你已经识别出自己还不会，这是有效复习。明天再从概念定义开始。",
            1: "已有初步印象，但解释还不稳定；下次尝试补上条件和例子。",
            2: "能够用自己的话回答；下次重点验证边界条件。",
            3: "回答流畅且有把握，复习间隔将明显拉长。",
        }[fallback_quality],
    }
    concept_profile = CONCEPT_PROFILES.get(task.get("concept", ""), {})
    keypoints = concept_profile.get("keypoints", [])
    if keypoints:
        hits = [item for item in keypoints if item in response]
        missing = [item for item in keypoints if item not in response]
        if hits:
            fallback["understood"] = f"你的回答已经触及：{'、'.join(hits)}。"
        if missing:
            fallback["followup"] = f"不要求复述术语：请用自己的话说明“{'”与“'.join(missing[:2])}”在你解释中分别起什么作用？"
    clean_history = [
        {"role": str(item.get("role", "")), "content": str(item.get("content", ""))[:1800]}
        for item in (history or [])[-8:] if item.get("role") in {"user", "assistant"}
    ]
    dialogue = "\n".join(f"{item['role']}：{item['content']}" for item in clean_history)
    learner = profile or {}
    try:
        assessed = chat_json(
            "你是理解优先的苏格拉底导师。评价语义和推理，不用关键词命中率判定；允许口语、类比、不完整但方向正确的回答。先复述你理解到的学生观点和正确部分，再指出一个最关键缺口。只有明显矛盾时才纠正，信息不足时用追问澄清，不能简单宣布错误。综合多轮回答判断累计理解。只返回 JSON。",
            f"学习画像：水平={learner.get('learning_level','本科入门')}，兴趣={learner.get('interests',[])}\n"
            f"概念：{task['concept']}\n问题：{question}\n参考知识：{context or concept_profile}\n"
            f"此前互动：\n{dialogue or '无'}\n最新回答：{response}\n"
            "返回 quality（0尚未形成解释/1部分理解/2能解释机制/3能举例迁移）、understood（准确复述学生已经表达对的内容）、feedback（温和补充或纠正）、followup（一次只问一个最能推进理解的问题）、needs_followup（是否需要继续一轮）、correct（null）。quality>=2 时通常可结束本轮，但若核心矛盾未澄清仍可追问。",
        )
        if assessed:
            quality = max(0, min(3, int(assessed.get("quality", fallback_quality))))
            return {
                "quality": quality,
                "correct": assessed.get("correct"),
                "feedback": str(assessed.get("feedback") or fallback["feedback"]),
                "understood": str(assessed.get("understood") or fallback["understood"]),
                "followup": str(assessed.get("followup") or fallback["followup"]),
                "needs_followup": bool(assessed.get("needs_followup", quality < 2)),
            }
    except LLMError:
        pass
    return fallback
