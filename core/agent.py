from __future__ import annotations

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from core.compiler import ingest_raw
from core.config import llm_config
from core.context_builder import ContextBuilder
from core.engine import CONCEPT_PROFILES, infer_taxonomy
from core.gardener_graph import run_gardener_graph
from core.learning_memory import LearningMemoryService
from core.llm import LLMError, chat, chat_json
from core.obsidian import append_backlink, parse_markdown, write_wiki_asset
from core.retrieval import search_notes, tokenize
from core.storage import GardenStore
from core.web_research import search_academic_articles


MANAGED_START = "<!-- knowledge-gardener:agent:start -->"
MANAGED_END = "<!-- knowledge-gardener:agent:end -->"


def _agents_path(vault: Path) -> Path:
    correct = vault / "AGENTS.md"
    legacy = vault / "AGENTS.md.md"
    if correct.is_file():
        return correct
    if legacy.is_file():
        legacy.replace(correct)
        return correct
    return correct


def _source_rows(vault: Path) -> list[str]:
    rows = []
    raw_dir = vault / "raw"
    if not raw_dir.is_dir():
        return rows
    for path in sorted(raw_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        heading = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        title = heading.group(1).strip() if heading else path.stem
        discipline, topic = infer_taxonomy(title, text)
        source_page = vault / "wiki" / "sources" / f"{path.stem}.md"
        concepts = []
        if source_page.is_file():
            source_text = source_page.read_text(encoding="utf-8-sig", errors="replace")
            section = re.search(r"## 核心概念\s*(.*?)(?:\n## |\Z)", source_text, re.DOTALL)
            if section:
                concepts = re.findall(r"\[\[([^\]|]+)", section.group(1))
        concept_text = "；知识点：" + "、".join(f"[[{item}]]" for item in concepts[:6]) if concepts else ""
        rows.append(
            f"- [[sources/{path.stem}]] → [[{discipline}]] / [[{topic}]]{concept_text}"
        )
    return rows


def update_agents_manifest(vault_path: str | Path) -> dict[str, Any]:
    """Preserve authored rules and replace only the garden-managed protocol block."""
    vault = Path(vault_path).expanduser().resolve()
    if not vault.is_dir():
        raise ValueError("Obsidian Vault 路径不存在")
    path = _agents_path(vault)
    original = path.read_text(encoding="utf-8-sig", errors="replace") if path.is_file() else "# AGENTS.md - 知识花园运行规则\n"
    sources = _source_rows(vault)
    managed = f"""{MANAGED_START}

## 7. 知识园丁 Agent 运行协议（自动维护）

### 7.1 Agent 的地位

Agent 不是知识库界面的附属功能，也不是等待点击的内容生成器；它是整个知识花园的**持续控制循环与学习导师**。界面负责呈现和接受选择，Agent 负责观察变化、理解材料、组织知识、发起教学互动、检查理解并把成果写回。

运行闭环统一为：

`观察 raw/ 与网页输入 → 编译有内容的知识页 → 嫁接学科树 → 主动提出下一步 → 通过回答检查理解 → 根据反馈安排复习 → 将新推导写回 Wiki`

### 7.2 主动行为

1. 自动巡检 `raw/`；新增或更新资料不需要再次手动执行 Ingest。
2. 每次编译后同步更新概念底座、主题 MOC、思维导图、对照卡、复习任务和本文件的来源清单。
   对每个新概念先判断其角色（对象、基础概念、组成、机制、方法、应用或同级概念），再选择父节点；结构判断通过 LangChain 持久化为 `contains` 关系，后续显示不得临时乱排。
3. 首页必须给出一条基于当前知识状态的观察、一项下一步行动和一道可以立即回答的问题。
4. 查询优先依据 `wiki/`，回答必须标明使用了哪些知识页；Wiki 与 raw 均不足且已配置理解 API 时，主动联网检索可信资料，区分本地证据与在线证据，并给出可访问来源。
5. 对话中形成的新解释、反例或研究 Idea，经用户确认后写回 `wiki/03-交叉火花/`。
6. 每次联网探索后都必须邀请用户选择“加入知识库”或“继续讨论”；未经确认不得把临时回答冒充已掌握知识。
7. 园丁问答必须保持多轮上下文；“继续想一步”必须真正进入下一轮，不得只展示一句无法回答的问题。用户确认沉淀后，同时生成对话资产与可教学概念页，并同步 MOC 与思维导图。
8. 学习画像必须影响可见结果：讲解深度、类比场景、复习追问和每日推荐都读取学习水平与兴趣；首页每日推送应说明“为什么推荐给你”，并允许讨论或确认加入知识库。

### 7.3 知识资产质量门槛

- 禁止只生成“等待后续充实”的空占位页。
- 概念页至少包含：核心定义、作用机制、来源证据、具体例子、适用边界、相关概念、主动回忆题和苏格拉底追问。
- 思维导图主图只放“学科 → 分支 → 方向 → 已学知识点”；问题、任务、讲义名和 PDF 页码只能进入详情与证据层。
- 新资料和 Obsidian 修改都必须重新核对受影响分支的层级；中英文别名与括号译名合并显示但不删除原笔记。只有存在“更基础、确实包含或明确前置”的证据时才建立父子关系，同级、对比和证据不足的概念不得强行串联，转入待归类或语义连接。
- 自动关联必须去重，并说明关联类型与依据；不能只显示“知识结构连接”。
- 复习不能一键完成，必须经过选择、解释、举例或边界判断中的至少一种，并给出反馈与下一次复习时间。
- 复习评价以语义和累计推理为准，不得以关键词漏写直接否定。先准确复述用户已经表达正确的部分，再一次追问一个关键缺口；理解尚未形成时保持任务开放，达到掌握标准后才结算经验并安排间隔。

### 7.4 人与 Agent 的分工

- Agent 决定：何时整理、如何检索、怎样提出候选连接、下一题问什么、哪些内容需要复习。
- 用户决定：连接是否成立、解释是否可信、哪些方向值得继续、哪些成果可以沉淀。
- 当证据不足时，Agent 应暴露不确定性并提出验证方案，不得用流畅文案代替理解。

### 7.5 TraceMemo 微信读取协议

- TraceMemo 是仅监听本机的 Local HTTP API，不是 MCP Server；知识园丁通过受限只读工具调用。
- 微信入口默认定位为“公众号文章收件箱”：优先识别 `isOfficialAccount`/`gh_` 来源，解析文章标题、摘要、发布时间、公众号名与原文链接；群聊和联系人折叠为辅助来源。
 - 公众号正文仅在用户点击“读取正文并互动导读”或“确认沉淀”后回源，并只允许 `mp.weixin.qq.com`；导读回源失败时明确降级为文章卡片摘要，不得声称已读正文；确认沉淀若没有取得正文则保持候选状态，不得仅凭标题或摘要自动归类。
- 群聊读取默认过滤入群、撤回、改群名等系统通知，界面不得用内部 MD5、wxid 替代已有的可读昵称。
- 只有用户明确要求查询自己的微信、指定联系人或群聊历史时才允许挂载；普通知识问答、兴趣推送和后台巡视不得扫描微信。
- 每次读取前检查 `/health`；相对时间先查 `/current_time`；先用 `/resolve` 明确会话，再用 `/chatlog` 读取最小必要时间范围与上下文。
- Token 只从环境变量或 Windows DPAPI 加密文件加载，禁止写入 URL、日志、回答、Wiki、AGENTS.md 或仓库。
- 微信片段只能证明“谁在何时表达过什么”，不能把聊天中的说法自动当作客观事实；回答中必须标为授权微信依据并显示适用边界。
- 读取结果默认不进入画像、掌握度、知识图谱或 L2/L3。只有用户审核确认的片段才进入 L1 候选；进一步事实核验通过后才可编译为正式知识资产。

### 7.6 灵感跃迁与领域概览协议

- 灵感中的推测、想象与待核验内容只进入 L1；用户可选择“建立中立领域概览”或“只核验当前假设”，系统不得替用户二选一。
- 用户第一次系统接触一个领域时，首个正式交付物是 800~1500 字、约 1000 字的准确、完整、中立领域概览，而不是个性化路线。正文固定覆盖：一句话定位、根本问题、核心框架、3~5 个发展节点、必要的边界澄清、3~4 个带理由的可选入口。
- 概览开头列主要来源与生成日期；核心定义、框架和每个历史节点分别绑定可追溯脚注；结尾标注资料截止日期并提示 6 个月后可复审。找不到权威依据时使用标准未核验说明，禁止模型常识伪装成检索结果。
- 首次概览正文不得主动提及用户专业、兴趣、旧笔记、掌握度，不得强行类比或预设学习路径。只有用户读完后主动追问、要求比较、表示兴趣或指定本地知识时，才进入第二层个性化深入。
- 横纵分析只作为压缩框架：纵向保留能解释认知演进的关键改变，横向只比较真正的邻近领域、替代解释或同类对象；不得把概览写成流水年表或冗长竞品报告。
- 教材只有在章节、正文、案例或公式直接覆盖当前问题时才是强相关依据。同属一个学科或只出现关键词都不够；弱相关教材停止扩搜并转向外部权威教材、综述、论文、官方机构或经交叉验证的百科。
- 用户导入新教材后，按每本教材的实际目录和代表页重新生成“学科 → 分支 → 方向 → 已学知识点”，每一级都保存正文证据。教材名、章节号、页码和问题不得成为主图节点。
- 微信公众号文章只有回源取得正文后才能抽取概念和自动分类；学科位置必须附正文证据。普通群聊只作为个人证据写入 raw，不自动生成事实概念或污染知识图谱。

## 8. 当前已编译来源（自动维护）

{chr(10).join(sources) if sources else '- 暂无已编译来源'}

{MANAGED_END}"""
    pattern = re.compile(re.escape(MANAGED_START) + r".*?" + re.escape(MANAGED_END), re.DOTALL)
    if pattern.search(original):
        updated = pattern.sub(managed, original)
    else:
        updated = original.rstrip() + "\n\n" + managed + "\n"
    changed = updated != original
    if changed:
        path.write_text(updated, encoding="utf-8")
    return {"path": str(path), "changed": changed, "sources": len(sources)}


def patrol_vault(vault_path: str | Path, store: GardenStore) -> dict[str, Any]:
    """Compile changed raw notes and refresh the Agent manifest."""
    vault = Path(vault_path).expanduser().resolve()
    raw_dir = vault / "raw"
    previous = store.setting("agent_ingest_hashes", {}) or {}
    current: dict[str, str] = {}
    ingested = []
    errors = []
    if raw_dir.is_dir():
        for path in sorted(raw_dir.glob("*.md")):
            relative = str(path.relative_to(vault)).replace("\\", "/")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            current[relative] = digest
            if previous.get(relative) == digest:
                continue
            try:
                result = ingest_raw(vault, relative, store)
                ingested.append({"source": relative, "topic": result["topic"], "concepts": result["concepts"]})
            except Exception as exc:
                errors.append({"source": relative, "error": str(exc)})
                current[relative] = previous.get(relative, "")
    store.set_setting("agent_ingest_hashes", current)
    manifest = update_agents_manifest(vault)
    if ingested:
        store.add_activity("agent_patrol", f"主动编译 {len(ingested)} 份 raw 资料", min(30, len(ingested) * 5))
    return {"ingested": ingested, "errors": errors, "manifest": manifest}


def briefing(store: GardenStore) -> dict[str, Any]:
    interests = store.setting("interests", []) or []
    interest_hint = str(interests[0]) if interests else "你的日常经验"
    tasks = store.list_tasks(limit=20)
    cards = store.list_cards(limit=3)
    task = next((item for item in tasks if item["status"] == "pending"), None)
    if task:
        payload = task.get("payload", {})
        question = payload.get("question") or f"请用自己的话解释“{task['concept']}”，并尝试用“{interest_hint}”举例或给出边界。"
        observation = f"你的花园里已经有关于“{task['concept']}”的内容，但它还没有经过这轮主动回忆。"
        action = "先回答下面的问题；园丁会根据答案决定是加深、提示还是延长复习间隔。"
    elif cards:
        task = None
        question = f"“{cards[0]['concept']}”和你已有的哪个知识点最容易混淆？"
        observation = f"最近的新枝是“{cards[0]['concept']}”，目前没有待完成复习。"
        action = "可以向园丁提问，或补充一个反例让这张卡继续生长。"
    else:
        question = "你最近遇到的哪个概念，看似懂了却还无法用自己的话解释？"
        observation = "花园还缺少可教学的知识资产。"
        action = "放入一篇资料或同步 Obsidian，园丁会先建立第一条学习路径。"
    return {
        "role": "持续观察、主动教学、根据反馈改写知识结构",
        "status": "巡视中",
        "observation": observation,
        "action": action,
        "question": question,
        "task": task,
    }


INTEREST_SEARCH_TERMS = {
    "音乐": "music cognition emotion learning", "古典诗词": "classical Chinese poetry cognition aesthetics",
    "电子电路": "electronic circuits emerging technology", "数学": "mathematics education applications",
    "AI": "artificial intelligence learning cognitive science", "心理学": "psychology recent research",
    "物理": "physics recent research applications", "摄影": "photography visual perception cognition",
}


def _frontier_profile(
    store: GardenStore, interests: list[str], knowledge_notes: list[dict[str, Any]],
) -> dict[str, Any]:
    explicit_text = str(store.setting("frontier_focus", "") or "").strip()
    explicit = [
        item.strip() for item in re.split(r"[，,、;；\n]+", explicit_text) if item.strip()
    ][:8]
    ignored_tags = {
        "教材", "课本", "概念", "知识", "前沿", "待阅读", "每日推荐", "用户确认",
        "视频解析", "字幕已解析", "B站", "raw", "source",
    }
    tag_counts: dict[str, int] = {}
    for note in knowledge_notes[:100]:
        for raw_tag in note.get("tags") or []:
            tag = str(raw_tag).strip()
            if len(tag) < 2 or tag in ignored_tags:
                continue
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    recent_topics = [
        tag for tag, _ in sorted(tag_counts.items(), key=lambda item: (-item[1], item[0]))[:4]
        if tag not in explicit and tag not in interests
    ]
    priorities = list(dict.fromkeys([*explicit, *interests, *recent_topics]))
    return {
        "explicit": explicit,
        "interests": interests,
        "recent_topics": recent_topics,
        "priorities": priorities,
        "basis": [
            *(["你主动填写的专业/当前重点：" + "、".join(explicit)] if explicit else []),
            *(["兴趣画像：" + "、".join(interests)] if interests else []),
            *(["近期知识树主题：" + "、".join(recent_topics)] if recent_topics else []),
        ],
    }


def daily_digest(store: GardenStore, force: bool = False) -> dict[str, Any]:
    """Create a vertical-domain frontier feed with an interactive reading guide."""
    interests = [str(item).strip() for item in (store.setting("interests", []) or []) if str(item).strip()]
    level = str(store.setting("learning_level", "本科入门"))
    today = date.today().isoformat()
    cache_key = "daily_digest"
    knowledge_notes = [
        note for note in store.list_notes(limit=500)
        if note["kind"] in {"concept", "knowledge", "course", "moc"}
    ]
    frontier_profile = _frontier_profile(store, interests, knowledge_notes)
    priorities = frontier_profile["priorities"]
    knowledge_signature = "|".join(f"{note['id']}:{note['updated_at']}" for note in knowledge_notes[:80])
    profile_signature = hashlib.sha256(
        (today + level + "|".join(priorities) + knowledge_signature).encode("utf-8")
    ).hexdigest()
    cached = store.setting(cache_key, {}) or {}
    # Never pin an empty transient failure for the whole day. Successful
    # recommendations may use the daily profile cache normally.
    if not force and cached.get("signature") == profile_signature and cached.get("items"):
        return cached
    if not priorities:
        result = {
            "signature": profile_signature, "date": today, "level": level, "interests": [], "items": [],
            "profile": frontier_profile,
            "message": "先在学习画像里填写专业/当前重点或兴趣，园丁才知道今天该巡视哪些方向。",
        }
        store.set_setting(cache_key, result)
        return result
    refresh_index = int(store.setting("daily_refresh_index", 0) or 0)
    if force:
        refresh_index += 1
        store.set_setting("daily_refresh_index", refresh_index)
    offset = (date.today().toordinal() + refresh_index) % len(priorities)
    chosen = [priorities[(offset + index) % len(priorities)] for index in range(min(2, len(priorities)))]
    articles = []
    retrieval_reports: list[dict[str, Any]] = []
    retrieval_errors: list[str] = []
    def retrieve_direction(interest: str) -> tuple[str, str, list[dict[str, Any]], dict[str, Any]]:
        query = INTEREST_SEARCH_TERMS.get(interest, f"{interest} recent advances research")
        diagnostics: dict[str, Any] = {}
        try:
            found = search_academic_articles(
                query, limit=6, from_publication_date=(date.today() - timedelta(days=45)).isoformat(),
                diagnostics=diagnostics, timeout=7, attempts_per_provider=1,
            )
        except Exception as exc:
            found = []
            if not diagnostics.get("errors"):
                diagnostics["errors"] = [f"{exc.__class__.__name__}：{exc}"]
        diagnostics["interest"] = interest
        diagnostics["query"] = query
        return interest, query, found, diagnostics

    # Two independent subject searches run together so a degraded provider
    # cannot make the user wait through both directions sequentially.
    with ThreadPoolExecutor(max_workers=min(2, len(chosen))) as executor:
        direction_results = list(executor.map(retrieve_direction, chosen))

    for interest, query, found, diagnostics in direction_results:
        retrieval_reports.append(diagnostics)
        retrieval_errors.extend(str(item) for item in diagnostics.get("errors", []) if str(item).strip())
        added_for_interest = 0
        for article in found:
            title_key = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(article.get("title") or "").lower())
            if any(
                item.get("url") == article.get("url")
                or re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(item.get("title") or "").lower()) == title_key
                for item in articles
            ):
                continue
            article = dict(article)
            article["interest"] = interest
            articles.append(article)
            added_for_interest += 1
            if added_for_interest >= 2:
                break
        if len(articles) >= 4:
            break
    read_urls = set(store.setting("frontier_read_urls", []) or [])
    ranked = []
    for article in articles:
        article_text = f"{article.get('title','')} {article.get('abstract','')}"
        article_tokens = set(tokenize(article_text))
        connections = []
        for note in knowledge_notes:
            shared = sorted(article_tokens & set(tokenize(note["title"] + " " + note["content"][:1200])), key=len, reverse=True)
            meaningful = [token for token in shared if len(token) >= 3]
            if meaningful:
                connections.append({"id": note["id"], "title": note["title"], "terms": meaningful[:3]})
        connections = connections[:3]
        if article["interest"] in frontier_profile["explicit"]:
            field_score = 1.0
        elif article["interest"] in interests:
            field_score = 0.9
        else:
            field_score = 0.76
        connection_score = min(1.0, 0.35 * len(connections))
        authority_score = min(0.95, 0.62 + min(int(article.get("cited_by_count") or 0), 100) / 500)
        freshness_score = 1.0 if int(article.get("year") or 0) >= date.today().year else 0.72
        total = 0.35 * field_score + 0.3 * connection_score + 0.2 * authority_score + 0.15 * freshness_score
        ranked.append((total, article, connections, {
            "domain_match": round(field_score, 2), "knowledge_connection": round(connection_score, 2),
            "source_quality": round(authority_score, 2), "freshness": round(freshness_score, 2),
        }))
    ranked.sort(key=lambda row: row[0], reverse=True)
    items = []
    for total, article, connections, scores in ranked[:4]:
        connection_titles = [item["title"] for item in connections]
        connection = f"它来自你选择的“{article['interest']}”垂类方向"
        if connection_titles:
            connection += "，并与知识树中的" + "、".join(f"《{title}》" for title in connection_titles) + "出现概念重叠"
        else:
            connection += "；当前属于知识树外缘的新种子，建议先判断是否值得嫁接"
        abstract = str(article.get("abstract") or "").strip()
        sentences = [part.strip() for part in re.split(r"(?<=[。！？.!?])\s+", abstract) if part.strip()]
        preview = sentences[0][:220] if sentences else "先从标题判断研究对象、变量与可能结论。"
        guide = {
            "before_reading": f"读前预测：仅看标题，你认为它会怎样连接“{connection_titles[0] if connection_titles else article['interest']}”？",
            "orientation": preview,
            "checkpoints": [
                "作者真正研究的对象、变量和比较基准分别是什么？",
                "摘要给出的是相关、因果，还是一种理论解释？",
                f"它与《{connection_titles[0]}》的连接是前置、扩展还是反例？" if connection_titles else "它应该嫁接到你知识树的哪个学科分支？",
            ],
            "after_reading": "请用两句话复述：一句写核心结论，一句写适用边界或仍未解决的问题。",
        }
        items.append({
            **{key: article.get(key) for key in ("title", "url", "year", "authors", "venue", "source", "abstract", "publication_date", "open_access", "pdf_url")},
            "interest": article["interest"], "why": connection + f"；导读按“{level}”难度组织。",
            "connections": connections, "scores": scores, "rank_score": round(total, 3),
            "reading_guide": guide, "read": article.get("url") in read_urls,
            "prompt": f"请带我导读《{article.get('title','')}》。先让我回答读前预测，再逐个检查研究对象、证据类型、与本地知识的关系和适用边界；不要一次把答案全部讲完。",
        })
    providers = list(dict.fromkeys(
        str(item.get("provider")) for item in retrieval_reports if item.get("provider")
    ))
    degraded = any(bool(item.get("degraded")) for item in retrieval_reports)
    if items and degraded:
        notice = f"主检索源暂时不可用，园丁已自动切换到{'、'.join(providers) or '备用学术源'}。"
    elif items:
        notice = f"今日推荐来自{'、'.join(providers) or '在线学术源'}。"
    else:
        notice = "；".join(dict.fromkeys(retrieval_errors)) or "在线来源没有返回符合当前兴趣和时间范围的文章。"
    result = {
        "signature": profile_signature, "date": today, "level": level, "interests": interests,
        "profile": frontier_profile, "chosen_directions": chosen,
        "items": items,
        "message": "" if items else f"今天没有生成推荐：{notice}",
        "notice": notice,
        "retrieval": {"providers": providers, "degraded": degraded, "errors": list(dict.fromkeys(retrieval_errors)), "reports": retrieval_reports},
    }
    if items:
        store.set_setting(cache_key, result)
    return result


def _legacy_answer_from_wiki(store: GardenStore, question: str, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    question = question.strip()
    if not question:
        raise ValueError("请先写下你想问园丁的问题")
    clean_history = []
    for item in (history or [])[-10:]:
        role = str(item.get("role", ""))
        content = str(item.get("content", "")).strip()[:2500]
        if role in {"user", "assistant"} and content:
            clean_history.append({"role": role, "content": content})
    dialogue = "\n".join(
        f"{'用户' if item['role'] == 'user' else '园丁'}：{item['content']}" for item in clean_history
    )
    first_question = next((item["content"] for item in clean_history if item["role"] == "user"), question)
    retrieval_question = f"{first_question}\n{question}" if clean_history else question
    hits = search_notes(
        store, retrieval_question, kinds={"concept", "moc", "bridge", "knowledge", "course"}, limit=6
    )
    evidence_layer = "wiki"
    if hits:
        peak = float(hits[0].get("score", 0))
        hits = [item for item in hits if float(item.get("score", 0)) >= peak * 0.5][:3]
    if not hits:
        hits = search_notes(store, question, kinds={"spark", "raw", "frontier"}, limit=3)
        evidence_layer = "raw"
    local_peak = float(hits[0].get("score", 0)) if hits else 0.0
    context = "\n\n".join(f"[{item['title']}] {item['snippet']}" for item in hits)
    needs_web = not hits or local_peak < 1.0
    search_query = retrieval_question
    if hits:
        try:
            plan = chat_json(
                "你是研究检索规划器。判断本地片段能否解释用户问题的核心机制。输出 JSON：needs_web 布尔值、search_query 英文学术检索词。若片段只碰巧共享词语，必须联网。",
                f"对话：\n{dialogue or '无'}\n\n最新问题：{question}\n\n本地片段：\n{context}",
            )
            if plan:
                needs_web = bool(plan.get("needs_web", needs_web))
                search_query = str(plan.get("search_query") or question).strip()
        except LLMError:
            pass
    articles: list[dict[str, Any]] = []
    research_error = ""
    if needs_web and llm_config().enabled:
        try:
            articles = search_academic_articles(search_query, limit=4)
        except Exception as exc:
            research_error = exc.__class__.__name__
    online_context = "\n\n".join(
        f"[O{index}] {item['title']} ({item.get('year') or 'n.d.'})\n{item.get('abstract') or '仅检索到题录，回答中不可推断其具体结论。'}"
        for index, item in enumerate(articles, 1)
    )
    answer = None
    followup = "这个解释中，哪一步最需要一个反例或亲身经验来检验？"
    discussion_prompts = ["这个机制有哪些适用边界？", "它与我已有的哪个知识点可以建立连接？"]
    learning_level = str(store.setting("learning_level", "本科入门"))
    interests = store.setting("interests", []) or []
    try:
        result = chat_json(
            "你是持续对话的主动式知识园丁。先理解用户最新一句是在回答、质疑、举例还是追问；必须承接前文，先指出其中合理或有启发的部分，再补充、纠正或追问，不能每轮重新讲一遍。综合本地知识和在线论文摘要；澄清隐喻或争议前提，解释机制、举例并指出边界。在线论断必须用 [O1] 标注；本地证据不足时明确说它只提供邻近概念。输出 JSON：answer（Markdown）、followup、discussion_prompts（2个递进问题）。不得虚构论文结论。",
            f"学习水平：{learning_level}\n兴趣：{'、'.join(map(str, interests)) or '未设置'}\n\n已有对话：\n{dialogue or '这是第一轮'}\n\n最新消息：{question}\n\n本地知识：\n{context or '无'}\n\n在线检索：\n{online_context or '无'}",
        )
        if result:
            answer = str(result.get("answer") or "").strip()
            followup = str(result.get("followup") or followup).strip()
            prompts = result.get("discussion_prompts")
            if isinstance(prompts, list):
                discussion_prompts = [str(item).strip() for item in prompts if str(item).strip()][:2] or discussion_prompts
    except LLMError:
        answer = None
    if not answer:
        if hits:
            lead = hits[0]
            answer = f"本地最接近的是《{lead['title']}》：{lead['snippet']}\n\n目前无法形成可靠的模型综合回答。"
        else:
            answer = "当前本地知识与在线检索都没有形成足够证据。可以换一组关键词，或先加入一篇可信材料再继续。"
    store.add_activity("agent_query", question[:80], 2)
    return {
        "answer": answer,
        "citations": [{"id": item["id"], "title": item["title"], "path": item["path"]} for item in hits],
        "web_sources": [{key: item.get(key) for key in ("title", "url", "year", "authors", "venue", "source")} for item in articles],
        "followup": followup,
        "discussion_prompts": discussion_prompts,
        "evidence_layer": "wiki + online" if articles and hits else "online" if articles else evidence_layer if hits else "none",
        "researched_online": bool(articles),
        "research_error": research_error,
        "offer_save": True,
    }


def answer_from_wiki(
    store: GardenStore,
    question: str,
    history: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Run the inspectable LangGraph gardener workflow.

    The legacy implementation remains in this module temporarily for rollback
    during the migration, but all normal calls now enter the graph.
    """
    memory = LearningMemoryService(store)
    turn = memory.begin_turn(question, session_id)
    context = ContextBuilder(store).build(
        question,
        history,
        session_id=turn["session_id"],
        request_id=turn["request_id"],
        message_id=turn["message_id"],
    )
    result = run_gardener_graph(store, context)
    persisted = memory.complete_turn(context, result)
    result["session_id"] = context.session_id
    result["request_id"] = context.request_id
    result["memory_update"] = persisted
    return result


def save_agent_insight(
    store: GardenStore, question: str, answer: str, citations: list[dict[str, Any]],
    web_sources: list[dict[str, Any]], followup: str = "", messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    vault_value = store.setting("vault_path", "")
    if not vault_value:
        raise ValueError("请先在设置中连接 Obsidian Vault")
    vault = Path(vault_value).expanduser().resolve()
    local_links = [f"[[{item['title']}]]" for item in citations if item.get("title")]
    source_lines = []
    for item in web_sources:
        label = item.get("title") or "在线来源"
        meta = " · ".join(str(value) for value in (item.get("year"), item.get("venue")) if value)
        source_lines.append(f"- [{label}]({item.get('url', '')})" + (f" — {meta}" if meta else ""))
    transcript = "\n\n".join(
        f"**{'你' if item.get('role') == 'user' else '园丁'}**：{str(item.get('content', '')).strip()}"
        for item in (messages or []) if item.get("role") in {"user", "assistant"} and str(item.get("content", "")).strip()
    )
    body = (
        f"> 用户最初的问题：{question}\n\n## 对话推导\n\n{transcript or answer.strip()}\n\n"
        f"## 当前结论\n\n{answer.strip()}\n\n"
        f"## 本地知识连接\n\n{chr(10).join(f'- {link}' for link in local_links) if local_links else '- 暂无已确认本地连接'}\n\n"
        f"## 在线来源\n\n{chr(10).join(source_lines) if source_lines else '- 本次未使用在线来源'}\n\n"
        f"## 下一步追问\n\n- {followup or '这个解释最需要怎样的反例来检验？'}"
    )
    title = f"{question[:54]}｜园丁探索"
    output = write_wiki_asset(vault, "03-交叉火花", title, body, ["知识花园", "交叉火花", "园丁对话"])
    note = parse_markdown(output, vault)
    note_id, _ = store.upsert_note(note)
    store.replace_wikilinks(note_id, note["wikilinks"])
    for citation in citations:
        relative = str(citation.get("path") or "")
        source_path = vault / relative
        if source_path.is_file():
            append_backlink(source_path, title)

    moc_candidates: list[str] = []
    citation_ids = [int(item["id"]) for item in citations if str(item.get("id", "")).isdigit()]
    if citation_ids:
        marks = ",".join("?" for _ in citation_ids)
        with store.connect() as conn:
            rows = conn.execute(
                f"""SELECT DISTINCT m.title FROM notes m JOIN links l
                    ON ((l.source_id=m.id AND l.target_id IN ({marks})) OR (l.target_id=m.id AND l.source_id IN ({marks})))
                    WHERE m.kind='moc' ORDER BY m.title""",
                (*citation_ids, *citation_ids),
            ).fetchall()
        moc_candidates = [row["title"] for row in rows]
    distilled = None
    try:
        distilled = chat_json(
            "你是知识资产编辑。把已确认的多轮对话提炼为一个可教学、可修订的知识点。标题必须是概念或机制名，不能使用问句；MOC 只能从候选列表选择，无法判断则为空。只返回 JSON。",
            f"原问题：{question}\n对话：\n{transcript or answer}\n候选 MOC：{moc_candidates}\n"
            "返回 title、definition、mechanism、boundary、moc。",
        )
    except LLMError:
        distilled = None
    concept_title = str((distilled or {}).get("title") or re.sub(r"[？?]$", "", question)).strip()[:54]
    if concept_title.startswith("为什么"):
        concept_title = concept_title[3:] + "的作用机制"
    definition = str((distilled or {}).get("definition") or answer.strip()[:500])
    mechanism = str((distilled or {}).get("mechanism") or "由本次对话形成，仍需结合更多来源与反例继续修订。")
    boundary = str((distilled or {}).get("boundary") or "这是一次对话沉淀出的当前理解，不等于已经被单一来源最终证明。")
    chosen_moc = str((distilled or {}).get("moc") or "")
    if chosen_moc not in moc_candidates:
        chosen_moc = moc_candidates[0] if moc_candidates else ""
    concept_body = (
        f"> **由园丁对话确认沉淀，可继续修订。**\n\n## 核心定义\n\n{definition}\n\n"
        f"## 作用机制\n\n{mechanism}\n\n## 适用边界\n\n{boundary}\n\n"
        f"## 推导来源\n\n- [[{title}]]\n" + (f"- [[{chosen_moc}]]\n" if chosen_moc else "") +
        "\n## 主动回忆\n\n- 请不用原句重新解释这一机制，并举一个可能失败的例子。"
    )
    concept_path = write_wiki_asset(
        vault, "01-概念底座", concept_title, concept_body,
        ["概念底座", "园丁对话", *([chosen_moc] if chosen_moc else [])],
    )
    append_backlink(output, concept_title)
    if chosen_moc:
        moc_path = vault / "wiki" / "04-主题索引" / f"{chosen_moc}.md"
        if moc_path.is_file():
            append_backlink(moc_path, concept_title)
    store.resolve_links()
    store.add_activity("agent_writeback", title, 8)
    return {
        "path": str(output), "title": title, "note_id": note_id,
        "concept_path": str(concept_path), "concept_title": concept_title, "moc": chosen_moc,
    }


def hint_for_task(store: GardenStore, task_id: int) -> dict[str, str]:
    task = store.get_task(task_id)
    if not task:
        raise ValueError("复习任务不存在")
    profile = CONCEPT_PROFILES.get(task.get("concept", ""), {})
    mechanism = profile.get("mechanism")
    if mechanism:
        first = re.split(r"[；。]", mechanism)[0]
        hint = f"先画出变量之间的方向：{first}。不要急着复述完整定义。"
    elif task["task_type"] == "quiz":
        hint = "先排除忽略对象、适用条件或证据来源的选项，再判断哪个选项描述了真正的作用路径。"
    else:
        hint = f"先回答三个词：‘{task['concept']}’作用于什么、在什么条件下、产生什么结果？"
    return {"hint": hint}
