from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from core.config import closed_loop_generation_timeout_seconds
from core.learning_memory import LearningMemoryService
from core.llm import LLMError, chat, chat_json
from core.obsidian import parse_markdown, write_wiki_asset
from core.reasoning_capability import (
    classify_reasoning_task,
    is_self_contained_reasoning,
    reasoning_prompt,
    review_reasoning_answer,
    science_precision_instruction,
)
from core.retrieval import STOPWORDS, relevance_gate, search_notes
from core.storage import GardenStore, utc_now


def _fallback_type(text: str) -> tuple[str, bool]:
    if re.search(r"如果|假如|假设|会怎样", text):
        return "counterfactual", bool(re.search(r"人类|大脑|物理|生物|心理|社会|算法", text))
    if re.search(r"是不是|有没有可能|能否.*解释|跨学科|像不像", text):
        return "cross_disciplinary_hypothesis", True
    if re.search(r"我感觉|我总觉得|直觉", text):
        return "intuition_record", True
    return "open_exploration", True


_CONVERSATIONAL_ANCHOR_TERMS = STOPWORDS | {
    "有人", "别人", "自己", "喜欢", "这么", "那么", "这样", "那样", "一直",
    "需要", "更多", "东西", "想法", "时候", "情况", "这种", "一种", "某种",
}


def _directly_relevant_anchor(question: str, item: dict[str, Any]) -> bool:
    """Require a real subject match, not shared conversational filler."""
    review = relevance_gate(
        question, str(item.get("title") or ""), str(item.get("snippet") or ""),
    )
    if not review["passed"]:
        return False
    title = re.sub(r"\s+", "", str(item.get("title") or "")).casefold()
    meaningful = list(dict.fromkeys(
        str(term).strip().casefold()
        for term in review.get("matched_terms", [])
        if str(term).strip()
        and str(term).strip() not in _CONVERSATIONAL_ANCHOR_TERMS
        and not re.match(r"^(?:么|这么|那么|这样|那样)", str(term).strip())
    ))
    return bool(
        any(len(term) >= 4 for term in meaningful)
        or any(len(term) >= 2 and term in title for term in meaningful)
        or len([term for term in meaningful if len(term) >= 3]) >= 2
    )


def _fallback_inspiration_answer(message: str, kind: str) -> str:
    focus = message.strip(" ：:，,。？！? ")[:140]
    opening = (
        "这个设想最有意思的地方，不是立刻判断它能不能实现，而是看它会改变哪些默认前提。"
        if kind == "counterfactual" else
        "你注意到的可能不是一个孤立现象，而是不同动机、环境和个人经验叠在一起的结果。"
    )
    return "\n\n".join([
        f"{opening}你真正想追问的是：{focus}。",
        "可以先把表面表现与背后的需要分开：当事人想获得什么，又在避免什么？"
        "同一种举动既可能来自好奇、认同和安全感，也可能受到评价、稀缺感或群体氛围推动；"
        "不能只凭一个现象就替所有人规定同一种心理。",
        "再换个方向想：如果拿掉外部赞许、身份象征或模仿对象，这个现象还会不会出现？"
        "这样的反例能帮助区分真正的内在需求，还是环境把某种行为放大了。",
        "这些都是探索假设，不是已经证实的结论。最值得继续挖的，是找到一个具体情境，"
        "比较不同解释各自能说明什么、又在哪些地方说不通。",
    ])


_META_REFUSAL_LANGUAGE = re.compile(
    r"这次先不{1,2}(?:补写|写)?答案|这次先不回答|"
    r"(?:^|[\n。！？])\s*(?:我|本次|当前)(?:先|暂时)?不(?:直接)?(?:回答|作答|给出答案)|"
    r"(?:^|\n)\s*(?:先|暂时)不(?:直接)?(?:回答|作答|给出答案)|"
    r"(?:^|[\n。！？])\s*(?:抱歉[，,]?\s*)?(?:我|本次|当前|现阶段)?\s*"
    r"(?:不能|无法|不方便)(?:直接)?(?:回答|作答|给出答案)(?:这个|该)?(?:问题)?|"
    r"(?:^|[\n。！？])\s*(?:这个|该)?问题\s*(?:我|本次|当前)?\s*"
    r"(?:不能|无法|不方便)(?:直接)?(?:回答|作答|给出答案)|"
    r"当前无法(?:给出)?(?:答案|回答)|请(?:重新提问|重试本题|稍后再试|换个问题)|"
    r"证据不足所以不回答|没有取得足够.{0,20}证据.{0,30}(?:不回答|补充.*来源)",
    re.I | re.S,
)

_AUDITABLE_CODE_REQUEST = re.compile(
    r"Python|SymPy|Z3|pgmpy|RDKit|Qiskit|COBRApy|代码|脚本|编程|"
    r"数值积分器|求解器|算法实现|编写.{0,12}(?:程序|算法)|实现.{0,12}(?:算法|模拟|求解|验证)",
    re.I,
)


def _decode_document_newlines(text: str) -> str:
    """Decode model-escaped document newlines without corrupting LaTeX \nabla."""
    value = str(text or "")
    looks_like_escaped_document = bool(re.search(
        r"\\n\\n|\\n(?:```|#{1,6}\s|[（(【]|\d+[.)、])",
        value,
    ))
    if not looks_like_escaped_document:
        return value
    return re.sub(
        r"\\n(?!(?:abla|eq|e\b|ot|u\b|atural|eg)\b)",
        "\n",
        value,
    )


def _rigorous_failure_answer(message: str, profile: dict[str, Any]) -> str:
    """Give a useful, bounded path when both reasoning model calls fail."""
    task = message.strip(" ：:，,。？！? ")[:240]
    key = str(profile.get("key") or "")
    method = {
        "mathematical_proof": (
            "先把命题写成“前提—待证结论”，检查量词与定义域；再分别尝试直接推导、反证或构造反例。"
            "每一步只使用已经写明的定义或定理，并用一个边界情形检查结论是否过强。"
        ),
        "physical_modelling": (
            "先列系统边界、状态量、守恒律和近似条件；随后做量纲检查，再求极限情形与数值尺度。"
            "若不同模型都能解释现象，应明确指出还缺哪一个可区分它们的观测量。"
        ),
        "algorithmic_reasoning": (
            "先明确输入、输出和不变量，再写最小伪代码；随后检查终止性、正确性与时间/空间复杂度，"
            "最后用空输入、单元素和最坏结构三个边界样例反测。"
        ),
        "constructive_calculation": (
            "先抄清已知量、未知量和单位，再选公式并逐步代入；计算后用量纲、符号、数量级和特殊值回代四项复核。"
        ),
        "argument_analysis": (
            "先拆出前提、隐含假设和结论，再检查结论是否真的由前提推出；随后给出最小反例或缺失条件。"
        ),
    }.get(key, "先列出已知、目标与约束，再逐步推导，并用边界情形和反例复核每个关键跳步。")
    return (
        f"这是一道题面自足的任务：{task}。可以先按下面的可验证路径推进。\n\n"
        f"{method}\n\n"
        "当前最重要的是把第一个不可跳过的中间结论写出来；即使完整推导较长，也应先给出这一步及其验证方法，"
        "而不是把外部检索当作回答前提。"
    )


def _normalize_inspiration_answer(value: Any) -> str:
    if isinstance(value, str):
        return _decode_document_newlines(value).strip()
    if isinstance(value, list):
        return "\n\n".join(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, dict):
        return "\n\n".join(
            str(item).strip() for item in value.values() if isinstance(item, str) and item.strip()
        )
    return ""


def _normalize_inspiration_branches(value: Any) -> list[dict[str, str]]:
    """Accept model wording drift while preserving useful follow-up questions."""
    if isinstance(value, dict):
        candidates = [
            {"title": key, "question": item} if isinstance(item, str) else item
            for key, item in value.items()
        ]
    elif isinstance(value, list):
        candidates = value
    else:
        candidates = [value] if isinstance(value, str) else []

    normalized: list[dict[str, str]] = []
    seen_questions: set[str] = set()
    for candidate in candidates:
        title = ""
        question = ""
        if isinstance(candidate, str):
            question = candidate.strip()
            parts = re.split(r"[：:]", question, maxsplit=1)
            if len(parts) == 2 and 1 <= len(parts[0].strip()) <= 18:
                title, question = parts[0].strip(), parts[1].strip()
        elif isinstance(candidate, dict):
            title = next((
                str(candidate.get(key) or "").strip()
                for key in ("title", "name", "label", "topic", "theme", "heading", "direction", "perspective", "angle", "标题", "主题", "方向", "视角")
                if isinstance(candidate.get(key), str) and str(candidate.get(key)).strip()
            ), "")
            question = next((
                str(candidate.get(key) or "").strip()
                for key in ("question", "prompt", "followup", "follow_up", "query", "text", "content", "description", "detail", "suggestion", "exploration", "research_question", "问题", "追问", "提示")
                if isinstance(candidate.get(key), str) and str(candidate.get(key)).strip()
            ), "")
            if not question:
                questions = candidate.get("questions")
                if isinstance(questions, list):
                    question = next((str(item).strip() for item in questions if isinstance(item, str) and item.strip()), "")
            if not question:
                question = next((
                    item.strip() for item in candidate.values()
                    if isinstance(item, str) and item.strip() and item.strip() != title
                ), "")
            if not question and title:
                question, title = title, ""
        if not question:
            continue
        key = re.sub(r"\s+", "", question).casefold()
        if key in seen_questions:
            continue
        seen_questions.add(key)
        if not title:
            short_title = re.split(r"[，,。；;？?]", question, maxsplit=1)[0].strip()
            title = short_title[:14] + ("…" if len(short_title) > 14 else "")
            if title == question.rstrip("？?"):
                title = f"继续探索 {len(normalized) + 1}"
        normalized.append({"title": title, "question": question})
        if len(normalized) == 4:
            break
    return normalized


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
    reasoning_profile = classify_reasoning_task(message)
    auditable_code_requested = bool(_AUDITABLE_CODE_REQUEST.search(message))
    rigorous_closed_loop = bool(
        reasoning_profile.get("activated")
        and (
            is_self_contained_reasoning(message, reasoning_profile)
            or auditable_code_requested
        )
        and (
            auditable_code_requested
            or reasoning_profile.get("key") in {
                "mathematical_proof", "physical_modelling", "algorithm_design",
                "code_diagnosis", "constructive_calculation", "argument_analysis",
            }
        )
        and (
            auditable_code_requested
            or re.search(
                r"证明|推导|计算|求出|导出|判断|收敛半径|特征值|不动点|Jacobian",
                message,
                re.I,
            )
        )
    )
    if rigorous_closed_loop:
        kind, needs_anchor = "rigorous_exploration", False
    reasoning_guide = reasoning_prompt(reasoning_profile, surface="inspiration")
    precision_guide = science_precision_instruction(message)
    memory = LearningMemoryService(store)
    turn = memory.begin_turn(message, session_id, capability="inspiration")
    session_id = turn["session_id"]
    recalled = memory.active_memory_context(
        surface="inspiration",
        task_keys=[str(reasoning_profile.get("task_key") or "general")],
    )
    preference_claims = [
        item for item in recalled.get("claims", [])
        if str(item.get("dimension") or "") == "teaching_preference"
    ][:4]
    preference_directives = [str(item.get("claim_text") or "").strip() for item in preference_claims]
    preference_guide = (
        "用户对同类推理表达的已确认反馈：" + "；".join(preference_directives)
        if preference_directives else "本轮没有足够的同类表达偏好证据，采用标准讨论方式。"
    )
    previous_message = next((
        item["content"] for item in reversed(history) if item["role"] == "user"
    ), "")
    retrieval_text = (
        f"{previous_message}\n当前追问：{message}"
        if previous_message and len(message) <= 12 else message
    )
    hits = search_notes(
        store, retrieval_text, kinds={"textbook", "course", "concept", "knowledge"}, limit=6,
    ) if needs_anchor else []
    anchors = [
        item for item in hits
        if item.get("knowledge_status") == "grounded"
        and _directly_relevant_anchor(retrieval_text, item)
    ][:3]
    anchor_text = "\n".join(
        f"[{index}]《{item['title']}》：{item['snippet']}" for index, item in enumerate(anchors, 1)
    ) or "本轮没有取得可核验的事实锚点"
    payload = None
    generation_failed = False
    generation_errors: list[str] = []
    if rigorous_closed_loop:
        auditable_code_required = auditable_code_requested
        auditable_code_instruction = (
            "题目明确要求代码或科学工具：最终正文必须至少包含一个以 ```python 开始、以 ``` 结束的"
            "完整 Python 代码块；代码要写明依赖与输入，禁止只给伪代码，禁止把未运行内容声称为实测结果。"
            if auditable_code_required else ""
        )
        closed_loop_system_prompt = (
            "你是灵感检测中的基础学科严谨推理伙伴。当前问题是题面自足的证明、推导或计算，"
            "必须直接完成推导，并给出关键公式、成立条件、反例或机制辨析。可以补充另一条思路，"
            "但严禁套用人物动机、情绪、群体氛围等心理分析模板。输出可独立展示的完整正文，"
            "不要输出 JSON、审校意见或增量补丁。"
            + auditable_code_instruction
            + precision_guide
        )
        closed_loop_user_prompt = (
            f"当前问题：{message}\n{reasoning_guide}\n{preference_guide}\n"
            f"{auditable_code_instruction}\n{precision_guide}"
        )
        try:
            plain_answer = chat(
                closed_loop_system_prompt,
                closed_loop_user_prompt,
                temperature=0.25, timeout=closed_loop_generation_timeout_seconds(), max_retries=0,
            )
            if plain_answer and plain_answer.strip():
                payload = {
                    "primary_type": kind, "secondary_types": ["closed_loop_reasoning"],
                    "answer": plain_answer.strip(),
                    "acknowledgement": "这是一道适合从形式推导与反例两侧展开的问题。",
                    "assumptions": [], "claims": [], "counter_view": "",
                    "branches": [
                        {"title": "复核关键等式", "question": "你想逐项检查哪一个中间公式？"},
                        {"title": "寻找边界反例", "question": "若放松一个成立条件，结论会怎样失效？"},
                    ],
                }
        except LLMError as exc:
            generation_errors.append(str(exc)[:500])
        if payload is None:
            # Keep retry budget below the outer evaluation wall clock. This
            # catches a transient overload/timeout without waiting another
            # full closed-loop timeout window.
            try:
                plain_answer = chat(
                    closed_loop_system_prompt,
                    closed_loop_user_prompt,
                    temperature=0.25,
                    timeout=min(30.0, closed_loop_generation_timeout_seconds()),
                    max_retries=0,
                )
                if plain_answer and plain_answer.strip():
                    payload = {
                        "primary_type": kind, "secondary_types": ["closed_loop_reasoning"],
                        "answer": plain_answer.strip(),
                        "acknowledgement": "这是一道适合从形式推导与反例两侧展开的问题。",
                        "assumptions": [], "claims": [], "counter_view": "",
                        "branches": [
                            {"title": "复核关键等式", "question": "你想逐项检查哪一个中间公式？"},
                            {"title": "寻找边界反例", "question": "若放松一个成立条件，结论会怎样失效？"},
                        ],
                    }
            except LLMError as exc:
                generation_errors.append(str(exc)[:500])
    else:
        try:
            payload = chat_json(
            "你是知识花园的灵感讨论伙伴：聪明、细腻、愿意思辨，不是分类器、答案裁判或教科书复读机。"
            "先理解用户真正惊讶、困惑、在意或想推演的具体问题，然后给出一段自然、连贯、信息密度高的中文回答。"
            "answer 是展示给用户的主体：通常写 450～850 个汉字；问题简单也至少把一个解释讲透，"
            "复杂或跨学科问题可写到 1000 字。不要注水，不要每句话套‘事实、推测、灵感、待核验’标签。"
            "按问题自身选择视角，而不是固定模板：例如个体动机与情绪、社会激励、历史文化、机制与边界、"
            "反例、思想实验、伦理影响、可观察后果等；选择真正适合当前问题的 2～4 个，自然衔接。"
            "允许深入分析，但用‘可能、可以这样理解、另一种解释’区分假设与结论；避免把猜测说成已证实事实。"
            "参考资料只在它直接解释当前问题的核心对象、机制或背景时使用；宁可完全不用，"
            "也不要因为几个相同日常词、用户学过某本书或同一学科而强行引用教材。"
            "确实使用资料时，在相应句子后标 [1] 等锚点编号；没有相关资料仍可开展诚实的思辨讨论。"
            "输出 JSON：answer、primary_type、secondary_types、acknowledgement、assumptions、claims、"
            "counter_view、branches、used_anchor_indexes。claims 仅供后台校准，不要照搬成正文；"
            "每项含 status(fact/inference/imagination/uncertain)、text、anchor_index，"
            "fact 必须真的由指定锚点直接支持；无锚点的现实断言标 uncertain。"
            "branches 必须是 2～4 个对象的数组；每个对象必须同时包含 title（简短方向标题）"
            "与 question（完整、具体、可直接继续讨论的问题）。不要固定写机制/反例/应用。"
            "不要替用户选择下一步。",
            f"已有对话：{history or '无'}\n当前输入：{message}\n初步类型：{kind}\n"
            f"{reasoning_guide}\n{preference_guide}\n"
            f"仅供选择的直接相关资料：\n{anchor_text}\n"
            "请直接回应当前问题，承接最近上下文，不要把不相关教材塞进回答。",
            )
        except LLMError as exc:
            generation_errors.append(str(exc)[:500])
            payload = None
    if payload is None and rigorous_closed_loop:
        generation_failed = True
        payload = {
            "primary_type": kind, "secondary_types": ["closed_loop_reasoning"],
            "answer": _rigorous_failure_answer(message, reasoning_profile),
            "acknowledgement": "先给出与题型匹配的可验证推进路径。",
            "assumptions": [], "claims": [], "counter_view": "",
            "branches": [
                {"title": "检查第一步", "question": "先把第一个关键中间式写出来并逐项核对条件，结果是什么？"},
                {"title": "寻找边界", "question": "把一个成立条件放宽后，最小反例会出现在哪里？"},
            ],
        }
    if payload is None:
        # A deterministic conversational fallback is still a model-generation
        # failure.  Keep the fallback usable, but never count it as model success.
        generation_failed = True
    payload = payload or {
        "primary_type": kind, "secondary_types": [],
        "answer": _fallback_inspiration_answer(message, kind),
        "acknowledgement": "这个观察值得先展开，再慢慢区分它依赖的前提。",
        "assumptions": [message],
        "claims": [
            {"status": "uncertain", "text": "目前只能提出解释假设，不能替具体人物断定真实动机。", "anchor_index": None},
            {"status": "imagination", "text": "改变情境条件后，可以比较不同解释会预期怎样的结果。", "anchor_index": None},
        ],
        "counter_view": "也可以反过来问：有没有一种更简单的解释，会产生同样的直觉？",
        "branches": [
            {"title": "把现象讲具体", "question": "你最在意的是哪个具体情境，或者哪一种行为？"},
            {"title": "换一种解释", "question": "如果不从表面动机解释，还可能是什么因素在推动？"},
            {"title": "寻找例外", "question": "在什么条件下，这个现象可能明显减弱或反过来？"},
        ],
    }
    valid_anchor_indexes = set(range(1, len(anchors) + 1))
    claims = []
    for raw in payload.get("claims") or []:
        if not isinstance(raw, dict):
            continue
        status = str(raw.get("status", "uncertain"))
        anchor_index = raw.get("anchor_index")
        if isinstance(anchor_index, str) and anchor_index.isdigit():
            anchor_index = int(anchor_index)
        if status == "fact" and anchor_index not in valid_anchor_indexes:
            status = "uncertain"
            anchor_index = None
        if status == "fact" and anchor_index in valid_anchor_indexes:
            anchor = anchors[anchor_index - 1]
            supported = relevance_gate(
                str(raw.get("text") or ""), anchor["title"], anchor.get("snippet", ""),
            )
            if not supported["passed"]:
                status = "uncertain"
                anchor_index = None
        if status not in {"fact", "inference", "imagination", "uncertain"}:
            status = "uncertain"
        claims.append({"status": status, "text": str(raw.get("text", "")).strip(), "anchor_index": anchor_index})
    answer = _normalize_inspiration_answer(payload.get("answer"))
    if not answer:
        answer = "\n\n".join(filter(None, [
            str(payload.get("acknowledgement") or "").strip(),
            *[item["text"] for item in claims if item["text"]],
            str(payload.get("counter_view") or "").strip(),
        ])) or _fallback_inspiration_answer(message, kind)
    if _META_REFUSAL_LANGUAGE.search(answer):
        generation_failed = True
        answer = (
            _rigorous_failure_answer(message, reasoning_profile)
            if rigorous_closed_loop else _fallback_inspiration_answer(message, kind)
        )
    used_indexes = {
        int(item["anchor_index"]) for item in claims
        if item["status"] == "fact" and item.get("anchor_index") in valid_anchor_indexes
    }
    for value in payload.get("used_anchor_indexes") or []:
        if isinstance(value, int) and value in valid_anchor_indexes and f"[{value}]" in answer:
            used_indexes.add(value)
    used_anchors = [item for index, item in enumerate(anchors, 1) if index in used_indexes]
    anchor_index_map = {
        old_index: new_index for new_index, old_index in enumerate(sorted(used_indexes), 1)
    }
    for claim in claims:
        old_index = claim.get("anchor_index")
        if old_index in anchor_index_map:
            claim["anchor_index"] = anchor_index_map[old_index]
        elif old_index is not None:
            claim["anchor_index"] = None
    answer = re.sub(
        r"\[(\d+)\]",
        lambda match: f"[{anchor_index_map[int(match.group(1))]}]"
        if int(match.group(1)) in anchor_index_map else "",
        answer,
    )
    event_id = memory.record_event(
        surface="inspiration", event_type="inspiration_turn", source_kind="observed",
        session_id=session_id, message_id=turn["message_id"], concepts=[], payload={
            "l1_only": True, "eligible_for_reflection": False,
            "primary_type": payload.get("primary_type", kind),
            "reasoning_type": reasoning_profile.get("key") if reasoning_profile.get("activated") else "general",
            "message": message[:1000],
        },
    )
    reasoning_summary = {
        "type": reasoning_profile.get("key") if reasoning_profile.get("activated") else "general",
        "label": reasoning_profile.get("label") if reasoning_profile.get("activated") else "开放探索",
        "confidence": reasoning_profile.get("confidence", 0.0),
        "task_key": reasoning_profile.get("task_key", "general"),
        "review": review_reasoning_answer(reasoning_profile, answer, surface="inspiration"),
    }
    personalization = {
        "status": "applied" if preference_claims else "standard",
        "task_key": reasoning_profile.get("task_key", "general"),
        "confidence": max(
            (float(item.get("effective_confidence") or 0.0) for item in preference_claims),
            default=0.0,
        ),
        "strategy_summary": preference_guide,
        "applied_claim_ids": [str(item.get("claim_id")) for item in preference_claims if item.get("claim_id")],
        "evidence": [evidence for item in preference_claims for evidence in item.get("evidence", [])][:4],
    }
    assistant_message_id = memory.complete_capability_turn(
        turn,
        answer=answer,
        capability="inspiration",
        metadata={"reasoning": reasoning_summary, "personalization": personalization},
    )
    return {
        "session_id": session_id, "request_id": turn["request_id"],
        "event_id": event_id, "assistant_message_id": assistant_message_id, "mode": "inspiration",
        "primary_type": payload.get("primary_type", kind),
        "secondary_types": payload.get("secondary_types") or [],
        "answer": answer,
        "acknowledgement": str(payload.get("acknowledgement") or "我认真接住了这个想法。"),
        "assumptions": [str(item) for item in (payload.get("assumptions") or [])][:4],
        "claims": claims[:8], "counter_view": str(payload.get("counter_view") or ""),
        "branches": _normalize_inspiration_branches(payload.get("branches")),
        "anchors": [
            {"id": item["id"], "title": item["title"], "path": item["path"]}
            for item in used_anchors
        ],
        "reasoning": reasoning_summary,
        "generation_failed": generation_failed,
        "generation_diagnostics": {
            "errors": generation_errors,
            "fallback_used": bool(generation_failed),
            "mode": "closed_loop" if rigorous_closed_loop else "exploration",
            "auditable_python_required": bool(
                rigorous_closed_loop and auditable_code_requested
            ),
            "auditable_python_present": bool(re.search(
                r"```(?:python|py)\s*\r?\n", answer, re.I,
            )),
        },
        "personalization": personalization,
        "notice": "开放讨论中的假设不会更新掌握度，也不会自动形成长期画像。",
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
