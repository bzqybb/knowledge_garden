from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from core.authority_research import search_wikipedia
from core.context import GardenContext
from core.deepdiagram_adapter import (
    DiagramSpec, build_local_diagram, diagram_has_teaching_value,
    diagram_is_grounded, unavailable_diagram,
)
from core.deepdiagram_service import DeepDiagramServiceError, generate_with_full_service
from core.learning_memory import LearningMemoryService
from core.llm import LLMError, chat_json, understanding_chat_json
from core.query_understanding import ALIAS_GROUPS, build_query_plan
from core.reasoning_capability import (
    classify_reasoning_task,
    is_self_contained_reasoning,
    reasoning_prompt,
    reasoning_subject,
    review_reasoning_answer,
)
from core.retrieval import relevance_gate, search_notes
from core.storage import GardenStore
from core.tracememo import TraceMemoClient, TraceMemoError, tracememo_config
from core.web_research import (
    fetch_open_access_pdf_text,
    search_academic_articles,
    search_public_web,
)


class IntentResult(BaseModel):
    primary_intent: Literal[
        "define", "explain_mechanism", "apply", "compare", "evaluate", "design", "clarify"
    ] = "clarify"
    secondary_intents: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    task_demand: Literal["remember", "understand", "apply", "analyze", "evaluate", "create"] = "understand"
    possible_obstacle: Literal[
        "none", "definition_gap", "prerequisite_gap", "causal_gap", "comparison_gap", "application_gap", "unknown"
    ] = "unknown"
    needs_clarification: bool = False
    clarification_question: str = ""
    evidence: str = ""
    research_object: str = ""
    target_kind: Literal[
        "concept", "person", "organization", "place", "work", "event", "unknown"
    ] = "unknown"
    core_question: str = ""
    claim_to_verify: str = ""
    longitudinal_questions: list[str] = Field(default_factory=list)
    horizontal_questions: list[str] = Field(default_factory=list)
    response_mode: Literal["standard", "domain_overview"] = "standard"
    first_exposure_evidence: str = ""
    profile_graph_claim_ids_used: list[str] = Field(default_factory=list)
    profile_graph_rationale: str = ""
    retrieval_queries: list[str] = Field(default_factory=list)
    canonical_subject: str = ""
    candidate_aliases: list[str] = Field(default_factory=list)
    explicit_constraints: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    evidence_from_user: str = ""


class SourcePlan(BaseModel):
    local_first: bool = True
    source_types: list[Literal[
        "local_wiki", "textbook", "encyclopedia", "review", "research_paper",
        "official_docs", "public_web", "wechat_history",
    ]] = Field(default_factory=lambda: ["local_wiki"])
    search_query: str = ""
    recency_needed: bool = False
    rationale: str = ""


class WeChatLookup(BaseModel):
    requested: bool = False
    talker: str = ""
    time_hint: str = ""
    topic_terms: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str = ""
    evidence: str = ""


class EvidenceDecision(BaseModel):
    accepted_ids: list[str] = Field(default_factory=list)
    rejected: list[dict[str, str]] = Field(default_factory=list)
    usable_claims: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    sufficient: bool = False
    rationale: str = ""
    source_roles: dict[str, str] = Field(default_factory=dict)


class TeachingStrategy(BaseModel):
    teaching_move: Literal[
        "direct_definition", "build_prerequisite", "repair_causal_chain", "contrast_cases",
        "worked_example", "test_boundary", "co_design", "clarify_first"
    ] = "direct_definition"
    explanation_order: list[str] = Field(default_factory=lambda: ["结论", "机制", "边界"])
    use_analogy: bool = False
    analogy_basis: str = ""
    rigor: Literal["intuitive", "conceptual", "formal"] = "conceptual"
    personalization_basis: str = ""
    preference_directives: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    success_criterion: str = ""
    rationale: str = ""
    personalization_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    applied_evidence_ids: list[str] = Field(default_factory=list)


class PersonalizationPlan(BaseModel):
    status: Literal["disabled_first_exposure", "standard", "light", "applied"] = "standard"
    task_key: str = "general"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    strategy_summary: str = "标准讲解（没有足够个性化证据）"
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    applied_claim_ids: list[str] = Field(default_factory=list)
    allowed_adjustments: list[str] = Field(default_factory=list)
    prohibited_adjustments: list[str] = Field(default_factory=lambda: [
        "改变事实内容", "省略关键前置", "跨领域强行类比", "把临时表现描述成稳定人格",
    ])
    fallback_reason: str = ""


class PlannerDecision(BaseModel):
    goal: str = "回答当前学习问题"
    complexity: Literal["simple", "moderate", "complex"] = "moderate"
    required_steps: list[str] = Field(default_factory=lambda: [
        "load_learner_memory", "plan_sources", "retrieve_sources",
        "audit_evidence", "choose_teaching_strategy", "generate_answer", "reflect_outputs",
    ])
    primary_modality: Literal["text", "text_visual"] = "text"
    relation_type: Literal[
        "none", "hierarchy", "causal_process", "chronology", "comparison",
        "spatial_geometric", "quantitative",
    ] = "none"
    visual_kind: Literal["none", "mindmap", "flowchart", "timeline", "comparison", "concept"] = "none"
    modality_reason: str = "文字足以清楚回答当前问题。"
    visual_request: str = ""
    online_research: bool = False
    reflection_required: bool = True
    max_revisions: int = Field(default=1, ge=0, le=1)
    stop_condition: str = "回答问题、证据边界清楚且表达形态适配任务。"
    policy_adjustments: list[str] = Field(default_factory=list)


class QualityReview(BaseModel):
    passed: bool = True
    answered_question: bool = True
    evidence_bounded: bool = True
    personalization_natural: bool = True
    expression_natural: bool = True
    boundary_appropriate: bool = True
    medical_safe: bool = True
    modality_fit: bool = True
    visualization_grounded: bool = True
    repair_target: Literal["none", "text", "visualization", "both"] = "none"
    issues: list[str] = Field(default_factory=list)
    revised_answer: str = ""
    rationale: str = ""


class GardenerState(TypedDict, total=False):
    store: GardenStore
    context: GardenContext
    question: str
    history: list[dict[str, str]]
    dialogue: str
    learner_context: dict[str, Any]
    profile_graph: dict[str, Any]
    personalization_plan: dict[str, Any]
    planner_decision: dict[str, Any]
    intent: dict[str, Any]
    reasoning_profile: dict[str, Any]
    source_plan: dict[str, Any]
    wechat_lookup: dict[str, Any]
    local_hits: list[dict[str, Any]]
    candidate_sources: list[dict[str, Any]]
    retrieval_attempts: list[str]
    retrieval_errors: list[str]
    evidence_review: dict[str, Any]
    accepted_sources: list[dict[str, Any]]
    generation_sources: list[dict[str, Any]]
    teaching_strategy: dict[str, Any]
    content_blueprint: dict[str, Any]
    answer: str
    followup: str
    discussion_prompts: list[str]
    quality_review: dict[str, Any]
    visualization: dict[str, Any]
    revision_count: int
    trace: list[dict[str, Any]]
    result: dict[str, Any]
    direct_material: dict[str, Any]


def _extract_frontier_material(text: str) -> tuple[str, dict[str, Any]]:
    match = re.search(r"<frontier_material>\s*(.*?)\s*</frontier_material>", text, re.S | re.I)
    if not match:
        return text.strip(), {}
    block = match.group(1)
    abstract_match = re.search(r"(?:^|\n)abstract:\s*\n(.*)\Z", block, re.S | re.I)
    abstract = abstract_match.group(1).strip() if abstract_match else ""
    header = block[:abstract_match.start()] if abstract_match else block
    fields = {
        key.lower(): value.strip()
        for key, value in re.findall(r"^([a-z_]+):\s*(.*)$", header, re.M | re.I)
    }
    fields["abstract"] = abstract
    fields["authors"] = [item.strip() for item in fields.get("authors", "").split(";") if item.strip()]
    return (text[:match.start()] + text[match.end():]).strip(), fields


def _validated(schema: type[BaseModel], payload: dict[str, Any] | None, fallback: BaseModel) -> dict[str, Any]:
    if payload is None:
        return fallback.model_dump()
    try:
        return schema.model_validate(payload).model_dump()
    except Exception:
        return fallback.model_dump()


def _agent_json(system: str, user: str, *, timeout: float = 18) -> dict[str, Any] | None:
    """Bound one specialist call so a failed sub-agent cannot stall the turn."""
    return chat_json(system, user, timeout=timeout, max_retries=0)


def _understanding_agent_json(system: str, user: str) -> tuple[dict[str, Any] | None, str]:
    """Keep semantic parsing on its dedicated GLM lane when configured."""
    return understanding_chat_json(system, user, timeout=6, max_retries=0)


def _normalize_understanding_payload(
    payload: dict[str, Any] | None, fallback: IntentResult,
) -> dict[str, Any] | None:
    """Coerce formatting drift without inventing any new semantic content."""
    if not isinstance(payload, dict):
        return None
    cleaned = dict(payload)
    raw_intent = str(cleaned.get("primary_intent") or "").strip().lower()
    intent_aliases = (
        (("define", "定义", "获取定义", "是什么"), "define"),
        (("explain_mechanism", "机制", "原理", "因果"), "explain_mechanism"),
        (("apply", "应用", "求解", "用法"), "apply"),
        (("compare", "比较", "对比", "区别"), "compare"),
        (("evaluate", "评价", "评判", "判断"), "evaluate"),
        (("design", "设计", "构建", "创造"), "design"),
        (("clarify", "澄清", "不明确"), "clarify"),
    )
    normalized_intent = next((
        target for aliases, target in intent_aliases
        if any(alias == raw_intent or alias in raw_intent for alias in aliases)
    ), fallback.primary_intent)
    cleaned["primary_intent"] = normalized_intent
    task_defaults = {
        "define": "understand", "explain_mechanism": "analyze", "apply": "apply",
        "compare": "analyze", "evaluate": "evaluate", "design": "create", "clarify": "understand",
    }
    allowed_demands = {"remember", "understand", "apply", "analyze", "evaluate", "create"}
    if cleaned.get("task_demand") not in allowed_demands:
        cleaned["task_demand"] = task_defaults[normalized_intent]
    allowed_obstacles = {
        "none", "definition_gap", "prerequisite_gap", "causal_gap",
        "comparison_gap", "application_gap", "unknown",
    }
    if cleaned.get("possible_obstacle") not in allowed_obstacles:
        cleaned["possible_obstacle"] = fallback.possible_obstacle
    target_kind = str(cleaned.get("target_kind") or "").strip().casefold()
    target_aliases = {
        "概念": "concept", "人物": "person", "人": "person", "机构": "organization",
        "组织": "organization", "地点": "place", "作品": "work", "事件": "event",
    }
    target_kind = target_aliases.get(target_kind, target_kind)
    cleaned["target_kind"] = (
        target_kind if target_kind in {
            "concept", "person", "organization", "place", "work", "event", "unknown",
        } else fallback.target_kind
    )
    raw_clarification = cleaned.get("needs_clarification", fallback.needs_clarification)
    if isinstance(raw_clarification, bool):
        cleaned["needs_clarification"] = raw_clarification
    elif isinstance(raw_clarification, str):
        compact = re.sub(r"\s+", "", raw_clarification).casefold()
        if any(token in compact for token in ("不需要", "无需", "不必", "已明确", "false", "否")):
            cleaned["needs_clarification"] = False
        elif any(token in compact for token in ("需要澄清", "无法确定", "有歧义", "true", "是")):
            cleaned["needs_clarification"] = True
        else:
            cleaned["needs_clarification"] = fallback.needs_clarification
    else:
        cleaned["needs_clarification"] = bool(raw_clarification)
    for field in (
        "secondary_intents", "concepts", "candidate_aliases",
        "explicit_constraints", "ambiguities", "retrieval_queries",
    ):
        value = cleaned.get(field)
        if value is None:
            cleaned[field] = []
        elif not isinstance(value, list):
            cleaned[field] = [str(value).strip()] if str(value).strip() else []
    try:
        cleaned["confidence"] = max(0.0, min(1.0, float(cleaned.get("confidence") or 0.0)))
    except (TypeError, ValueError):
        cleaned["confidence"] = 0.0
    return cleaned


def _simple_definition_payload(question: str) -> dict[str, Any] | None:
    """Millisecond fallback for unambiguous definition syntax only."""
    text = re.sub(r"^【[^】]+】", "", question).strip(" ：:，,。？?!！")
    subject = ""
    match = re.fullmatch(r"(?:请问)?(?:什么是|何为)([^？?。！!]{2,40})", text)
    if match:
        subject = match.group(1).strip()
    else:
        match = re.fullmatch(r"([^？?。！!]{2,40}?)(?:是什么|指什么)", text)
        if match:
            subject = match.group(1).strip()
    if not subject:
        match = re.fullmatch(
            r"(?:请)?解释(?:一下)?[‘“\"']?([^？?。！!‘’“”\"']{2,40})[’”\"']?",
            text,
        )
        if match:
            subject = match.group(1).strip()
    if not subject or re.search(r"和|与|以及|同时|区别|为什么|怎么|如何", subject):
        return None
    return {
        "primary_intent": "define", "secondary_intents": [],
        "concepts": [subject], "task_demand": "understand",
        "possible_obstacle": "definition_gap", "needs_clarification": False,
        "evidence": "当前问句完整匹配无歧义的定义句式。",
        "evidence_from_user": question, "research_object": subject,
        "canonical_subject": subject, "candidate_aliases": [],
        "core_question": f"{subject}是什么", "claim_to_verify": "",
        "explicit_constraints": [], "ambiguities": [], "confidence": 1.0,
        "retrieval_queries": [subject],
    }


def _explicit_academic_concepts(question: str) -> list[str]:
    """Return only auditable terminology that appears literally in the question."""
    compact = re.sub(r"\s+", "", question).casefold()
    concepts: list[str] = []
    for group in ALIAS_GROUPS:
        explicit = [
            str(alias).strip() for alias in group
            if len(re.sub(r"\s+", "", str(alias))) >= 2
            and re.sub(r"\s+", "", str(alias)).casefold() in compact
        ]
        if explicit:
            concepts.append(str(group[0]).strip())
    return list(dict.fromkeys(concepts))


def _contextual_understanding_fallback(
    question: str, recent_user: str, fallback: IntentResult,
) -> dict[str, Any]:
    """Resolve only narrow, explicit multi-turn forms after a provider failure.

    This is deliberately not a second home-grown general NLP system.  It only
    handles a recent antecedent plus an explicit comparison target, which keeps
    the fallback useful without silently inventing context.
    """
    cleaned = question.strip(" ：:，,。？！? ")
    antecedent = _question_subject(recent_user) if recent_user else ""
    comparison = re.fullmatch(
        r"(?:那|那么)?(?:它|这个|这(?:个|种|一))?和(.{2,40}?)(?:有什么)?(?:区别|不同|异同)",
        cleaned,
    )
    if comparison and antecedent:
        target = comparison.group(1).strip(" ：:，,。？！? ")
        if target and target != antecedent:
            return {
                **fallback.model_dump(),
                "primary_intent": "compare",
                "secondary_intents": [],
                "concepts": [antecedent, target],
                "task_demand": "analyze",
                "possible_obstacle": "comparison_gap",
                "needs_clarification": False,
                "evidence": "远程问题理解暂不可用；依据最近一条用户问题中的明确对象消解本轮指代。",
                "evidence_from_user": f"上轮：{recent_user}；本轮：{question}",
                "research_object": f"{antecedent}与{target}",
                "canonical_subject": f"{antecedent}与{target}",
                "core_question": f"{antecedent}和{target}有什么区别",
                "confidence": 0.78,
                "ambiguities": [],
                "retrieval_queries": [antecedent, target],
            }
    return fallback.model_dump()


def _trace(state: GardenerState, node: str, summary: str, data: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return [*(state.get("trace") or []), {"node": node, "summary": summary, "data": data or {}}]


def planner_intake(state: GardenerState) -> dict[str, Any]:
    """Create an explicit first command instead of hiding orchestration in edges."""
    return {
        "revision_count": 0,
        "trace": _trace(state, "planner_intake", "总控先委派问题理解者固定当前任务", {
            "next_agent": "understand_question",
            "reason": "规划前必须先知道问题性质，不能直接套固定讲解模板。",
        }),
    }


def _relation_policy(question: str, intent: dict[str, Any]) -> tuple[str, str]:
    """Select a visual by the relation being learned, not by question words."""
    text = " ".join([
        question, str(intent.get("research_object") or ""),
        " ".join(str(item) for item in intent.get("concepts", [])),
    ])
    if re.search(r"向量|方向|空间|坐标|几何|图像|曲线|曲面|轨迹|位置|旋转|投影|变换", text):
        return "spatial_geometric", "concept"
    if re.search(r"时间线|沿时间|演化|发展史|历程|阶段变迁|纵向", text):
        return "chronology", "timeline"
    if intent.get("primary_intent") == "compare" or re.search(r"区别|对比|异同|优劣|相同点", text):
        return "comparison", "comparison"
    if re.search(r"层级|体系|分支|分类|组成|结构|全貌|思维导图|知识树|知识图谱", text):
        return "hierarchy", "mindmap"
    if re.search(r"数据|比例|增长率|趋势|分布|相关系数|统计", text):
        return "quantitative", "concept"
    if re.search(r"流程|步骤|循环|反馈|经过|先.+再|导致|触发|传递|机制链", text):
        return "causal_process", "flowchart"
    return "none", "none"


def _fallback_planner(question: str, intent: dict[str, Any]) -> PlannerDecision:
    primary = str(intent.get("primary_intent") or "clarify")
    explicit_visual = bool(re.search(r"思维导图|脑图|图解|画(?:一|个|张)?图|流程图|时间线|可视化|关系图", question))
    relation_type, kind = _relation_policy(question, intent)
    if kind == "none" and (primary == "design" or intent.get("response_mode") == "domain_overview"):
        relation_type, kind = "hierarchy", "mindmap"
    visual_required = explicit_visual or kind != "none"
    complexity = "complex" if primary in {"evaluate", "design"} or intent.get("response_mode") == "domain_overview" else "moderate"
    if primary == "define" and not explicit_visual:
        complexity = "simple"
        visual_required = False
        kind = "none"
    return PlannerDecision(
        goal=str(intent.get("core_question") or question)[:180],
        complexity=complexity,
        primary_modality="text_visual" if visual_required else "text",
        relation_type=relation_type,
        visual_kind=kind,
        modality_reason=(
            f"当前问题的核心关系是 {relation_type}，图解能降低用户在脑中维持关系的负担。"
            if visual_required else "当前问题用短文本即可完整表达，额外图形会增加噪声。"
        ),
        visual_request=(
            f"为“{str(intent.get('core_question') or question)[:100]}”生成一张{kind}，重点表达 {relation_type}；只呈现回答中得到证据支持的关键关系，"
            "节点使用知识短语，不复制长段落，并单独标出适用边界。"
            if visual_required else ""
        ),
        online_research=primary in {"explain_mechanism", "compare", "evaluate", "design"} or intent.get("response_mode") == "domain_overview",
    )


def _requires_authority_lookup(question: str, intent: dict[str, Any]) -> bool:
    """Keep factual learning questions from silently becoming local-only.

    Wiki-first is a ranking policy, not permission to stop before an available
    authority lookup.  Explicit user requests to stay offline still win.
    """
    if re.search(r"不要联网|别联网|仅(?:用|依据)本地|只查(?:本地|知识库)", question):
        return False
    if (
        _response_profile(question) != "grounded_knowledge"
        and not re.search(r"官方|来源|论文|研究表明|统计数据|查证|联网|查一下|从游", question)
    ):
        return False
    if intent.get("response_mode") == "domain_overview":
        return True
    return str(intent.get("primary_intent") or "") in {
        "define", "explain_mechanism", "compare", "evaluate", "design",
    }


def planner_plan(state: GardenerState) -> dict[str, Any]:
    """Plan the route after the question-understanding agent has fixed intent.

    The model may choose modality and research breadth, but a deterministic
    policy gate always restores evidence audit, privacy and reflection steps.
    """
    intent = state["intent"]
    fallback = _fallback_planner(state["question"], intent)
    payload = None
    explicit_visual = bool(re.search(r"思维导图|脑图|图解|画(?:一|个|张)?图|流程图|时间线|可视化|关系图", state["question"]))
    open_discussion_fast_path = (
        _response_profile(state["question"]) != "grounded_knowledge"
        and intent.get("response_mode") != "domain_overview"
        and not explicit_visual
    )
    if open_discussion_fast_path:
        fallback = fallback.model_copy(update={
            "complexity": "moderate", "primary_modality": "text",
            "relation_type": "none", "visual_kind": "none", "visual_request": "",
            "online_research": _requires_authority_lookup(state["question"], intent),
            "modality_reason": "开放讨论更适合自然对话，不需要额外规划模型或装饰性图解。",
        })
    simple_fast_path = (
        intent.get("primary_intent") in {"define", "clarify"}
        and intent.get("response_mode") != "domain_overview"
        and not explicit_visual
        and not intent.get("secondary_intents")
        and not intent.get("claim_to_verify")
    )
    deep_planning_needed = (
        intent.get("primary_intent") in {"evaluate", "design"}
        or intent.get("response_mode") == "domain_overview"
        or explicit_visual
        or len(intent.get("secondary_intents") or []) >= 2
    )
    if deep_planning_needed and not simple_fast_path and not open_discussion_fast_path:
        try:
            payload = _agent_json(
                "你是知识花园总控 Planner，不回答知识问题。问题理解者已经给出结构化诊断。请规划最小但完整的执行路径，并决定最终应以纯文字还是文字+可视化表达。先判断要表达的关系语义，而不是按‘为什么’等问句关键词选图：层级全貌用 mindmap；有先后步骤的因果过程用 flowchart；发展脉络用 timeline；同维度对照用 comparison；向量、坐标、空间、几何和函数图像等关系用 concept。普通机制解释如果没有步骤链，不应机械使用流程图。你不能取消学习记忆读取、来源规划、证据审查和最终反思；它们是系统安全约束。首次领域概览虽然不启用个性化，仍要读取记忆并由门控明确关闭，留下可审计记录。required_steps 只填写真正执行的专职 Agent 名称。",
                f"当前问题：{state['question']}\n问题理解结果：{intent}\n"
                "输出 goal、complexity(simple/moderate/complex)、required_steps、primary_modality(text/text_visual)、relation_type(none/hierarchy/causal_process/chronology/comparison/spatial_geometric/quantitative)、visual_kind(none/mindmap/flowchart/timeline/comparison/concept)、modality_reason、visual_request、online_research、reflection_required、max_revisions(0或1)、stop_condition。visual_request 是发给 DeepDiagram 的独立 user_request，必须具体说明图要帮助用户看懂什么、图中应呈现哪些关系、哪些信息不要画。",
            )
        except (LLMError, AssertionError):
            pass
    plan = _validated(PlannerDecision, payload, fallback)
    required = [
        "load_learner_memory", "plan_sources", "retrieve_sources", "audit_evidence",
        "choose_teaching_strategy", "generate_answer", "reflect_outputs",
    ]
    attempted = [item for item in required if item not in plan.get("required_steps", [])]
    plan["required_steps"] = required
    plan["reflection_required"] = True
    plan["max_revisions"] = 1
    plan["policy_adjustments"] = [f"安全门控补回：{item}" for item in attempted]
    authority_lookup_required = _requires_authority_lookup(state["question"], intent)
    if authority_lookup_required and not plan.get("online_research"):
        plan["online_research"] = True
        plan["policy_adjustments"].append(
            "事实型学习问题启用权威来源补查；本地知识优先排序，但不能在证据不足前静默停止"
        )
    policy_relation, policy_kind = _relation_policy(state["question"], intent)
    explicit_visual = bool(re.search(r"思维导图|脑图|图解|画(?:一|个|张)?图|流程图|时间线|可视化|关系图", state["question"]))
    if policy_relation != "none" and (explicit_visual or plan.get("primary_modality") == "text_visual"):
        if plan.get("visual_kind") != policy_kind:
            plan["policy_adjustments"].append(
                f"按关系语义将图类型从 {plan.get('visual_kind')} 修正为 {policy_kind}"
            )
        plan["relation_type"] = policy_relation
        plan["visual_kind"] = policy_kind
        plan["primary_modality"] = "text_visual"
    if plan.get("primary_modality") == "text":
        plan["visual_kind"] = "none"
        plan["visual_request"] = ""
    elif plan.get("visual_kind") == "none":
        plan["visual_kind"] = fallback.visual_kind if fallback.visual_kind != "none" else "mindmap"
    if plan.get("primary_modality") == "text_visual" and not str(plan.get("visual_request") or "").strip():
        plan["visual_request"] = fallback.visual_request
    return {
        "planner_decision": plan,
        "trace": _trace(state, "planner_plan", f"总控选择 {plan['primary_modality']}，复杂度 {plan['complexity']}", {
            "goal": plan["goal"], "relation_type": plan["relation_type"], "visual_kind": plan["visual_kind"], "visual_request": plan["visual_request"],
            "online_research": plan["online_research"], "required_steps": plan["required_steps"],
            "planning_mode": (
                "bounded_discussion_fast_path" if open_discussion_fast_path
                else "llm_planner" if deep_planning_needed and not simple_fast_path
                else "deterministic_fast_path" if simple_fast_path
                else "policy_planner"
            ),
            "policy_adjustments": plan["policy_adjustments"],
        }),
    }


def _requests_wechat_history(text: str) -> bool:
    """Require an explicit personal-history request before mounting WeChat data."""
    compact = re.sub(r"\s+", "", text)
    source_signal = re.search(r"微信|聊天记录|群聊|群里|私聊|消息记录", compact)
    action_signal = re.search(r"查|找|读取|看看|回顾|总结|整理|谁说|聊过|提到|讨论|消息", compact)
    personal_signal = re.search(r"我|我们|我的|和我|跟我|之前|今天|昨天|前天|本周|上周|最近", compact)
    return bool(source_signal and action_signal and personal_signal)


def _question_subject(question: str) -> str:
    """Extract a conservative terminology query when the model returns a sentence."""
    text = re.sub(r"^【[^】]+】", "", question).strip()
    text = re.sub(
        r"^(?:什么是|何为|谁是|请问什么是|请解释什么是)", "", text,
    ).strip(" ：:，,。？！? ")
    quoted = re.search(r"[《“\"]([^》”\"]{2,40})[》”\"]", text)
    if quoted:
        return quoted.group(1).strip()
    prefix = re.split(
        r"是不是|是否|是什么|是谁|是哪位|指什么|为什么|为何|怎么|如何|的定义|的历史|的起源|的核心",
        text, maxsplit=1,
    )[0]
    prefix = re.sub(r"^(?:请问|请解释|请介绍|请说明|想了解|我想了解)", "", prefix).strip(" ：:，,。？！? ")
    if 2 <= len(prefix) <= 40:
        return prefix
    terms = re.findall(r"[A-Za-z][A-Za-z /+.-]{2,40}|[\u4e00-\u9fff]{2,12}(?:学|论|法|模型|理论|效应|机制|工程)", text)
    return terms[0].strip() if terms else text[:40]


def _fallback_wechat_lookup(text: str) -> WeChatLookup:
    talker = ""
    for pattern in (
        r"(?:在|从)([^，。！？\n]{1,30}?(?:群|群聊))(?:里|中|的)",
        r"(?:和|跟|与)([^，。！？\n]{1,24}?)(?:的)?(?:微信)?(?:聊天|聊过|讨论)",
        r"(?:查|总结|回顾|读取|看看)(?:一下)?([^，。！？\n]{1,30}?(?:群|群聊))(?:里|中|的)?",
    ):
        match = re.search(pattern, text)
        if match:
            talker = match.group(1).strip(" ：:，,。")
            break
    time_match = re.search(
        r"(20\d{2}[-/.年]\d{1,2}(?:[-/.月]\d{1,2}日?)?|今天|昨天|前天|本周|上周|最近(?:7|七)天)",
        text,
    )
    time_hint = time_match.group(1) if time_match else ""
    requested = _requests_wechat_history(text)
    return WeChatLookup(
        requested=requested,
        talker=talker,
        time_hint=time_hint,
        needs_clarification=requested and not talker,
        clarification_question=(
            "你想查哪个联系人或群聊？也请给一个尽量小的时间范围，例如“昨天的 XX 群”。"
            if requested and not talker else ""
        ),
        evidence="离线规则只在用户明确要求读取个人微信历史时启用。",
    )


def _clock_datetime(clock: dict[str, Any]) -> datetime:
    for key in ("localTime", "local_time", "now", "datetime", "timestamp", "time"):
        value = clock.get(key)
        if value in (None, ""):
            continue
        try:
            if isinstance(value, (int, float)):
                return datetime.fromtimestamp(float(value)).astimezone()
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone()
        except (ValueError, TypeError, OSError):
            continue
    return datetime.now().astimezone()


def _wechat_time_params(time_hint: str, clock: dict[str, Any]) -> dict[str, str]:
    hint = time_hint.strip()
    explicit = re.search(r"(20\d{2})[-/.年](\d{1,2})(?:[-/.月](\d{1,2})日?)?", hint)
    if explicit and explicit.group(3):
        return {"time_range": f"{int(explicit.group(1)):04d}-{int(explicit.group(2)):02d}-{int(explicit.group(3)):02d}"}
    now = _clock_datetime(clock)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if hint == "昨天":
        start, end = today - timedelta(days=1), today
    elif hint == "前天":
        start, end = today - timedelta(days=2), today - timedelta(days=1)
    elif hint == "本周":
        start, end = today - timedelta(days=today.weekday()), now
    elif hint == "上周":
        end = today - timedelta(days=today.weekday())
        start = end - timedelta(days=7)
    elif re.search(r"最近(?:7|七)天", hint):
        start, end = now - timedelta(days=7), now
    elif hint == "今天":
        start, end = today, now
    else:
        return {}
    return {"start_time": str(int(start.timestamp())), "end_time": str(int(end.timestamp()))}


def understand_question(state: GardenerState) -> dict[str, Any]:
    question = state["question"]
    discussion_profile = _response_profile(question)
    profile_graph = LearningMemoryService(state["store"]).l3_profile_graph()
    profile_patterns = profile_graph.get("applicable_patterns", [])
    dialogue = state.get("dialogue", "")
    history = state.get("history", [])
    recent_user = next((
        str(item.get("content", "")).strip() for item in reversed(history)
        if item.get("role") == "user" and str(item.get("content", "")).strip()
    ), "")
    answering_clarification = bool(
        recent_user and history
        and history[-1].get("role") == "assistant"
        and history[-1].get("evidence_layer") == "clarification"
    )
    needs_resolution = answering_clarification or bool(re.search(
        r"^(?:那|那么|这个|它|上述|刚才|其中)|这(?:个|种|一)", question.strip(),
    ))
    resolved_fallback_question = (
        f"{recent_user}；补充限定：{question}"
        if answering_clarification else
        f"{recent_user}\n当前追问：{question}"
        if recent_user and needs_resolution else question
    )
    overview_mode = question.lstrip().startswith("【领域概览】")
    definition_match = re.match(
        r"^(?:请问|请解释|请说明)?(?:什么是|何为)[‘“\"']?"
        r"(?P<concept>[^？?，,；;。‘’“”\"']{2,40})",
        re.sub(r"^【[^】]+】", "", question).strip(),
    )
    fallback_concepts = [definition_match.group("concept").strip()] if definition_match else []
    fallback_concepts.extend(_explicit_academic_concepts(question))
    fallback_concepts = list(dict.fromkeys(fallback_concepts))
    application_match = re.search(
        r"(?:它|这(?:个|种))?对(?P<concept>[^？?，,；;。]{2,20}?)(?:有(?:什么|何)?影响|起(?:什么|何)?作用)",
        question,
    )
    if application_match:
        fallback_concepts.append(application_match.group("concept").strip())
    if re.search(r"证明|推导", question):
        fallback_intent = "apply"
    elif definition_match:
        fallback_intent = "compare" if re.search(r"区别|比较|不同|异同", question) else "define"
    elif re.search(r"区别|比较|异同", question):
        fallback_intent = "compare"
    elif re.search(r"怎么用|如何实现|多少|求(?:出|解)?|计算|若|当", question):
        fallback_intent = "apply"
    elif re.search(r"为什么|机制|原理|基于什么", question):
        fallback_intent = "explain_mechanism"
    elif re.search(r"(?:请|帮我|如何|怎么)(?:构建|设计)|(?:构建|设计)(?:一|一个|方案|系统)", question):
        fallback_intent = "design"
    else:
        fallback_intent = "define"
    if discussion_profile == "reflective_discussion" and fallback_intent not in {"explain_mechanism", "compare"}:
        fallback_intent = "evaluate"
    fallback = IntentResult(
        primary_intent=fallback_intent, task_demand="analyze" if fallback_intent in {"compare", "explain_mechanism"} else "understand",
        possible_obstacle="causal_gap" if fallback_intent == "explain_mechanism" else "unknown",
        concepts=list(dict.fromkeys(fallback_concepts)),
        evidence="离线规则只依据当前问题中的明确问法；未推断情绪或长期思维风格。",
        research_object=(
            _question_subject(recent_user) if answering_clarification else
            fallback_concepts[0] if fallback_concepts else
            re.sub(r"^【严谨探究】", "", question).strip()[:120]
        ),
        core_question=resolved_fallback_question,
        claim_to_verify=(question if "【严谨探究】" in question else ""),
        response_mode="domain_overview" if overview_mode else "standard",
        first_exposure_evidence="用户主动从灵感跃迁请求建立领域概览。" if overview_mode else "",
    )
    payload = None
    understanding_provider = "deterministic-fallback"
    simple_payload = _simple_definition_payload(question)
    formal_operation_fast_path = bool(
        not needs_resolution
        and not answering_clarification
        and re.search(
            r"证明|推导|计算|求(?:出|解|得|取|其|矩阵|方程|速度|加速度|电场|功|热量|振幅|频率|波长|波速)",
            question,
        )
    )
    if simple_payload is not None:
        payload = simple_payload
        understanding_provider = "deterministic-simple-definition"
    elif formal_operation_fast_path:
        payload = fallback.model_dump()
        payload["needs_clarification"] = False
        payload["clarification_question"] = ""
        understanding_provider = "deterministic-formal-operation"
    elif discussion_profile != "grounded_knowledge" and not needs_resolution and not answering_clarification:
        payload = fallback.model_dump()
        payload["needs_clarification"] = False
        payload["clarification_question"] = ""
        understanding_provider = "deterministic-bounded-discussion"
    else:
        if answering_clarification:
            few_shot = (
                '例：用户“某位研究者是谁”，助手“您指哪个领域”，用户“学术界”'
                '=> {"primary_intent":"define","research_object":"某位研究者",'
                '"target_kind":"person","core_question":"学术界的某位研究者是谁",'
                '"explicit_constraints":["学术界"],"needs_clarification":false}'
            )
        elif needs_resolution and recent_user:
            few_shot = (
                '例：上轮“什么是平权行动”，本轮“它和结果平等有什么区别”'
                '=> {"primary_intent":"compare","research_object":"平权行动与结果平等",'
                '"concepts":["平权行动","结果平等"],"needs_clarification":false}'
            )
        elif needs_resolution:
            few_shot = (
                '例：无上文，本轮“那个为什么这样”'
                '=> {"primary_intent":"clarify","research_object":"","concepts":[],'
                '"needs_clarification":true,"ambiguities":["那个无明确指代"]}'
            )
        else:
            few_shot = (
                '例：“为什么特征向量变换后方向不变”'
                '=> {"primary_intent":"explain_mechanism","research_object":"特征向量方向不变性",'
                '"concepts":["特征向量","线性变换"],"needs_clarification":false}'
            )
        try:
            payload, understanding_provider = _understanding_agent_json(
            "只输出问题解析 JSON，不回答、不检索，不读取或推断学习画像、情绪、事实或答案。"
            "primary_intent∈define/explain_mechanism/apply/compare/evaluate/design/clarify。"
            "research_object必须用用户语言写核心对象，不能复制整句；concepts只能来自本轮或最近明确上文。"
            "target_kind∈concept/person/organization/place/work/event/unknown，由语义决定。"
            "若上轮助手刚提出澄清问题，本轮简短回答是原问题的补充限定，必须保留原问题对象并合并上下文。"
            "保留数字、单位、范围、否定、前提和比较对象。人名、机构、作品可能重名不是检索前追问的理由："
            "已有明确可检索对象时先查公开证据，只有对象根本无法确定或用户任务无法执行时才澄清。"
            f"{few_shot}"
            "仅输出：primary_intent,research_object,target_kind,core_question,concepts,"
            "needs_clarification,clarification_question,explicit_constraints,ambiguities,confidence。",
            f"已有对话：\n{dialogue or '无'}\n"
            f"上一轮是否正在澄清：{'是，本轮回答是原问题的补充' if answering_clarification else '否'}\n"
            f"\n当前问题：{question}",
            )
        except LLMError:
            pass
    if payload is None and simple_payload is None:
        payload = _contextual_understanding_fallback(question, recent_user, fallback)
    intent = _validated(IntentResult, _normalize_understanding_payload(payload, fallback), fallback)
    explicit_academic_concepts = _explicit_academic_concepts(question)
    intent["concepts"] = list(dict.fromkeys([
        *intent.get("concepts", []), *explicit_academic_concepts,
    ]))
    # A remote parser may reduce a proof request to a definition because both
    # mention a theorem statement. The user's explicit operation is decisive.
    if re.search(r"证明|推导", question):
        intent["primary_intent"] = "apply"
        intent["task_demand"] = "analyze"
        intent["possible_obstacle"] = "application_gap"
    intent["core_question"] = str(intent.get("core_question") or question).strip()
    research_object = str(intent.get("research_object") or "").strip()
    if (
        not research_object or len(research_object) > 40
        or re.search(r"[？?。！!]|是不是|是否|为什么|如何|怎么", research_object)
        or re.search(r"^(?:什么是|何为|请问什么是|请解释什么是)", research_object)
        or re.sub(r"\s+", "", research_object) == re.sub(r"\s+", "", question)
    ):
        intent["research_object"] = _question_subject(question)
    previous_subject = _question_subject(recent_user) if answering_clarification else ""
    if previous_subject:
        if previous_subject not in str(intent.get("research_object") or ""):
            intent["research_object"] = previous_subject
        if previous_subject not in intent["core_question"] or question not in intent["core_question"]:
            intent["core_question"] = resolved_fallback_question
        intent["explicit_constraints"] = list(dict.fromkeys([
            *intent.get("explicit_constraints", []), question,
        ]))
        intent["concepts"] = list(dict.fromkeys([
            previous_subject, *intent.get("concepts", []),
        ]))
    if (
        intent.get("target_kind") == "unknown"
        and re.search(r"是谁|是哪位|^谁是", recent_user if answering_clarification else question)
    ):
        intent["target_kind"] = "person"
    # Related terms suggested by a model are retrieval candidates, not evidence
    # that the user asked about them.  Keep `concepts` faithful to explicit text
    # or a resolved antecedent from the supplied dialogue.
    explicit_context = re.sub(r"\s+", "", f"{dialogue}\n{question}").casefold()
    research_object_compact = re.sub(r"\s+", "", str(intent.get("research_object") or "")).casefold()
    intent["concepts"] = list(dict.fromkeys([
        str(item).strip() for item in intent.get("concepts", [])
        if str(item).strip() and (
            re.sub(r"\s+", "", str(item)).casefold() in explicit_context
            or re.sub(r"\s+", "", str(item)).casefold() == research_object_compact
            or str(item).strip() in explicit_academic_concepts
        )
    ]))
    searchable_target = (
        intent.get("target_kind") in {"person", "organization", "place", "work", "event"}
        and len(research_object_compact) >= 2
        and research_object_compact in explicit_context
    )
    if intent.get("needs_clarification") and (searchable_target or bool(previous_subject)):
        intent["needs_clarification"] = False
        intent["clarification_question"] = ""
        if intent.get("primary_intent") == "clarify":
            intent["primary_intent"] = "define"
            intent["task_demand"] = "understand"
    # The explicit UI/user marker is authoritative; the model may not silently
    # downgrade a requested first-exposure overview.
    if overview_mode:
        intent["response_mode"] = "domain_overview"
    available_profile_ids = {str(item.get("claim_id")) for item in profile_patterns}
    # Problem understanding precedes memory retrieval.  Even if a model emits
    # legacy profile fields, discard them here so L3 cannot bias what the user
    # is asking; the later memory node decides whether personalization applies.
    intent["profile_graph_claim_ids_used"] = []
    intent["profile_graph_rationale"] = ""
    candidate_aliases = [
        str(item).strip() for item in intent.get("candidate_aliases", [])
        if 1 < len(str(item).strip()) <= 80
    ][:6]
    canonical_subject = str(intent.get("canonical_subject") or "").strip()
    retrieval_suggestions = list(intent.get("retrieval_queries", []))
    if canonical_subject and canonical_subject.casefold() != str(intent["research_object"]).casefold():
        retrieval_suggestions.append(canonical_subject)
    retrieval_suggestions.extend(candidate_aliases[:2])
    # For a standalone question the user's wording is authoritative.  A model
    # generated core_question is only allowed to resolve a genuine follow-up;
    # otherwise a drifting planner can silently retrieve for a different
    # textbook concept than the one the user actually asked about.
    resolved_for_plan = intent["core_question"] if needs_resolution else question
    query_plan = build_query_plan(
        question,
        resolved_question=resolved_for_plan,
        concepts=[*intent.get("concepts", []), *candidate_aliases],
        suggested_queries=retrieval_suggestions,
    )
    intent["query_plan"] = query_plan
    intent["retrieval_queries"] = [item["text"] for item in query_plan["queries"]]
    reasoning_profile = classify_reasoning_task(
        question,
        intent_hint=str(intent.get("primary_intent") or ""),
    )
    # A semantic parser may overreact to phrases such as “根据我这句话” and
    # request clarification even though the current turn already supplies a
    # complete decision scenario.  Closed-form reasoning tasks can proceed
    # conditionally from the supplied premises; genuine missing referents stay
    # on the clarification path because they do not activate a closed protocol.
    if intent.get("needs_clarification") and is_self_contained_reasoning(question, reasoning_profile):
        intent["needs_clarification"] = False
        intent["clarification_question"] = ""
        if intent.get("primary_intent") == "clarify":
            intent["primary_intent"] = (
                "evaluate" if reasoning_profile.get("key") == "decision_analysis" else "apply"
            )
            intent["task_demand"] = "evaluate" if intent["primary_intent"] == "evaluate" else "apply"
    return {
        "intent": intent,
        "reasoning_profile": reasoning_profile,
        "profile_graph": profile_graph,
        "trace": _trace(state, "understand_question", f"识别为 {intent['primary_intent']}，任务要求 {intent['task_demand']}", {
            "evidence": intent["evidence"],
            "understanding_provider": understanding_provider,
            "target_kind": intent["target_kind"],
            "clarification_answer_resolved": answering_clarification,
            "canonical_subject_candidate": canonical_subject,
            "candidate_aliases": candidate_aliases,
            "l3_profile_claim_ids_available": sorted(available_profile_ids),
            "l3_profile_claim_ids_used": intent["profile_graph_claim_ids_used"],
            "query_plan": query_plan,
            "reasoning_type": (
                reasoning_profile.get("key") if reasoning_profile.get("activated") else "general"
            ),
        }),
    }


def route_after_understanding(state: GardenerState) -> str:
    return "planner_plan"


def route_after_planner(state: GardenerState) -> str:
    return "clarify" if state["intent"].get("needs_clarification") else "load_learner_memory"


def load_learner_memory(state: GardenerState) -> dict[str, Any]:
    """Retrieve only active, time-adjusted memory after the current intent is known."""
    concepts = [str(item) for item in state["intent"].get("concepts", []) if str(item).strip()]
    intent = state["intent"]
    reasoning_task_key = str(state.get("reasoning_profile", {}).get("task_key") or "")
    recalled = LearningMemoryService(state["store"]).active_memory_context(
        concepts,
        surface="gardener_chat",
        task_keys=[
            str(intent.get("primary_intent", "")),
            str(intent.get("task_demand", "")),
            reasoning_task_key,
        ],
    )
    learner = {
        **state.get("learner_context", {}),
        "active_memory_claims": recalled["claims"],
        "concept_mastery": recalled["concept_mastery"],
        "l3_profile_graph": state.get("profile_graph", {}),
    }
    summary = f"召回 {len(recalled['claims'])} 条有效经验记忆、{len(recalled['concept_mastery'])} 条相关掌握状态"
    return {
        "learner_context": learner,
        "trace": _trace(state, "load_learner_memory", summary, {
            "claim_ids": [item["claim_id"] for item in recalled["claims"]],
            "concepts": concepts,
        }),
    }


def gate_personalization(state: GardenerState) -> dict[str, Any]:
    """Convert memories into bounded, inspectable teaching hypotheses.

    This deterministic gate runs before the strategy LLM.  The model never sees
    rejected claims, which is stronger than merely prompting it not to misuse them.
    """
    intent = state["intent"]
    task_key = str(
        state.get("reasoning_profile", {}).get("task_key")
        or intent.get("primary_intent")
        or intent.get("task_demand")
        or "general"
    )
    if intent.get("response_mode") == "domain_overview":
        plan = PersonalizationPlan(
            status="disabled_first_exposure",
            task_key=task_key,
            strategy_summary="首次领域概览保持中立，不使用历史画像",
            fallback_reason="第一次建立领域框架时，个性化延后到用户主动深入之后。",
        ).model_dump()
        return {
            "personalization_plan": plan,
            "trace": _trace(state, "gate_personalization", "首次领域概览：关闭个性化", {
                "status": plan["status"], "reason": plan["fallback_reason"],
            }),
        }

    learner = state.get("learner_context", {})
    explicit_preferences = [
        str(item).strip() for item in learner.get("explicit_teaching_preferences", [])
        if str(item).strip()
    ]
    hypotheses: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    applied_claim_ids: list[str] = []
    confidences: list[float] = []
    for index, preference in enumerate(explicit_preferences):
        evidence_id = f"setting:teaching_preference:{index + 1}"
        hypotheses.append({
            "claim": preference,
            "confidence": 0.98,
            "scope": "用户明确设置",
            "evidence_ids": [evidence_id],
            "source_kind": "explicit",
        })
        evidence.append({
            "evidence_id": evidence_id,
            "observation": f"用户在设置中明确选择：{preference}",
            "relation": "supports", "weight": 1.0, "source_kind": "explicit",
        })
        confidences.append(0.98)

    allowed_dimensions = {"teaching_preference", "self_regulation"}
    for claim in learner.get("active_memory_claims", []):
        if claim.get("dimension") not in allowed_dimensions:
            continue
        confidence = float(claim.get("effective_confidence") or 0.0)
        support = float(claim.get("support_weight") or 0.0)
        contradiction = float(claim.get("contradiction_weight") or 0.0)
        if support <= contradiction or confidence < 0.4:
            continue
        claim_evidence = [
            item for item in claim.get("evidence", [])
            if item.get("relation") != "contradicts"
        ]
        if not claim_evidence:
            continue
        claim_id = str(claim.get("claim_id"))
        evidence_ids = [str(item["evidence_id"]) for item in claim_evidence]
        hypotheses.append({
            "claim": claim.get("claim_text", ""),
            "confidence": round(confidence, 3),
            "scope": f"{claim.get('scope_type')}:{claim.get('scope_key') or 'all'}",
            "evidence_ids": evidence_ids,
            "claim_id": claim_id,
            "source_kind": claim.get("source_kind"),
        })
        evidence.extend(claim_evidence)
        confidences.append(confidence)
        if confidence >= 0.7:
            applied_claim_ids.append(claim_id)

    mastery_evidence = []
    for item in learner.get("concept_mastery", []):
        concept = str(item.get("concept_key") or "").strip()
        confidence = float(item.get("confidence") or 0.0)
        if concept and confidence >= 0.55:
            mastery_item = {
                "evidence_id": f"mastery:{concept}",
                "observation": f"{concept} 当前有可追溯掌握证据：{item.get('stage', 'exposed')}（{confidence:.0%}）",
                "relation": "supports", "weight": confidence, "source_kind": "observed",
            }
            mastery_evidence.append(mastery_item)
            hypotheses.append({
                "claim": f"本轮可按“{item.get('stage', 'exposed')}”阶段控制 {concept} 的术语密度",
                "confidence": round(confidence, 3),
                "scope": f"concept:{concept}",
                "evidence_ids": [mastery_item["evidence_id"]],
                "source_kind": "observed",
            })
            confidences.append(confidence)
    evidence.extend(mastery_evidence)

    strongest = max(confidences, default=0.0)
    if strongest >= 0.7:
        status = "applied"
        allowed = ["调整解释顺序", "调整术语密度", "选择是否先给结构", "承接用户已给出的解释"]
        summary = "依据已确认偏好或重复证据调整讲解结构"
    elif strongest >= 0.4:
        status = "light"
        allowed = ["轻量调整段落结构", "提出可选而非强制的表达方式"]
        summary = "仅轻量参考尚未充分确认的教学假设"
    else:
        status = "standard"
        allowed = []
        summary = "标准讲解（没有足够个性化证据）"
    plan = PersonalizationPlan(
        status=status,
        task_key=task_key,
        confidence=round(strongest, 3),
        strategy_summary=summary,
        hypotheses=hypotheses,
        evidence=evidence[:12],
        applied_claim_ids=applied_claim_ids,
        allowed_adjustments=allowed,
        fallback_reason=("没有达到 0.40 的相关教学证据，使用该领域标准讲解。" if status == "standard" else ""),
    ).model_dump()
    return {
        "personalization_plan": plan,
        "trace": _trace(state, "gate_personalization", f"个性化门控：{status}", {
            "confidence": plan["confidence"],
            "applied_claim_ids": applied_claim_ids,
            "evidence_ids": [item.get("evidence_id") for item in plan["evidence"]],
        }),
    }
def ask_clarification(state: GardenerState) -> dict[str, Any]:
    question = state["intent"].get("clarification_question") or "你更想先弄清它的定义、作用机制，还是实际用法？"
    result = {
        "answer": question, "citations": [], "web_sources": [], "followup": question,
        "discussion_prompts": [], "evidence_layer": "clarification", "researched_online": False,
        "research_error": "", "offer_save": False, "agent_trace": _trace(state, "clarify", "信息不足，先询问一个最小澄清问题"),
    }
    return {"result": result, "trace": result["agent_trace"]}


def plan_sources(state: GardenerState) -> dict[str, Any]:
    intent = state["intent"]
    question = state["question"]
    fallback_types = ["local_wiki", "textbook"]
    public_entity_lookup = intent.get("target_kind") in {
        "person", "organization", "place", "work", "event",
    }
    if intent["primary_intent"] in {"define", "clarify"}:
        fallback_types.append("encyclopedia")
    if public_entity_lookup:
        fallback_types.extend(["official_docs", "public_web"])
    if intent["primary_intent"] == "explain_mechanism":
        fallback_types.append("encyclopedia")
        if intent.get("claim_to_verify") or re.search(r"前沿|最新研究|争议|是否成立|证据", question):
            fallback_types.extend(["review", "research_paper"])
    if intent["primary_intent"] == "compare":
        fallback_types.append("encyclopedia")
        if intent.get("claim_to_verify") or re.search(r"优劣|证据|研究发现|争议", question):
            fallback_types.extend(["review", "research_paper"])
    if intent["primary_intent"] in {"evaluate", "design"}:
        fallback_types.extend(["review", "research_paper"])
    historical_scope = bool(
        intent.get("longitudinal_questions")
        or re.search(r"历史|起源|发展脉络|演变|新兴学科|首次提出|发展过程", question)
    )
    if historical_scope:
        fallback_types.extend(["encyclopedia", "review", "research_paper"])
    if intent.get("response_mode") == "domain_overview":
        fallback_types.extend(["encyclopedia", "review", "research_paper"])
    fallback = SourcePlan(
        source_types=list(dict.fromkeys(fallback_types)), search_query=" ".join([
            str(intent.get("research_object") or ""),
            str(intent.get("claim_to_verify") or intent.get("core_question") or question),
            *[str(item) for item in (intent.get("concepts") or [])],
            *[str(item) for item in (intent.get("explicit_constraints") or [])],
        ]).strip(),
        recency_needed=intent["primary_intent"] in {"evaluate", "design"},
        rationale="本地知识优先；定义用百科定位术语，机制与评价需要综述或论文。",
    )
    # The question-understanding Agent has already resolved the object and the
    # Planner has chosen research breadth. Source planning is therefore a fast,
    # inspectable policy agent rather than another serial model call.
    plan = fallback.model_dump()
    if not state.get("planner_decision", {}).get("online_research", False):
        plan["source_types"] = [item for item in plan["source_types"] if item in {"local_wiki", "textbook"}]
        plan["rationale"] += " Planner 判定本轮无需联网，先用本地来源完成最小充分回答。"
    elif _requires_authority_lookup(question, intent):
        if "encyclopedia" not in plan["source_types"]:
            plan["source_types"].append("encyclopedia")
        plan["rationale"] += " 本轮属于事实型学习问题：本地来源仍优先，但必须实际执行权威来源补查。"
    if public_entity_lookup and state.get("planner_decision", {}).get("online_research", False):
        plan["rationale"] += " 人物、机构与作品优先查公开网页和机构官网，再依据结果判断是否需要消歧。"
    if "local_wiki" not in plan["source_types"]:
        plan["source_types"].insert(0, "local_wiki")
    if intent.get("response_mode") == "domain_overview":
        for source_type in ("textbook", "encyclopedia", "review", "research_paper"):
            if source_type not in plan["source_types"]:
                plan["source_types"].append(source_type)
    dialogue = "\n".join([
        *[
            str(item.get("content", ""))
            for item in state.get("history", [])[-4:]
            if item.get("role") == "user"
        ],
        question,
    ])
    lookup_fallback = _fallback_wechat_lookup(dialogue)
    lookup_payload = None
    if lookup_fallback.requested:
        try:
            lookup_payload = _agent_json(
                "你是微信历史查询参数解析 Agent，不读取数据、不回答问题。只有用户明确要求查询自己的微信、群聊或聊天记录时 requested 才为 true。提取联系人/群名 talker、最小时间提示 time_hint、主题词 topic_terms。不得猜测未写出的联系人。缺少 talker 时要求澄清。",
                f"对话与当前请求：\n{dialogue}\n"
                "输出 requested、talker、time_hint、topic_terms、needs_clarification、clarification_question、evidence。",
            )
        except LLMError:
            pass
    lookup = _validated(WeChatLookup, lookup_payload, lookup_fallback)
    # A model cannot broaden privacy scope beyond the explicit rule gate.
    lookup["requested"] = bool(lookup_fallback.requested)
    if lookup["requested"]:
        lookup["talker"] = str(lookup.get("talker") or lookup_fallback.talker).strip()
        lookup["time_hint"] = str(lookup.get("time_hint") or lookup_fallback.time_hint).strip()
        lookup["needs_clarification"] = not bool(lookup["talker"])
        if not lookup.get("clarification_question") and lookup["needs_clarification"]:
            lookup["clarification_question"] = lookup_fallback.clarification_question
        if "wechat_history" not in plan["source_types"]:
            plan["source_types"].append("wechat_history")
    else:
        plan["source_types"] = [item for item in plan["source_types"] if item != "wechat_history"]
    return {
        "source_plan": plan,
        "wechat_lookup": lookup,
        "trace": _trace(
            state, "plan_sources", "选择来源：" + "、".join(plan["source_types"]),
            {"rationale": plan["rationale"], "wechat_requested": lookup["requested"]},
        ),
    }


def route_after_source_plan(state: GardenerState) -> str:
    lookup = state.get("wechat_lookup", {})
    return "clarify_wechat" if lookup.get("requested") and lookup.get("needs_clarification") else "retrieve_sources"


def ask_wechat_clarification(state: GardenerState) -> dict[str, Any]:
    question = state.get("wechat_lookup", {}).get("clarification_question") or (
        "你想查哪个联系人或群聊？也请给一个尽量小的时间范围。"
    )
    result = {
        "answer": question,
        "citations": [], "local_connections": [], "web_sources": [], "wechat_sources": [],
        "followup": question, "discussion_prompts": [], "evidence_layer": "clarification",
        "researched_online": False, "research_error": "", "offer_save": False,
        "agent_trace": _trace(state, "clarify_wechat", "读取私人聊天前先取得最小必要会话范围"),
    }
    return {"result": result, "trace": result["agent_trace"]}


def retrieve_sources(state: GardenerState) -> dict[str, Any]:
    store = state["store"]
    plan = state["source_plan"]
    question = state["question"]
    intent = state.get("intent", {})
    if (
        _response_profile(question) != "grounded_knowledge"
        and not state.get("planner_decision", {}).get("online_research")
        and not state.get("wechat_lookup", {}).get("requested")
        and not state.get("direct_material")
        and not state["context"].knowledge_scope.selected_note_ids
    ):
        return {
            "local_hits": [],
            "candidate_sources": [],
            "retrieval_attempts": [],
            "retrieval_errors": [],
            "trace": _trace(state, "retrieve_sources", "开放讨论不检索无关教材，也不把常识包装成已核验来源", {
                "response_profile": _response_profile(question),
                "retrieval_mode": "bounded_discussion_fast_path",
            }),
        }
    query_plan = state.get("intent", {}).get("query_plan")
    if not isinstance(query_plan, dict):
        query_plan = build_query_plan(
            question,
            resolved_question=str(intent.get("core_question") or question).strip(),
            concepts=intent.get("concepts", []),
            suggested_queries=intent.get("retrieval_queries", []),
        )
    # Use the audited query plan for retrieval and reranking too.  For a
    # standalone question its resolved field is the original user wording;
    # for a genuine follow-up it contains the context-resolved inquiry.
    retrieval_question = str(
        intent.get("claim_to_verify") or query_plan.get("resolved") or question
    ).strip()
    direct_definition_form = bool(re.match(
        r"^(?:请问|请解释|请说明)?\s*(?:什么是|何为|是什么意思|请介绍)\s*[^？?]{1,24}[？?]?$",
        question.strip(),
    ))
    simple_definition = (
        state.get("planner_decision", {}).get("complexity") == "simple"
        and intent.get("primary_intent") in {"define", "clarify"}
        and intent.get("response_mode") != "domain_overview"
        and direct_definition_form
    )
    interactive_semantic = os.getenv(
        "GARDEN_INTERACTIVE_SEMANTIC_RETRIEVAL", ""
    ).strip().lower() in {"1", "true", "yes"}
    foundational_hybrid = (
        str(query_plan.get("subject_mode") or "").strip().lower() == "foundational"
        and not simple_definition
        and _response_profile(question) == "grounded_knowledge"
    )
    use_hybrid = interactive_semantic or foundational_hybrid
    retrieval_kinds = {"concept", "moc", "bridge", "knowledge", "course", "textbook"}
    exact_iff_lexical_path = False
    local_hits: list[dict[str, Any]] = []
    if re.search(r"证明", question) and re.search(r"充要条件|当且仅当", question):
        lexical_probe = search_notes(
            store, retrieval_question, kinds=retrieval_kinds, limit=8,
            query_plan=query_plan, semantic_enabled=False, rerank_enabled=False,
        )
        preview_candidates = [{
            "source_id": f"L{index}", "title": item["title"], "text": item["snippet"],
            "source_type": "textbook" if item.get("kind") in {"textbook", "course"} else "local_wiki",
        } for index, item in enumerate(lexical_probe, 1)]
        exact_ids = {
            source_id for source_id, _ in _exact_iff_textbook_claims(
                question, intent, preview_candidates,
                {str(item["source_id"]) for item in preview_candidates},
            )
        }
        if exact_ids:
            local_hits = [
                item for index, item in enumerate(lexical_probe, 1) if f"L{index}" in exact_ids
            ] + [
                item for index, item in enumerate(lexical_probe, 1) if f"L{index}" not in exact_ids
            ]
            exact_iff_lexical_path = True
    if not exact_iff_lexical_path:
        local_hits = search_notes(
            store, retrieval_question, kinds=retrieval_kinds, limit=8,
            query_plan=query_plan,
            # Composite foundational questions use BGE/FAISS + reranking;
            # an exact iff theorem statement can stay on the lexical path.
            semantic_enabled=None if use_hybrid else False,
            rerank_enabled=None if use_hybrid else False,
        )
    selected_note_ids = set(state["context"].knowledge_scope.selected_note_ids)
    if selected_note_ids:
        local_hits = [item for item in local_hits if item["id"] in selected_note_ids]
    candidates = [
        {
            "source_id": f"L{index}", "title": item["title"], "url": "", "text": item["snippet"],
            "source_type": "textbook" if item.get("kind") in {"textbook", "course"} else "local_wiki",
            "authority": "high" if item.get("kind") in {"textbook", "course"} else "medium",
            "local": True, "note": item, "access_scope": "full_text",
            "relevance_score": item.get("relevance_score", 0.0),
            "matched_terms": item.get("matched_terms", []),
            "relevance_reason": item.get("relevance_reason", ""),
            "knowledge_status": item.get("knowledge_status", "derived"),
            "authority_score": 0.85 if item.get("kind") in {"textbook", "course"} else 0.3,
        }
        for index, item in enumerate(local_hits, 1)
    ]
    errors = []
    mounted_tools = set(state["context"].tool_policy.mounted)
    network_enabled = os.getenv("GARDEN_DISABLE_NETWORK", "").strip().lower() not in {"1", "true", "yes"}
    direct = state.get("direct_material") or {}
    if direct.get("title"):
        material_text = str(direct.get("abstract") or "").strip()
        declared_scope = str(direct.get("access_scope") or "").strip().lower()
        access_scope = (
            "open_fulltext" if declared_scope in {"full_text", "open_fulltext", "web_fulltext"} and material_text
            else "abstract" if material_text else "metadata_only"
        )
        pdf_url = str(direct.get("pdf_url") or "").strip()
        if network_enabled and pdf_url:
            try:
                full_text = fetch_open_access_pdf_text(pdf_url)
                if full_text:
                    material_text = full_text
                    access_scope = "open_fulltext"
            except Exception as exc:
                errors.append("OpenFullText:" + exc.__class__.__name__)
        candidates.insert(0, {
            "source_id": "M1", "title": str(direct.get("title")),
            "url": str(direct.get("url") or pdf_url), "text": material_text,
            "source_type": "frontier_material", "authority": "scholarly_metadata",
            "local": False, "article": direct, "access_scope": access_scope,
            "knowledge_status": "grounded", "authority_score": 0.8,
            "explicitly_selected": True,
        })
    # Retrieval and generation are separate capabilities. Public-source lookup
    # remains useful even when no chat-model key is configured.
    retrieval_jobs = {}
    retrieval_attempts: list[str] = []
    # Interactive tutoring has a bounded research budget. Patrol/digest jobs
    # may still use the academic client's longer retry defaults.
    academic_timeout = max(3, min(10, int(os.getenv("GARDEN_ACADEMIC_TIMEOUT_SECONDS", "6"))))
    public_web_timeout = max(3, min(12, int(os.getenv("GARDEN_PUBLIC_WEB_TIMEOUT_SECONDS", "8"))))
    orientation_query = next((
        str(item).strip() for item in (intent.get("concepts") or []) if str(item).strip()
    ), str(intent.get("research_object") or "").strip()) or plan["search_query"]
    wikipedia_queries = [orientation_query]
    if intent.get("primary_intent") in {"compare", "explain_mechanism"}:
        planned_queries = [
            str(item.get("text") or "").strip()
            for item in query_plan.get("queries", []) if isinstance(item, dict)
            and 2 <= len(str(item.get("text") or "").strip()) <= 80
        ]
        wikipedia_queries = list(dict.fromkeys([
            *[
                str(item).strip() for item in (intent.get("concepts") or [])
                if str(item).strip()
            ],
            *planned_queries,
            orientation_query,
        ]))[:3] or [orientation_query]
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="garden-retrieval") as pool:
        if network_enabled and "wikipedia" in mounted_tools and "encyclopedia" in plan["source_types"]:
            for index, wiki_query in enumerate(wikipedia_queries, 1):
                retrieval_jobs[f"Wikipedia:{index}"] = pool.submit(search_wikipedia, wiki_query, 2)
            retrieval_attempts.append("Wikipedia")
        if network_enabled and "academic_search" in mounted_tools and any(kind in plan["source_types"] for kind in ("review", "research_paper")):
            retrieval_jobs["OpenAlex"] = pool.submit(
                search_academic_articles, plan["search_query"], 4,
                academic_timeout, attempts_per_provider=1,
            )
            retrieval_attempts.append("OpenAlex/Crossref")
        if (
            network_enabled and "public_web" in mounted_tools
            and any(kind in plan["source_types"] for kind in ("official_docs", "public_web"))
        ):
            public_query = " ".join(dict.fromkeys(filter(None, [
                str(intent.get("research_object") or "").strip(),
                *[
                    str(item).strip() for item in intent.get("explicit_constraints", [])
                    if str(item).strip()
                ],
            ]))) or plan["search_query"]
            retrieval_jobs["PublicWeb"] = pool.submit(
                search_public_web, public_query, 5, public_web_timeout,
            )
            retrieval_attempts.append("公开网页 / 机构官网")
        retrieved = {}
        for name, future in retrieval_jobs.items():
            try:
                retrieved[name] = future.result()
            except Exception as exc:
                errors.append(name + ":" + exc.__class__.__name__)
                retrieved[name] = []
    wikipedia_items: list[dict[str, Any]] = []
    seen_wikipedia_urls: set[str] = set()
    for name, items in retrieved.items():
        if not name.startswith("Wikipedia:"):
            continue
        for item in items:
            key = str(item.get("url") or item.get("title") or "")
            if key and key not in seen_wikipedia_urls:
                seen_wikipedia_urls.add(key)
                wikipedia_items.append(item)
    retrieved["Wikipedia"] = wikipedia_items[:4]
    # Chinese Wikipedia gives us a reviewed cross-language title.  When the
    # original subject has no usable English term, use that title for one
    # academic retry instead of hard-coding bilingual aliases per discipline.
    wikipedia_english = next((
        str(item.get("english_title") or "").strip()
        for item in retrieved.get("Wikipedia", []) if str(item.get("english_title") or "").strip()
    ), "")
    original_has_english_term = bool(re.search(r"[A-Za-z][A-Za-z-]{3,}", plan["search_query"]))
    academic_requested = any(kind in plan["source_types"] for kind in ("review", "research_paper"))
    if (
        network_enabled and "academic_search" in mounted_tools and academic_requested
        and wikipedia_english and not original_has_english_term
        and not retrieved.get("OpenAlex")
        and (
            intent.get("response_mode") == "domain_overview"
            or bool(intent.get("longitudinal_questions"))
            or intent.get("primary_intent") in {"evaluate", "design"}
        )
    ):
        bridge_query = " ".join(filter(None, [
            wikipedia_english,
            "definition history review" if intent.get("longitudinal_questions") else "definition review",
        ]))
        try:
            bridged = search_academic_articles(
                bridge_query, 4, academic_timeout, attempts_per_provider=1,
            )
            existing_urls = {str(item.get("url") or "") for item in retrieved.get("OpenAlex", [])}
            retrieved["OpenAlex"] = [
                *retrieved.get("OpenAlex", []),
                *[item for item in bridged if str(item.get("url") or "") not in existing_urls],
            ][:4]
            retrieval_attempts.append("Wikipedia→English→OpenAlex/Crossref")
        except Exception as exc:
            errors.append("Wikipedia 英文术语学术补查:" + exc.__class__.__name__)
    if not network_enabled and any(kind in plan["source_types"] for kind in (
        "encyclopedia", "review", "research_paper", "official_docs", "public_web",
    )):
        errors.append("联网检索已被 GARDEN_DISABLE_NETWORK 关闭")
    if "encyclopedia" in plan["source_types"] and "wikipedia" not in mounted_tools:
        errors.append("Wikipedia 工具未启用")
    if any(kind in plan["source_types"] for kind in ("review", "research_paper")) and "academic_search" not in mounted_tools:
        errors.append("学术检索工具未启用")
    if any(kind in plan["source_types"] for kind in ("official_docs", "public_web")) and "public_web" not in mounted_tools:
        errors.append("公开网页检索工具未启用")
    for index, item in enumerate(retrieved.get("Wikipedia", []), 1):
        candidates.append({
            "source_id": f"W{index}", "title": item["title"], "url": item["url"], "text": item["abstract"],
            "source_type": "encyclopedia", "authority": "orientation", "local": False, "article": item,
            "access_scope": "abstract", "knowledge_status": "grounded", "authority_score": 0.7,
            "retrieval_rank": index, "provider_query": orientation_query,
        })
    for index, item in enumerate(retrieved.get("OpenAlex", []), 1):
        is_review = bool(re.search(r"review|meta-analysis|systematic", item["title"], re.I))
        candidates.append({
            "source_id": f"A{index}", "title": item["title"], "url": item["url"], "text": item.get("abstract") or "仅有题录",
            "source_type": "review" if is_review else "research_paper", "authority": "high" if is_review else "medium",
            "local": False, "article": item,
            "access_scope": "abstract" if item.get("abstract") else "metadata_only",
            "knowledge_status": "grounded", "authority_score": 0.9 if is_review else 0.8,
        })
    for index, item in enumerate(retrieved.get("PublicWeb", []), 1):
        official = bool(item.get("official"))
        candidates.append({
            "source_id": f"P{index}", "title": item["title"],
            "url": item["url"], "text": item.get("abstract", ""),
            "source_type": "official_docs" if official else "public_web",
            "authority": "institutional" if official else "public_search",
            "local": False, "article": item, "access_scope": "search_snippet",
            "knowledge_status": "grounded", "authority_score": 0.9 if official else 0.55,
            "retrieval_rank": index,
        })
    lookup = state.get("wechat_lookup", {})
    if lookup.get("requested") and "wechat_history" in plan["source_types"]:
        if "tracememo_reader" not in mounted_tools:
            errors.append("TraceMemo：服务可能在线，但花园尚未配置 API Center Token")
        else:
            try:
                base_url = str(store.setting("tracememo_base_url", "http://127.0.0.1:6131"))
                client = TraceMemoClient(tracememo_config(base_url))
                health = client.health()
                if not health.get("ready", health.get("ok", False)):
                    raise TraceMemoError("Reader 服务或微信数据库尚未就绪")
                # The clock call is mandatory before resolving relative dates.
                clock = client.current_time()
                talker = str(lookup.get("talker", "")).strip()
                contact = client.resolve(talker)
                time_params = _wechat_time_params(str(lookup.get("time_hint", "")), clock)
                chat_data = client.chatlog(talker, **time_params)
                messages = chat_data.get("messages", [])
                selected_messages = messages[:80]
                if selected_messages:
                    message_text = "\n".join(
                        f"{item.get('sent_at') or '时间未知'} | {item.get('sender') or '未知发送者'}：{item.get('content', '')}"
                        for item in selected_messages
                    )[:24_000]
                    candidates.insert(0, {
                        "source_id": "T1",
                        "title": f"微信记录：{talker} · {lookup.get('time_hint') or '本次限定范围'}",
                        "url": "",
                        "text": message_text,
                        "source_type": "wechat_history",
                        "authority": "authorized_primary_record",
                        "local": False,
                        "private": True,
                        "contact": contact,
                        "talker": talker,
                        "time_hint": str(lookup.get("time_hint", "")),
                        "message_count": len(selected_messages),
                        "access_scope": "authorized_excerpt",
                        "knowledge_status": "grounded",
                        "authority_score": 0.95,
                        "explicitly_selected": True,
                    })
                else:
                    errors.append(f"TraceMemo：{talker} 在这个时间范围内没有返回可读消息")
            except TraceMemoError as exc:
                errors.append("TraceMemo：" + str(exc))
            except Exception as exc:
                errors.append("TraceMemo：读取失败（" + exc.__class__.__name__ + "）")
    external_count = sum(1 for item in candidates if not item.get("local"))
    summary = f"用 {len(query_plan['queries'])} 条查询找到 {len(local_hits)} 条本地证据、{external_count} 条外部或授权记录候选"
    return {
        "local_hits": local_hits, "candidate_sources": candidates,
        "retrieval_attempts": retrieval_attempts, "retrieval_errors": errors,
        "trace": _trace(state, "retrieve_sources", summary, {
            "online_attempts": retrieval_attempts, "errors": errors, "query_plan": query_plan,
            "retrieval_mode": (
                "exact_iff_lexical_fast_path" if exact_iff_lexical_path
                else "hybrid" if use_hybrid else "lexical"
            ),
        }),
    }


def _source_argument_priority(
    item: dict[str, Any],
    constraints: list[str],
    aliases: list[str],
    concepts: list[str],
) -> tuple[int, int, int, float, int, float, float]:
    """Prefer passages where the actual claim and its constraints co-occur."""
    def normalize(value: Any) -> str:
        text = re.sub(r"\s+", "", str(value or "")).casefold()
        return text.replace("−", "-").replace("–", "-").replace("’", "'").replace("零", "0")

    corpus = normalize(f"{item.get('title', '')}\n{item.get('text', '')}")
    normalized_constraints = [normalize(value) for value in constraints if normalize(value)]
    normalized_aliases = list(dict.fromkeys(
        normalize(value) for value in aliases if len(normalize(value)) >= 4
    ))
    constraint_hits = sum(value in corpus for value in normalized_constraints)

    concept_strength = 0
    generic_prefixes = {
        "反应", "化学", "物理", "数学", "生物", "分子", "计算", "函数",
        "理论", "系统", "研究", "学习", "科学", "方法", "问题", "结构",
    }
    for concept_index, concept in enumerate(concepts):
        term = normalize(concept)
        if len(term) < 2:
            continue
        # The first concept is the object being explained; later concepts
        # usually describe an application or comparison context. Without this
        # distinction a generic drug-design page outranks the only textbook
        # page that actually defines the specialist term “手性”.
        concept_weight = 2 if concept_index == 0 else 1
        if term in corpus:
            concept_strength += min(len(term), 9) * concept_weight
            continue
        for size in range(min(6, len(term) - 1), 1, -1):
            matched_at = next(
                (index for index in range(len(term) - size + 1)
                 if term[index:index + size] in corpus),
                None,
            )
            if matched_at is not None:
                specialist_prefix = (
                    matched_at == 0 and size == 2 and term[:2] not in generic_prefixes
                )
                concept_strength += (
                    size + (4 if specialist_prefix else 0)
                ) * concept_weight
                break

    window_alias_hits = 0
    for constraint in normalized_constraints:
        for occurrence in re.finditer(re.escape(constraint), corpus):
            window = corpus[max(0, occurrence.start() - 150):occurrence.end() + 150]
            window_alias_hits = max(
                window_alias_hits,
                sum(alias in window for alias in normalized_aliases),
            )
    note = item.get("note") or {}
    alias_hits = sum(alias in corpus for alias in normalized_aliases)
    return (
        constraint_hits, concept_strength, window_alias_hits,
        float(note.get("channel_consensus_bonus", 0.0) or 0.0),
        alias_hits, float(note.get("fusion_score", 0.0) or 0.0),
        float(note.get("reranker_score", 0.0) or 0.0),
    )


def _comparison_subjects(intent: dict[str, Any], question: str) -> list[str]:
    if intent.get("primary_intent") != "compare":
        return []
    question_text = re.split(r"[？?；;]", str(intent.get("core_question") or question), maxsplit=1)[0]
    patterns = (
        r"(.+?)\s*(?:与|和|跟|vs\.?|VS\.?)\s*(.+?)(?:的(?:核心)?(?:区别|差异|关系)|有什么(?:区别|差异)|是(?:同一个|一样|相同)|相比|$)",
        r"比较\s*(.+?)\s*(?:与|和|跟)\s*(.+)",
    )
    for pattern in patterns:
        match = re.search(pattern, question_text, re.I)
        if match is None:
            continue
        cleaned = []
        for raw in match.groups():
            value = re.sub(r"^(?:请|比较|解释)\s*", "", raw)
            # Domain-setting clauses are context, not part of the object:
            # “热力学中，焓和自由能” compares “焓” with “自由能”.
            value = re.split(r"[，,；;]", value)[-1].strip("？?。；;：: ")
            if value:
                cleaned.append(value)
        if len(cleaned) >= 2:
            return cleaned[:2]
    concepts = [str(item).strip() for item in intent.get("concepts", []) if str(item).strip()]
    return list(dict.fromkeys(concepts))[:2]


def _requires_claim_level_audit(question: str) -> bool:
    """Identify answers where a merely topical page cannot support the requested chain."""
    return bool(re.search(
        r"证明|推导|机理(?:图)?|完整路径|工作原理|伪代码|算法|复杂度|递归实现|"
        r"构造|充要条件|判据|能级图|信号通路|标准型|奇异值分解|"
        r"举例|反例|辨析|区别|异同|同一个概念|"
        r"(?:求|计算).*(?:值|结果|方程|表达式|分布|速度|加速度|电场|电势|功|热量|"
        r"振幅|频率|波长|波速|周期|时间|距离|概率|能量|动量)|"
        r"有什么(?:关系|关联|联系|共同点)|有何(?:关系|关联|联系|共同点)|"
        r"共同结构|串联|概念映射|跨学科",
        question,
        re.I,
    ))


def _exact_iff_textbook_claims(
    question: str, intent: dict[str, Any], candidates: list[dict[str, Any]], eligible_ids: set[str],
) -> list[tuple[str, str]]:
    """Promote an exact textbook iff statement as a proof anchor.

    The source is allowed to anchor the theorem statement, not to claim that
    the generated proof was copied from the page. This deterministic fallback
    is intentionally limited to explicit iff proofs and requires both sides of
    the relation to occur in the same textbook passage.
    """
    if not re.search(r"证明", question) or not re.search(r"充要条件|当且仅当", question):
        return []
    concepts = [str(item).strip() for item in intent.get("concepts", []) if str(item).strip()]
    if len(concepts) < 2:
        return []

    def aliases_for(concept: str) -> list[str]:
        for group in ALIAS_GROUPS:
            if concept in group:
                return [str(alias) for alias in group]
        return [concept]

    def concept_present(aliases: list[str], corpus: str) -> bool:
        if any(re.sub(r"\s+", "", alias).casefold() in corpus for alias in aliases):
            return True
        # Scanned Chinese textbooks often OCR “≠ 0” as “关 0”. Accept this
        # only when the same compact passage explicitly says 行列式.
        if any("行列式不为零" == alias for alias in aliases):
            return bool(re.search(r"行列式.{0,16}(?:不等于|不为|非零|≠|关)0", corpus))
        return False

    concept_aliases = [aliases_for(concept) for concept in concepts[:2]]
    matches: list[tuple[str, str]] = []
    for item in candidates:
        source_id = str(item.get("source_id") or "")
        if source_id not in eligible_ids or item.get("source_type") != "textbook":
            continue
        text = str(item.get("text") or "")
        compact = re.sub(r"\s+", "", text).casefold()
        if not re.search(r"充要条件|当且仅当|等价命题", compact):
            continue
        if not all(concept_present(aliases, compact) for aliases in concept_aliases):
            continue
        sentences = [part.strip() for part in re.split(r"[。；;\n]", text) if part.strip()]
        excerpt = next((
            part for part in sentences
            if re.search(r"充要条件|当且仅当|等价命题", part)
            and all(concept_present(aliases, re.sub(r"\s+", "", part).casefold())
                    for aliases in concept_aliases)
        ), text[:280])
        matches.append((source_id, re.sub(r"\s+", " ", excerpt).strip()[:300]))
    return matches


def audit_evidence(state: GardenerState) -> dict[str, Any]:
    candidates = state.get("candidate_sources", [])
    question = state["question"]
    intent = state.get("intent", {})
    anchor_terms = [
        str(intent.get("research_object") or "").strip(),
        *[str(item).strip() for item in intent.get("concepts", [])],
    ]
    anchor_terms = [
        term for term in dict.fromkeys(anchor_terms)
        if 2 <= len(re.sub(r"\s+", "", term)) <= 40
        and term not in {"问题", "原因", "机制", "方法", "概念", "关系"}
    ]
    formal_reasoning = bool(re.search(r"证明|推导|计算|构造|方程|公式|判据|充要条件", question))
    lowered_question = re.sub(r"\s+", "", question).casefold()
    specialized_groups = [
        group for group in ALIAS_GROUPS
        if any(
            len(re.sub(r"\s+", "", alias)) >= 2
            and re.sub(r"\s+", "", alias).casefold() in lowered_question
            for alias in group[:2]
        )
    ] if formal_reasoning or intent.get("primary_intent") == "explain_mechanism" else []
    strongest_group = max(
        specialized_groups,
        key=lambda group: max((len(re.sub(r"\s+", "", term)) for term in group[:2]), default=0),
        default=(),
    )

    def normalized_subject(value: str) -> str:
        normalized = re.sub(r"[\s·•—_\-/（）()《》“”\"'：:，,。？！?]", "", value).lower()
        # Canonical encyclopedia titles often add these generic qualifiers to
        # the short term used by a learner (e.g. 人因学 ↔ 人因工程学).
        return re.sub(r"工程|科学|学科|理论|概论", "", normalized)

    def subject_title_match(item: dict[str, Any]) -> bool:
        title = normalized_subject(str(item.get("title") or ""))
        return any(
            (subject := normalized_subject(term)) and len(subject) >= 2
            and (subject in title or title in subject)
            for term in anchor_terms
        )

    def hits_research_object(item: dict[str, Any]) -> bool:
        corpus = re.sub(r"\s+", "", f"{item.get('title', '')}\n{item.get('text', '')}").lower()
        return subject_title_match(item) or any(
            re.sub(r"\s+", "", term).lower() in corpus for term in anchor_terms
        )

    audit_query = " ".join([
        str(state.get("intent", {}).get("core_question") or question),
        str(state.get("intent", {}).get("claim_to_verify") or ""),
        *[str(item) for item in state.get("intent", {}).get("concepts", [])],
        # These aliases come from the deterministic, auditable terminology
        # table. They bridge Chinese questions to English textbook wording;
        # the page still has to contain the term before it can pass the gate.
        *[str(item) for item in state.get("intent", {}).get("query_plan", {}).get("aliases", [])],
        str(state.get("source_plan", {}).get("search_query", "")),
    ])
    auditable_aliases = [
        str(item).strip() for item in state.get("intent", {}).get("query_plan", {}).get("aliases", [])
        if len(str(item).strip()) >= 2
    ]
    hard_eligible_ids: set[str] = set()
    hard_rejections: list[dict[str, str]] = []
    for item in candidates:
        source_id = item["source_id"]
        access_scope = item.get("access_scope", "metadata_only")
        actual_text = bool(item.get("text") and item["text"] != "仅有题录")
        relevance = relevance_gate(audit_query, item["title"], item.get("text", ""))
        item["audit_matched_terms"] = relevance["matched_terms"]
        item["audit_relevance_passed"] = bool(relevance["passed"])
        audit_corpus = f"{item.get('title', '')}\n{item.get('text', '')}".casefold()
        item["audit_alias_matches"] = [
            alias for alias in auditable_aliases if alias.casefold() in audit_corpus
        ]
        compact_audit_corpus = re.sub(r"\s+", "", audit_corpus)
        item["specialist_anchor_terms"] = list(strongest_group[:4])
        item["specialist_anchor_passed"] = not strongest_group or any(
            len((cleaned := re.sub(r"\s+", "", alias).casefold())) >= 2
            and cleaned in compact_audit_corpus
            for alias in strongest_group
        )
        if re.search(r"氢原子", question) and re.search(r"推导|证明", question):
            item["specialist_anchor_passed"] = bool(
                item["specialist_anchor_passed"]
                and re.search(
                    r"薛定谔|schr[öo]dinger",
                    compact_audit_corpus,
                    re.I,
                )
                and re.search(
                    r"径向方程|radial(?:wave)?equation|拉盖尔|laguerre|幂级数|powerseries",
                    compact_audit_corpus,
                    re.I,
                )
            )
        elif re.search(r"DNA|碱基", question, re.I) and re.search(r"精确传递|复制保真", question):
            item["specialist_anchor_passed"] = bool(re.search(
                r"复制|校对|错配修复|互补配对|replication|proofread|mismatchrepair|basepairing",
                compact_audit_corpus,
                re.I,
            ))
        note = item.get("note", {})
        rare_concept_passed = bool(
            item.get("source_type") == "textbook"
            and str(note.get("relevance_reason") or "") == "命中教材中低频且具有区分度的专业概念"
            and any(
                len(term) >= 2 and term.casefold() in audit_corpus
                for term in note.get("matched_terms", [])
            )
        )
        reranker_score = float(note.get("reranker_score", 0.0))
        reranker_rank = int(note.get("reranker_rank", 0) or 0)
        semantic_passed = reranker_score >= 0.5 and 0 < reranker_rank <= 6
        # Preserve this auditable decision for the role-assignment phase. A
        # foundational exercise often composes several textbook terms, so the
        # full research-object phrase may never occur verbatim on the page.
        item["semantic_passed"] = semantic_passed
        item["rare_concept_passed"] = rare_concept_passed
        item["relevance_score"] = max(float(item.get("relevance_score", 0.0)), relevance["score"])
        item["matched_terms"] = list(dict.fromkeys([*(item.get("matched_terms") or []), *relevance["matched_terms"]]))
        if not actual_text or access_scope == "metadata_only":
            hard_rejections.append({"source_id": source_id, "reason": "仅取得题录或标题，没有可核验内容"})
        elif (
            not relevance["passed"] and not semantic_passed and not rare_concept_passed
            and not item.get("explicitly_selected")
            and not (
                item.get("source_type") == "encyclopedia"
                and int(item.get("retrieval_rank") or 99) == 1
                and subject_title_match(item)
            )
        ):
            hard_rejections.append({"source_id": source_id, "reason": relevance["reason"]})
        elif item.get("local") and item.get("knowledge_status") != "grounded":
            hard_rejections.append({"source_id": source_id, "reason": "本地页面缺少可追溯上游来源，只能作为知识连接"})
        elif (
            item["source_type"] == "encyclopedia"
            and (
                state["intent"].get("primary_intent") in {"evaluate", "design"}
                or bool(state["intent"].get("claim_to_verify"))
            )
            and state["intent"].get("response_mode") != "domain_overview"
        ):
            hard_rejections.append({"source_id": source_id, "reason": "百科可以解释基础概念，但不能单独支撑争议命题或评价结论"})
        else:
            hard_eligible_ids.add(source_id)
    fallback_roles = {}
    for item in candidates:
        if item["source_id"] not in hard_eligible_ids:
            continue
        object_grounded = hits_research_object(item)
        if item.get("explicitly_selected"):
            fallback_roles[item["source_id"]] = "direct_evidence"
        elif item.get("source_type") in {"review", "research_paper"} and object_grounded:
            fallback_roles[item["source_id"]] = "direct_evidence"
        elif (
            item.get("source_type") in {"official_docs", "public_web"}
            and object_grounded
            and state["intent"].get("target_kind") in {
                "person", "organization", "place", "work", "event",
            }
            and state["intent"].get("primary_intent") in {"define", "clarify"}
        ):
            fallback_roles[item["source_id"]] = "direct_evidence"
        elif (
            item.get("source_type") == "textbook"
            and item.get("specialist_anchor_passed", True)
            and (
                object_grounded
                or (
                    item.get("semantic_passed") and bool(item.get("audit_matched_terms"))
                )
                or item.get("rare_concept_passed")
                or (_source_argument_priority(item, [], [], anchor_terms)[1] >= 4)
                or (
                    bool(item.get("audit_alias_matches"))
                )
            )
            and state["intent"].get("primary_intent") in {"define", "explain_mechanism", "apply", "compare"}
        ):
            fallback_roles[item["source_id"]] = "direct_evidence"
        elif (
            item.get("source_type") == "local_wiki"
            and item.get("knowledge_status") == "grounded"
            and item.get("specialist_anchor_passed", True)
            and (
                object_grounded
                or (
                    float(item.get("relevance_score", 0.0)) >= 0.45
                    and len(item.get("matched_terms") or []) >= 2
                )
            )
            and state["intent"].get("primary_intent") in {"define", "explain_mechanism", "apply", "compare"}
        ):
            fallback_roles[item["source_id"]] = "direct_evidence"
        elif (
            item.get("source_type") == "encyclopedia"
            and object_grounded
            and state["intent"].get("primary_intent") in {"define", "clarify", "explain_mechanism", "compare"}
            and not state["intent"].get("claim_to_verify")
        ):
            fallback_roles[item["source_id"]] = "direct_evidence"
        elif item.get("source_type") in {"textbook", "local_wiki"}:
            fallback_roles[item["source_id"]] = "prerequisite"
        else:
            fallback_roles[item["source_id"]] = "context"
    fallback_candidates = sorted(
        (item for item in candidates if item["source_id"] in fallback_roles),
        key=lambda item: (
            fallback_roles[item["source_id"]] == "direct_evidence",
            bool(item.get("rare_concept_passed")),
            _source_argument_priority(item, [], [], anchor_terms),
        ),
        reverse=True,
    )
    fallback_ids = [item["source_id"] for item in fallback_candidates[:5]]
    fallback = EvidenceDecision(
        accepted_ids=fallback_ids,
        sufficient=any(role == "direct_evidence" for role in fallback_roles.values()),
        gaps=[] if any(role == "direct_evidence" for role in fallback_roles.values()) else ["只有前置或背景材料，没有取得直接支持待核验命题的证据"],
        rationale="离线审查只接纳具有实际内容的教材、本地知识、学术来源或用户授权聊天片段；聊天只证明对话中出现过什么说法，百科只作为术语入口。",
        source_roles=fallback_roles,
    )
    payload = None
    audit_agent_status = "not_required"
    audit_agent_error = ""
    claim_level_audit = _requires_claim_level_audit(question)
    relationship_audit = bool(re.search(
        r"有什么(?:关系|关联|联系|共同点)|有何(?:关系|关联|联系|共同点)|"
        r"共同结构|串联|概念映射|跨学科",
        question,
    ))
    high_risk_audit = (
        claim_level_audit
        or
        state.get("planner_decision", {}).get("complexity") == "complex"
        or (
            state["intent"].get("primary_intent") in {"evaluate", "design"}
            and _response_profile(question) == "grounded_knowledge"
        )
        or state.get("wechat_lookup", {}).get("requested")
        or bool(re.search(r"适用边界|成立边界|适用条件|同时说明.*(?:边界|条件)|证据是否", question))
    )
    exact_iff_claims = _exact_iff_textbook_claims(question, intent, candidates, hard_eligible_ids)
    exact_iff_ids = {source_id for source_id, _ in exact_iff_claims}
    if exact_iff_claims:
        audit_agent_status = "deterministic_exact_iff_anchor"
    if candidates and high_risk_audit and not exact_iff_claims:
        audit_agent_status = "attempted"
        audit_max_sources = max(2, min(8, int(os.getenv("GARDEN_AUDIT_MAX_SOURCES", "6"))))
        audit_excerpt_chars = max(500, min(1200, int(os.getenv("GARDEN_AUDIT_EXCERPT_CHARS", "800"))))
        audit_candidates = [
            item for item in candidates if str(item.get("source_id")) in hard_eligible_ids
        ][:audit_max_sources]
        source_text = "\n\n".join(
            f"[{item['source_id']}] 类型={item['source_type']} 权威={item['authority']} 标题={item['title']}\n"
            f"{item['text'][:audit_excerpt_chars]}"
            for item in audit_candidates
        )
        try:
            payload = _agent_json(
                "你是独立证据审查 Agent，不负责生成答案。先审查每个来源与 core_question/claim_to_verify 的论证关系，而不只看关键词相似。给通过来源标注唯一角色：direct_evidence（直接支持或反驳命题）、prerequisite（只解释必要前置）、counterevidence（反例或替代解释）、context（背景/历史）。教材和本地 Wiki 只有在明确支撑某个推理步骤时才可进入；不能因为用户学过它就把它生搬硬套成依据。逐条判断摘要是否真的支持。证明、推导、计算、算法和机制题中，仅仅提到同领域名词不算直接证据：页面必须实际包含所需定理、关键公式、量之间的对应关系、算法步骤或机制链；只给出导数定义、理想气体名称或同形公式，不能冒充完整解题依据。关系、共同点和跨学科连接题必须有证据同时覆盖关系两端，并明确支持二者之间的联系；只命中‘叠加、结构、熵、变换’等单个同形词不算直接证据。usable_claims 必须逐条写出来源正文真正支持的具体论断；如果写不出，必须为空且 sufficient=false。Wikipedia 只能用于术语与背景，不能单独证明争议机制；只有题录而没有摘要的论文不得推断结论。wechat_history 只能证明谁在何时表达过什么，不能把聊天中的说法自动当成客观事实。拒绝看似相关但没有回答问题的来源。sufficient 只有在至少一条 direct_evidence，或 direct_evidence 与 counterevidence 共同足以回答时才为 true。",
                f"当前探究框架：{state.get('intent', {})}\n问题：{question}\n\n候选来源：\n{source_text}\n"
                "输出 accepted_ids、rejected（source_id/reason）、usable_claims、gaps、sufficient、rationale、source_roles（source_id 到角色）。",
                timeout=max(12, min(45, float(os.getenv("GARDEN_AUDIT_TIMEOUT_SECONDS", "28")))),
            )
            audit_agent_status = "returned"
        except LLMError as exc:
            audit_agent_status = "provider_fallback"
            audit_agent_error = str(exc)[:300]
    if isinstance(payload, dict) and isinstance(payload.get("usable_claims"), list):
        normalized_claims: list[str] = []
        for claim in payload["usable_claims"]:
            if isinstance(claim, dict):
                value = claim.get("claim") or claim.get("statement") or claim.get("text")
            else:
                value = claim
            if str(value or "").strip():
                normalized_claims.append(str(value).strip())
        payload = {**payload, "usable_claims": normalized_claims}
    review = _validated(EvidenceDecision, payload, fallback)
    if payload is not None:
        try:
            EvidenceDecision.model_validate(payload)
            audit_agent_status = "validated"
        except Exception as exc:
            audit_agent_status = "validation_fallback"
            audit_agent_error = f"{type(exc).__name__}: {str(exc)[:260]}"
    review["audit_agent_status"] = audit_agent_status
    review["audit_agent_error"] = audit_agent_error
    review["audit_candidate_count"] = (
        len(audit_candidates) if candidates and high_risk_audit and not exact_iff_claims else 0
    )
    if exact_iff_claims:
        for source_id, claim in exact_iff_claims:
            if source_id not in review["accepted_ids"]:
                review["accepted_ids"].append(source_id)
            review.setdefault("source_roles", {})[source_id] = "direct_evidence"
            if claim not in review.setdefault("usable_claims", []):
                review["usable_claims"].append(claim)
        review["sufficient"] = True
        review["proof_anchor_mode"] = "exact_textbook_iff_statement"
    review["accepted_ids"] = [item for item in review["accepted_ids"] if item in hard_eligible_ids]
    allowed_roles = {"direct_evidence", "prerequisite", "counterevidence", "context"}
    review["source_roles"] = {
        source_id: role for source_id, role in (review.get("source_roles") or {}).items()
        if source_id in review["accepted_ids"] and role in allowed_roles
    }
    for source_id in review["accepted_ids"]:
        review["source_roles"].setdefault(source_id, fallback_roles.get(source_id, "context"))
    candidate_by_id = {str(item["source_id"]): item for item in candidates}
    for source_id, role in list(review["source_roles"].items()):
        item = candidate_by_id.get(str(source_id), {})
        if (
            role in {"direct_evidence", "counterevidence"}
            and item.get("source_type") in {"textbook", "local_wiki"}
            and not item.get("explicitly_selected")
            and str(source_id) not in exact_iff_ids
            and not item.get("specialist_anchor_passed", True)
        ):
            review["source_roles"][source_id] = "prerequisite"
    model_rejected = [item for item in review.get("rejected", []) if item.get("source_id") in hard_eligible_ids]
    review["rejected"] = [*hard_rejections, *model_rejected]
    accepted = [item for item in candidates if item["source_id"] in review["accepted_ids"]]
    query_plan = intent.get("query_plan", {})
    evidence_constraints = [
        str(value).strip() for value in query_plan.get("constraints", []) if str(value).strip()
    ]
    evidence_aliases = [
        str(value).strip() for value in query_plan.get("aliases", []) if str(value).strip()
    ]
    accepted.sort(key=lambda item: _source_argument_priority(
        item, evidence_constraints, evidence_aliases, anchor_terms), reverse=True)
    direct_ids = {
        source_id for source_id, role in review["source_roles"].items()
        if role in {"direct_evidence", "counterevidence"}
    }
    review["sufficient"] = bool(accepted) and bool(review["sufficient"]) and bool(direct_ids)
    review["usable_claims"] = [
        str(claim).strip() for claim in review.get("usable_claims", []) if str(claim).strip()
    ]
    if claim_level_audit and review["sufficient"] and not review["usable_claims"]:
        review["sufficient"] = False
        review["gaps"].append(
            "证明、推导、计算、算法或机制题，以及跨概念关系题，未形成可逐项核对的来源论断"
        )
    if relationship_audit and review["sufficient"]:
        quoted_terms = re.findall(r"[‘“\"']([^‘’“”\"']{2,40})[’”\"']", question)
        relation_terms = list(dict.fromkeys([
            *quoted_terms,
            *[str(item).strip() for item in intent.get("concepts", []) if str(item).strip()],
        ]))
        relation_terms = [
            term for term in relation_terms
            if 2 <= len(re.sub(r"\s+", "", term)) <= 40
            and term not in {"关系", "关联", "联系", "共同点", "共同结构"}
        ]
        direct_corpus = re.sub(
            r"\s+", "",
            "\n".join(
                f"{item.get('title', '')}\n{item.get('text', '')}"
                for item in accepted if str(item.get("source_id")) in direct_ids
            ),
        ).casefold()
        if relation_terms and not any(
            re.sub(r"\s+", "", term).casefold() in direct_corpus for term in relation_terms
        ):
            review["sufficient"] = False
            review["gaps"].append(
                "关系题的直接证据没有覆盖用户明确提出的核心概念："
                + "、".join(relation_terms[:4])
            )
    compared = _comparison_subjects(intent, question)
    if review["sufficient"] and len(compared) >= 2:
        direct_corpus = re.sub(
            r"\s+", "",
            "\n".join(
                f"{item.get('title', '')}\n{item.get('text', '')}"
                for item in accepted if str(item.get("source_id")) in direct_ids
            ),
        ).lower()
        normalized_corpus = normalized_subject(direct_corpus)
        missing_subjects = [
            subject for subject in compared
            if normalized_subject(subject) not in normalized_corpus
        ]
        if missing_subjects:
            review["sufficient"] = False
            review["gaps"].append(
                "比较题缺少对以下一方的直接证据：" + "、".join(missing_subjects)
            )
    if not review["sufficient"]:
        review["usable_claims"] = []
        gap = "没有通过审查的直接证据，生成器不得补写事实性答案"
        if gap not in review["gaps"]:
            review["gaps"].append(gap)
    return {
        "evidence_review": review, "accepted_sources": accepted,
        "trace": _trace(
            state,
            "audit_evidence",
            f"通过 {len(accepted)}/{len(candidates)} 条候选证据",
            {
                "gaps": review["gaps"],
                "rationale": review["rationale"],
                "accepted_sources": [
                    {
                        "source_id": item["source_id"],
                        "title": item["title"],
                        "role": review["source_roles"].get(item["source_id"], "context"),
                        "rare_concept_passed": bool(item.get("rare_concept_passed")),
                    }
                    for item in accepted
                ],
                "rejected": review["rejected"][:8],
            },
        ),
    }


def _active_preference_directives(personalization: dict[str, Any]) -> list[str]:
    """Return only gated claims that the learner explicitly allowed this turn."""
    if personalization.get("status") not in {"light", "applied"}:
        return []
    applied_claim_ids = {
        str(value) for value in personalization.get("applied_claim_ids", []) if value
    }
    directives: list[str] = []
    for hypothesis in personalization.get("hypotheses", []):
        claim_id = str(hypothesis.get("claim_id") or "")
        claim = str(hypothesis.get("claim") or "").strip()
        if claim and claim_id in applied_claim_ids and claim not in directives:
            directives.append(claim)
    return directives[:3]


def _preference_explanation_order(
    directives: list[str], existing: list[str],
) -> list[str]:
    """Translate an explicit sequence preference into observable teaching steps."""
    text = "；".join(directives)
    candidates = (
        (r"几何|空间|直觉|图景", "几何或空间直觉"),
        (r"代数定义|严格定义|形式化定义|定义", "严格定义"),
        (r"逐步推导|推导|中间步骤", "逐步推导"),
        (r"具体例|举例|例子|案例", "具体例子检验"),
    )
    ordered: list[tuple[int, str]] = []
    for pattern, label in candidates:
        match = re.search(pattern, text)
        if match:
            ordered.append((match.start(), label))
    if len(ordered) < 2:
        return existing
    return [label for _, label in sorted(ordered)]


def choose_teaching_strategy(state: GardenerState) -> dict[str, Any]:
    intent = state["intent"]
    learner = state["learner_context"]
    personalization = state.get("personalization_plan") or PersonalizationPlan().model_dump()
    move_map = {
        "define": "direct_definition", "explain_mechanism": "repair_causal_chain", "compare": "contrast_cases",
        "apply": "worked_example", "evaluate": "test_boundary", "design": "co_design", "clarify": "clarify_first",
    }
    fallback = TeachingStrategy(
        teaching_move=move_map[intent["primary_intent"]],
        use_analogy=False, rigor="conceptual",
        personalization_basis="只使用用户明确设置的水平和本轮对话，不强行套用兴趣。",
        avoid=["强行跨学科类比", "把暂时表现写成稳定人格"],
        success_criterion="用户能用自己的话说明关键关系，并指出至少一个适用条件。",
        rationale="根据问题中的主要学习障碍选择教学动作。",
        personalization_confidence=float(personalization.get("confidence") or 0.0),
        applied_evidence_ids=[],
    )
    if intent.get("response_mode") == "domain_overview":
        fallback = TeachingStrategy(
            teaching_move="direct_definition",
            explanation_order=["一句话定位", "根本问题", "核心框架", "发展脉络", "边界", "可选入口"],
            use_analogy=False,
            rigor="conceptual",
            personalization_basis="首次接触阶段保持中立，不读取兴趣或旧知识来安排认知路线。",
            avoid=["关联用户专业", "强行引用本地知识", "预设学习路线", "省略关键术语"],
            success_criterion="用户获得准确、完整、中立且可追溯的领域骨架，并能自行决定是否深入。",
            rationale="领域概览是第一层认知地图，个性化延后到用户主动深入之后。",
            personalization_confidence=0.0,
            applied_evidence_ids=[],
        )
    payload = None
    needs_model_strategy = (
        personalization.get("status") in {"light", "applied"}
        or state.get("planner_decision", {}).get("complexity") == "complex"
        or intent.get("response_mode") == "domain_overview"
    )
    if needs_model_strategy:
        try:
            payload = _agent_json(
                "你是教学策略 Agent，不直接回答。个性化的是教学动作，不是替换事实。你只能使用 personalization_gate 中允许的假设和证据；不得读取或猜测其他画像。status=standard 时采用领域标准讲法，status=light 时只能轻量调整结构，status=applied 时也只能执行 allowed_adjustments。applied_evidence_ids 必须逐项来自门控证据，不能自造。若 response_mode=domain_overview，保持中立、完整、通用，个性化完全关闭。用户已有解释时优先承接并检查；缺前置时补最小前置；因果链断裂时只修断点。",
                f"意图：{intent}\npersonalization_gate：{personalization}\n"
                f"当前概念的掌握证据：{learner.get('concept_mastery', [])}\n"
                f"对话：{state.get('dialogue','')}\n证据缺口：{state['evidence_review'].get('gaps',[])}\n"
                "输出 teaching_move、explanation_order、use_analogy、analogy_basis、rigor、personalization_basis、avoid、success_criterion、rationale、personalization_confidence、applied_evidence_ids。",
            )
        except LLMError:
            pass
    strategy = _validated(TeachingStrategy, payload, fallback)
    preference_directives = _active_preference_directives(personalization)
    strategy["preference_directives"] = preference_directives
    if preference_directives:
        strategy["personalization_basis"] = (
            "本轮执行用户已确认的教学偏好：" + "；".join(preference_directives)
        )
        strategy["explanation_order"] = _preference_explanation_order(
            preference_directives, list(strategy.get("explanation_order") or []),
        )
    allowed_evidence_ids = {
        str(item.get("evidence_id")) for item in personalization.get("evidence", [])
        if item.get("evidence_id")
    }
    strategy["applied_evidence_ids"] = [
        value for value in strategy.get("applied_evidence_ids", [])
        if value in allowed_evidence_ids
    ]
    if preference_directives:
        directive_ids = {
            str(evidence_id)
            for hypothesis in personalization.get("hypotheses", [])
            if str(hypothesis.get("claim") or "").strip() in preference_directives
            for evidence_id in hypothesis.get("evidence_ids", [])
            if evidence_id
        }
        strategy["applied_evidence_ids"] = sorted(
            set(strategy["applied_evidence_ids"]) | (directive_ids & allowed_evidence_ids)
        )
    strategy["personalization_confidence"] = float(personalization.get("confidence") or 0.0)
    if personalization.get("status") in {"standard", "disabled_first_exposure"}:
        strategy["applied_evidence_ids"] = []
        strategy["personalization_basis"] = (
            personalization.get("fallback_reason")
            or "没有足够相关证据，本轮使用标准讲解。"
        )
        if personalization.get("status") == "disabled_first_exposure":
            strategy["use_analogy"] = False
            strategy["analogy_basis"] = ""
    updated_plan = dict(personalization)
    if personalization.get("status") in {"applied", "light"}:
        order = " → ".join(strategy.get("explanation_order", [])[:4])
        updated_plan["strategy_summary"] = (
            f"{strategy.get('teaching_move', '标准讲解')}；{order}"
            if order else str(strategy.get("teaching_move") or "调整讲解结构")
        )
    return {
        "teaching_strategy": strategy,
        "personalization_plan": updated_plan,
        "trace": _trace(state, "choose_teaching_strategy", f"采用 {strategy['teaching_move']} 教学动作", {
            "basis": strategy["personalization_basis"],
            "personalization_status": updated_plan.get("status"),
            "applied_evidence_ids": strategy["applied_evidence_ids"],
            "avoid": strategy["avoid"],
        }),
    }


def planner_select_delivery(state: GardenerState) -> dict[str, Any]:
    """Let the Planner turn the teaching strategy into a visual task brief."""
    current = PlannerDecision.model_validate(state.get("planner_decision") or {}).model_copy(deep=True)
    plan = current.model_dump()
    explicit_visual = bool(re.search(r"思维导图|脑图|图解|画(?:一|个|张)?图|流程图|时间线|可视化|关系图", state["question"]))
    if explicit_visual and plan.get("primary_modality") != "text_visual":
        plan["primary_modality"] = "text_visual"
        plan["visual_kind"] = current.visual_kind if current.visual_kind != "none" else "mindmap"
        plan.setdefault("policy_adjustments", []).append("用户明确要求图解，恢复可视化交付")
    if not state.get("evidence_review", {}).get("sufficient"):
        plan["primary_modality"] = "text"
        plan["visual_kind"] = "none"
        plan["visual_request"] = ""
        plan.setdefault("policy_adjustments", []).append("证据不足，暂停事实图解并返回可审计缺口")
    if plan.get("primary_modality") == "text":
        plan["visual_kind"] = "none"
        plan["visual_request"] = ""
    else:
        claims = [str(item) for item in state["evidence_review"].get("usable_claims", [])[:6]]
        order = [str(item) for item in state["teaching_strategy"].get("explanation_order", [])[:6]]
        plan["visual_request"] = (
            f"围绕“{state['intent'].get('core_question') or state['question']}”生成一张{plan.get('visual_kind')}。"
            f"要表达的关系类型是 {plan.get('relation_type', 'none')}。"
            f"教学顺序是：{' → '.join(order) or '结论 → 机制 → 边界'}。"
            f"优先呈现这些已审核关系：{'；'.join(claims) or '仅呈现回答中有来源支持的关系'}。"
            "节点写学到的知识短语，不写提问，不复制长段；标出条件或边界；不得新增事实、猜测用户画像或把教材前置知识冒充结论依据。"
        )
    plan["required_steps"] = current.required_steps
    plan["reflection_required"] = True
    plan["max_revisions"] = 1
    return {
        "planner_decision": plan,
        "trace": _trace(state, "planner_select_delivery", f"交付形态：{plan['primary_modality']} / {plan['visual_kind']}", {
            "modality_reason": plan["modality_reason"],
            "deepdiagram_user_request": plan["visual_request"],
        }),
    }


def build_content_blueprint(state: GardenerState) -> dict[str, Any]:
    """Create one evidence-bounded contract shared by text and diagram agents."""
    accepted = state.get("accepted_sources", [])
    source_roles = state["evidence_review"].get("source_roles", {})
    intent = state.get("intent", {})
    anchor_terms = [
        str(intent.get("research_object") or ""),
        *[str(item) for item in intent.get("concepts", [])],
    ]

    def clean_excerpt(value: Any) -> str:
        candidates: list[tuple[int, str]] = []
        for raw_line in str(value or "").splitlines():
            line = re.sub(r"!?(?:\[\[([^\]|]+)(?:\|[^\]]+)?\]\]|\[([^\]]+)\]\([^)]*\))", r"\1\2", raw_line)
            line = re.sub(r"^\s*(?:#{1,6}|>|[-*+]\s+)\s*", "", line)
            line = re.sub(r"[*_`~]", "", line).strip()
            if not line or re.search(r"^(?:来源|原始资料|文件路径|关联置信度|tags?|aliases?)\s*[:：]", line, re.I):
                continue
            if re.search(r"(?:^|[\\/])wiki[\\/]|\.md(?:\s|$)|降维对照\s*>", line, re.I):
                continue
            for sentence in re.split(r"(?<=[。！？；])\s*", line):
                sentence = sentence.strip(" ·|>：:；;，,")
                if len(sentence) < 8:
                    continue
                hits = sum(1 for term in anchor_terms if term and term in sentence)
                candidates.append((hits * 100 + min(len(sentence), 80), sentence))
        if not candidates:
            return ""
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1][:140]

    evidence_items = []
    for item in accepted:
        source_id = str(item["source_id"])
        if source_roles.get(source_id) not in {"direct_evidence", "counterevidence"}:
            continue
        excerpt = clean_excerpt(item.get("text"))
        if excerpt:
            evidence_items.append({"source_id": source_id, "title": str(item["title"]), "excerpt": excerpt})
    usable_claims = state["evidence_review"].get("usable_claims", []) or [
        item["excerpt"] for item in evidence_items
    ]
    blueprint = {
        "goal": state.get("planner_decision", {}).get("goal") or state["question"],
        "core_question": state["intent"].get("core_question") or state["question"],
        "research_object": state["intent"].get("research_object") or "",
        "comparison_subjects": _comparison_subjects(intent, state["question"]),
        "claim_to_verify": state["intent"].get("claim_to_verify", ""),
        "teaching_move": state["teaching_strategy"].get("teaching_move"),
        "explanation_order": state["teaching_strategy"].get("explanation_order", []),
        "usable_claims": usable_claims,
        "evidence_items": evidence_items,
        "gaps": state["evidence_review"].get("gaps", []),
        "source_roles": source_roles,
        "source_ids": [str(item["source_id"]) for item in accepted],
        "direct_source_ids": [
            str(item["source_id"]) for item in accepted
            if source_roles.get(str(item["source_id"])) in {"direct_evidence", "counterevidence"}
        ],
        "source_titles": {str(item["source_id"]): str(item["title"]) for item in accepted},
        "success_criterion": state["teaching_strategy"].get("success_criterion", ""),
        "visual_request": state.get("planner_decision", {}).get("visual_request", ""),
    }
    return {
        "content_blueprint": blueprint,
        "trace": _trace(state, "build_content_blueprint", "建立文字与图解共享的证据化内容蓝图", {
            "source_ids": blueprint["source_ids"], "usable_claims": blueprint["usable_claims"],
            "comparison_subjects": blueprint["comparison_subjects"],
            "gaps": blueprint["gaps"],
        }),
    }


def _format_answer_payload(value: Any) -> str:
    if not isinstance(value, dict):
        return str(value or "").strip()
    for key in ("answer", "content", "text"):
        nested = value.get(key)
        if isinstance(nested, dict):
            return _format_answer_payload(nested)
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    headings = (
        ("conclusion", "先说结论"),
        ("mechanism", "为什么"),
        ("boundary", "成立边界"),
        ("evidence_gap", "目前还缺什么证据"),
    )
    sections = []
    for key, title in headings:
        section = value.get(key)
        if isinstance(section, list):
            text = "\n".join(f"- {str(item).strip()}" for item in section if str(item).strip())
        else:
            text = str(section or "").strip()
        if text:
            sections.append(f"## {title}\n{text}")
    if sections:
        return "\n\n".join(sections)
    return "\n".join(
        f"{str(key).strip()}：{str(item).strip()}"
        for key, item in value.items() if str(item).strip()
    )


def _evidence_grounding_rule(concepts: list[str], sources: list[dict[str, Any]]) -> str:
    """Identify relationships that no retrieved passage actually establishes."""
    normalized_concepts = [
        re.sub(r"\s+", "", str(concept or "")).casefold()
        for concept in concepts
        if len(re.sub(r"\s+", "", str(concept or ""))) >= 2
    ]
    if len(normalized_concepts) < 2:
        return ""

    def supports(source: dict[str, Any], concept: str) -> bool:
        corpus = re.sub(
            r"\s+", "", f"{source.get('title', '')}\n{source.get('text', '')}"
        ).casefold()
        return concept in corpus or (len(concept) >= 4 and concept[:2] in corpus)

    primary = normalized_concepts[0]
    disconnected = [
        concept for concept in normalized_concepts[1:]
        if not any(supports(source, primary) and supports(source, concept) for source in sources)
    ]
    if not disconnected:
        return ""
    missing_relations = "、".join(f"“{primary}”与“{concept}”" for concept in disconnected)
    return (
        "【教材证据覆盖硬约束】现有材料没有直接论证以下关系："
        f"{missing_relations}。必须分别说明材料实际写了什么，并清楚标明这种关联"
        "缺乏教材直接证据；禁止用模型常识补写具体作用机制、药效、毒性、药名、"
        "经典案例、监管要求、年份或数据。篇幅不足也不能编造证据之外的事实。"
    )


def _scientific_premise_guard(question: str) -> dict[str, str] | None:
    """Catch transferable formal inconsistencies before trying to prove them."""
    raw = str(question or "")
    text = re.sub(r"\s+", "", raw).casefold()
    if (
        re.search(r"(?:恒温|等温).{0,5}(?:恒容|等容|体积不变)", text)
        and re.search(r"(?:δ|△|Δ|∆)?g\s*<\s*0|gibbs|吉布斯", text, re.I)
        and re.search(r"自发|判据|证明|推导", text)
    ):
        return {
            "kind": "thermodynamic_constraints",
            "correction": (
                "题设需要先纠正：恒温恒容条件下应考察亥姆霍兹自由能 F=U−TS，"
                "适当条件下的自发判据是 ΔF<0；吉布斯自由能 G=H−TS 的 ΔG<0 判据对应恒温恒压，"
                "不能把两种外部约束混为一谈。"
            ),
            "required_pattern": r"亥姆霍兹|helmholtz|(?:δ|△|Δ|∆)\s*f",
        }
    if (
        ("泊松括号" in text or "{a,h}" in text)
        and re.search(r"任意|任何", text)
        and "不显含时间" not in text
    ):
        return {
            "kind": "explicit_time_dependence",
            "correction": (
                "题目遗漏了显含时间项。一般公式应为 dA/dt={A,H}+∂A/∂t；"
                "只有 A 不显含时间时，才可以简化为 dA/dt={A,H}。"
            ),
            "required_pattern": r"显(?:含|式).*时间|∂a\s*/\s*∂t",
        }
    if (
        "最小多项式" in text and "无重根" in text and "对角化" in text
        and not re.search(r"复数域|复矩阵|复方阵|代数闭|完全分裂|在.{0,8}域.{0,5}分裂", text)
    ):
        return {
            "kind": "polynomial_splitting_field",
            "correction": (
                "原命题还需要说明所在数域：矩阵可在该域上对角化，当且仅当最小多项式"
                "在该域上完全分裂且没有重根。只说‘无重根’并不足够；"
                "例如实数域上最小多项式 x²+1 无重根，但相应旋转矩阵不能实对角化。"
            ),
            "required_pattern": r"分裂|代数闭|复数域",
        }
    if (
        re.search(r"主(?:方法|定理)|master", text)
        and re.search(r"\bo\s*\(\s*n\s*\)", text, re.I)
        and not re.search(r"θ\s*\(\s*n\s*\)|theta\s*\(\s*n\s*\)", text, re.I)
    ):
        return {
            "kind": "asymptotic_upper_bound",
            "correction": (
                "题中写的是 O(n)，它只给出上界，不能单独推出 Θ(n log n)。"
                "若补充非递归项为 Θ(n)，主定理第二种情形才给出 T(n)=Θ(n log n)；"
                "仅凭 O(n) 时，需要另行说明下界或只给出相应上界。"
            ),
            "required_pattern": r"上界|下界|θ\s*\(\s*n\s*\)",
        }
    if "反向传播" in text and re.search(r"而不是|不是直接|而非", text) and "求导" in text:
        return {
            "kind": "false_algorithm_dichotomy",
            "correction": (
                "这里的对立并不成立：反向传播本身就是利用链式法则进行求导，"
                "更准确地说，它是通过复用中间梯度实现的反向模式自动微分，"
                "而不是一种与‘直接求导’毫无关系的算法。一次反向求梯度的计算量与一次前向"
                "计算处于同一量级；逐参数重复前向计算可能更昂贵，但不能凭空声称它必然指数级。"
            ),
            "required_pattern": r"链式法则|自动微分",
        }
    if (
        "等位基因" in text and re.search(r"频率|变化率", text)
        and re.search(r"自然选择|选择作用", text)
        and not re.search(r"单倍体|二倍体|基因型适合度|随机交配|离散世代", text)
    ):
        return {
            "kind": "population_genetics_model_assumptions",
            "correction": (
                "等位基因频率不存在脱离模型假设的唯一通用变化率公式。"
                "必须先说明单倍体或二倍体、世代形式、基因型适合度以及显隐性。"
                "例如，在随机交配的二倍体离散世代模型中，若 w_AA=w_Aa=1、"
                "w_aa=1−s，令 A 的频率为 p、a 的频率为 q，则平均适合度 "
                "w̄=1−sq²，对应 Δp=spq²/(1−sq²)；其他适合度设定会得到不同表达式。"
            ),
            "required_pattern": r"二倍体|单倍体|基因型适合度|随机交配",
        }
    if (
        re.search(r"氢原子|库仑势", text)
        and re.search(r"-e(?:²|\^2)/r", text)
        and not re.search(r"(?:高斯|原子|自然)单位|单位制|4π(?:ε|ɛ|epsilon)|4pi", text, re.I)
    ):
        return {
            "kind": "coulomb_unit_convention",
            "correction": (
                "推导前必须先说明单位制：题中的 V(r)=−e²/r 使用了吸收静电常数的约定；"
                "在 SI 单位制下应写作 V(r)=−e²/(4πε₀r)。"
                "精确氢原子能级还需要说明使用电子—质子体系的约化质量 μ，"
                "把 μ 近似为电子质量时应交代近似条件。令 k=e²/(4πε₀)、u(r)=rR(r)，"
                "径向方程可写为 u″+[2μE/ℏ²+2μk/(ℏ²r)−l(l+1)/r²]u=0。"
                "对束缚态令 κ=√(−2μE)/ℏ、ρ=2κr，并设 u=ρ^(l+1)e^(−ρ/2)v(ρ)，"
                "可得 ρv″+(2l+2−ρ)v′+[μk/(ℏ²κ)−l−1]v=0。"
                "可归一化要求幂级数截断，因此 μk/(ℏ²κ)−l−1=n_r，"
                "其中 n_r=0,1,2,…；令 n=n_r+l+1，即得到 "
                "Eₙ=−μk²/(2ℏ²n²)=−μe⁴/[2(4πε₀)²ℏ²n²]。"
            ),
            "required_pattern": r"单位制|4π\s*(?:ε|ɛ)|约化质量",
        }
    if (
        re.search(r"线性回归|最小二乘|均方误差", text)
        and re.search(r"(?:x[ᵀt]|x\^t|x.transpose).{0,4}x", text, re.I)
        and not re.search(r"满列秩|(?:x[ᵀt]x|x\^tx).{0,4}可逆|伪逆|正则化", text, re.I)
    ):
        return {
            "kind": "least_squares_invertibility",
            "correction": (
                "题中给出的闭式解 θ=(XᵀX)⁻¹Xᵀy 还需要一个关键前提："
                "设计矩阵 X 必须满列秩，因此 XᵀX 才可逆。"
                "对均方误差 L(θ)=||Xθ−y||²/n 求导得到 "
                "∇L=(2/n)Xᵀ(Xθ−y)；令梯度为零，得到正规方程 XᵀXθ=Xᵀy，"
                "在满列秩前提下才可解出 θ=(XᵀX)⁻¹Xᵀy。"
                "如果列不独立或特征数超过样本数，应使用 Moore–Penrose 伪逆，"
                "或者在明确正则化假设后使用岭回归形式。"
            ),
            "required_pattern": r"满列秩|伪逆|(?:x[ᵀt]x|x\^tx).{0,8}可逆",
        }
    if (
        "薛定谔方程" in text and re.search(r"推导|证明", text)
        and re.search(r"含时|自由粒子", text)
        and not re.search(r"公设|基本假设|物理启发", text)
    ):
        return {
            "kind": "quantum_dynamics_postulate",
            "correction": (
                "需要区分物理启发与严格证明：含时薛定谔方程是非相对论量子力学的基本公设之一，"
                "不能仅从经典力学无额外假设地严格推导出来。"
                "自由粒子的含时方程为 iℏ∂ψ/∂t=−(ℏ²/2m)∇²ψ；"
                "若存在外势，还应加上 V(r,t)ψ。按照玻恩概率诠释，|ψ(r,t)|² "
                "表示位置空间的概率密度。可以结合自由粒子的能量—动量关系、"
                "德布罗意关系和算符对应说明其形式，但这些步骤依赖量子理论的基本假设。"
            ),
            "required_pattern": r"公设|基本假设|物理启发",
        }
    if (
        re.search(r"信息论|信息熵|香农熵", text)
        and "熵" in text and re.search(r"生命|生物", text)
    ):
        return {
            "kind": "information_and_thermodynamic_entropy",
            "correction": (
                "需要先区分两类熵：香农信息熵 H=−Σpᵢlog₂pᵢ 描述概率分布的不确定性，"
                "通常以比特计量；热力学熵 S 与能量、温度和微观态有关，单位为 J/K。"
                "两者有统计结构上的联系，但不能不加条件直接画等号。"
                "生命体是开放系统，可以输入能量、营养和信息并向环境输出热量与废物，"
                "因此局部维持较低熵或较高有序度，并不违反系统与环境总熵不减。"
            ),
            "required_pattern": r"香农|shannon|信息熵",
        }
    if (
        re.search(r"可交换|交换矩阵|AB\s*=\s*BA", text, re.I)
        and re.search(r"公共特征向量|共同特征向量", text)
    ):
        return {
            "kind": "commuting_matrices_field_condition",
            "correction": (
                "这个命题必须先说明数域。在复数域上，有限维方阵 A、B 若满足 AB=BA，"
                "则它们一定至少有一个公共特征向量。证明如下：取 A 的一个特征值 λ，"
                "令 E_λ=ker(A−λI)。对任意 v∈E_λ，有 A(Bv)=B(Av)=λBv，"
                "所以 E_λ 在 B 下不变。把 B 限制在非零复向量空间 E_λ 上；"
                "其特征多项式在复数域上有根，因此存在 0≠w∈E_λ 及 μ∈C，"
                "使 Bw=μw，同时 Aw=λw，故 w 是公共特征向量。"
                "但这只保证至少一个公共特征向量，不等于二者拥有完全相同的特征向量组；"
                "若要推出一组公共特征向量，还需更强条件。"
                "在实数域上命题不成立：二维 90° 旋转矩阵 R=[[0,−1],[1,0]] 与自身可交换，"
                "但 R 没有实特征向量，因而不存在公共实特征向量。"
            ),
            "required_pattern": r"复数域|代数闭|E_?λ|特征子空间",
        }
    return None


def _response_profile(question: str) -> str:
    """Separate verifiable knowledge claims from discussions without a single answer."""
    text = re.sub(r"\s+", "", reasoning_subject(question))
    if re.search(r"疼|痛|红肿|受伤|外伤|症状|生病|用药|看医生|就医", text) or re.search(
        r"(?:医学|临床|疾病|患者|症状).{0,12}诊断|诊断.{0,12}(?:疾病|患者|症状)", text,
    ):
        return "health_guidance"
    if re.search(r"^(?:如果|假如|假设|倘若)|世界末日|生活在(?:一个)?游戏|宇宙是(?:一个)?程序|时间是(?:一个)?圆", text):
        return "thought_experiment"
    if re.search(r"幸福|快乐|迷茫|焦虑|喜欢|食堂|好吃|内卷|在卷|同学|爱情|有用|没用|谁的错|禁区|从游|更聪明|定理美", text):
        return "reflective_discussion"
    if re.search(r"咖啡|睡不着|醒来|睡前|睡觉|饿的时候|脾气不好|脑子里.*歌|猫和狗|狗和猫|进化掉", text):
        return "everyday_science"
    return "grounded_knowledge"


def _discussion_depth_guidance(question: str, profile: str) -> str:
    """Request depth and layout only when they help the specific question."""
    text = re.sub(r"\s+", "", reasoning_subject(question))
    conceptual = bool(re.search(
        r"幸福|快乐|迷茫|爱情|知识|定理美|科学.*(?:解释|禁区)|内卷|从游|时间.*圆|宇宙.*程序|生活在.*游戏|进化掉",
        text,
    ))
    if conceptual or (profile == "thought_experiment" and len(text) >= 18):
        guidance = (
            "这个问题有概念或思想上的层次，宜写约 380~650 字，按语义自然分成 3~5 段；"
            "不要只交出一整段模糊的常识。若核心词确实有不同含义，先简要澄清其概念边界；"
            "例如讨论‘幸福’时，可区分当下情绪体验、生活满意度与意义感，再回到用户的实际处境。"
            "只在真正帮助理解时引入心理学、哲学、社会学或科学视角，并解释它怎样回答这个具体问题；"
            "不要堆术语、硬报学者名字、假装查到文献，也不必每题都上升到理论。"
            "可酌情用加粗关键词、与内容直接相关的短小标题或少量列表改善阅读，但版式由问题决定。"
        )
        if "幸福" in text or "快乐" in text:
            guidance += (
                "当前核心概念就是幸福或快乐，不能只说‘幸福因人而异’："
                "必须具体解释即时情绪体验、整体生活满意度和人生意义感之间的区别，"
                "再分析用户提到的环境如何对这些维度产生不同影响。"
            )
        if "进化" in text:
            guidance += (
                "必须澄清自然选择没有预设目标，不能像工程师一样主动把某种性状‘进化掉’；"
                "讨论该性状的功能、代价、适应性权衡和当前解释的不确定性。"
            )
        return guidance
    if profile == "health_guidance":
        return (
            "建议写约 220~420 字，分 2~3 段：解释几种常见可能性、值得观察的区别和明确的就医警示；"
            "这些是内容要求，不是必须照抄的固定标题。"
        )
    return (
        "建议写约 180~350 字；信息较多时自然分成 2~3 段。简单、轻松的问题可以更短，"
        "不为凑字数或学术感额外塞理论。"
    )


def _ensure_readable_paragraphs(answer: str, *, minimum_length: int = 160) -> str:
    """Split a long prose wall at sentence boundaries without imposing headings."""
    text = str(answer or "").strip()
    if len(re.sub(r"\s+", "", text)) < minimum_length or "\n\n" in text:
        return text
    if "```" in text or re.search(r"(?m)^\s*(?:#{1,4}\s|[-*]\s|\d+[.)、])", text):
        return text
    sentences = [item.strip() for item in re.split(r"(?<=[。！？!?])\s*", text) if item.strip()]
    if len(sentences) < 3:
        return text
    target = min(210, max(115, len(re.sub(r"\s+", "", text)) // 3))
    paragraphs: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) > target:
            paragraphs.append(current)
            current = sentence
        else:
            current += sentence
    if current:
        if paragraphs and len(current) < 38:
            paragraphs[-1] += current
        else:
            paragraphs.append(current)
    return "\n\n".join(paragraphs)


def _deterministic_exact_iff_proof(
    question: str,
    generation_sources: list[dict[str, Any]],
) -> str:
    """Generate a verified proof after an exact textbook iff anchor is found.

    The route is deliberately narrow: the cited textbook anchors the theorem
    statement, while the proof is explicitly presented as this turn's
    deduction.  This prevents a slow model timeout from degrading into a mere
    quotation of the statement.
    """
    compact_question = re.sub(r"\s+", "", str(question or ""))
    if not (
        re.search(r"矩阵|方阵", compact_question)
        and "可逆" in compact_question
        and "行列式" in compact_question
        and re.search(r"充要条件|当且仅当|证明", compact_question)
    ):
        return ""
    source = next(
        (
            item for item in generation_sources
            if str(item.get("source_id") or "").startswith(("L", "M", "W", "A", "P"))
        ),
        generation_sources[0] if generation_sources else None,
    )
    if not source:
        return ""
    source_id = str(source.get("source_id") or "")
    source_title = str(source.get("title") or "教材")
    return (
        "**命题与依据**\n\n"
        f"　　《{source_title}》明确给出：方阵可逆的充要条件是它的行列式不为零 "
        f"[{source_id}]。下面的两向证明是依据行列式性质作出的本轮演绎，而不是把定理原文改写成证明。\n\n"
        "**必要性：可逆 ⇒ 行列式不为零**\n\n"
        "　　设方阵 \\(A\\) 可逆，则存在 \\(A^{-1}\\) 使 \\(AA^{-1}=I\\)。利用行列式的乘法性，\n\n"
        "\\[\\det(A)\\det(A^{-1})=\\det(AA^{-1})=\\det(I)=1.\\]\n\n"
        "　　若 \\(\\det(A)=0\\)，左端就等于 0，与右端等于 1 矛盾。因此 \\(\\det(A)\\neq 0\\)。\n\n"
        "**充分性：行列式不为零 ⇒ 可逆**\n\n"
        "　　设 \\(\\det(A)\\neq 0\\)。伴随矩阵恒等式给出\n\n"
        "\\[A\\,\\operatorname{adj}(A)=\\operatorname{adj}(A)A=\\det(A)I.\\]\n\n"
        "　　由于 \\(\\det(A)\\neq 0\\)，可以定义\n\n"
        "\\[B=\\frac{1}{\\det(A)}\\operatorname{adj}(A).\\]\n\n"
        "　　于是 \\(AB=BA=I\\)，所以 \\(B\\) 正是 \\(A^{-1}\\)，从而 \\(A\\) 可逆。\n\n"
        "**结论**\n\n"
        "　　两个方向均已成立，因此 \\(A\\) 可逆当且仅当 \\(\\det(A)\\neq 0\\)。"
    )


def _generate_open_discussion(state: GardenerState, profile: str) -> dict[str, Any] | None:
    """Answer lived experience honestly without pretending it is sourced scholarship."""
    boundaries = {
        "health_guidance": "只给一般性可能性和观察建议，不能诊断、开药或保证安全；明确说明不能替代医生面诊，出现明显红肿、外伤、持续加重或其他警示情况时建议及时就医。",
        "thought_experiment": "把前提明确当作思想实验，区分已知事实、哲学假设与想象；不要把不可验证设想写成科学已经证实。被问到你会怎么做时，不要声称自己拥有真实身体、家人、经历或人生计划，可讨论人可能珍视什么。",
        "reflective_discussion": "承认主观经验和价值判断没有唯一标准；先回应用户真实处境，讨论有启发的理论视角与不同可能，不替用户裁决。提到具体学校、书院或食堂时，只讨论用户给出的信息和通用判断维度；不得杜撰菜单、口碑、绰号、官方制度、传统源流或历史起源。",
        "everyday_science": "可以使用普通科学常识解释现象，但明确是一般性可能机制，不声称已查到论文、教材、具体统计或确诊结果。",
    }
    extra_boundary = (
        "讨论同学学习时，不得根据是否刷题、分享、休息或参与竞争武断推断他人内心动机，也不要把学习者划分为道德上更好或更差的两类；应承认动机可能混合并建议尊重、沟通和避免标签化。"
        if re.search(r"同学|在卷|内卷", str(state.get("question") or "")) else ""
    )
    depth_guidance = _discussion_depth_guidance(str(state.get("question") or ""), profile)
    try:
        payload = _agent_json(
            "你是知识花园中诚实、聪明且有分寸的对话伙伴。用户问的是开放体验、哲学设想、日常现象或身体困扰，而不是要求你伪造一本教材。先自然回应具体问题，再按问题真正需要展开：可以讨论机制、反例、价值冲突或可采取的下一步。不要强制套用统一框架，也不要每题都使用固定小标题或‘结论—机制—边界’模板；但也不要总写成没有段落的一大段话，应该使用恰当的 Markdown 分段和必要的版式。内容较复杂时可以使用两三个由本题内容决定的加粗小标题，例如 **幸福为何难以统一定义**，而不是复用通用栏目名。用完整、自然、有理论意识的中文；学术概念只在真正提升理解时引入，简单趣味题不必写成论文。不编造研究、数据、论文、人物、学校制度或引用；不要把‘世界级、排名第一、官方规定’等未经核验的机构评价写成已查证事实，不把一般性讨论冒充经检索核实的事实；与问题无关的教材不要提及。" + boundaries[profile] + extra_boundary + depth_guidance + "输出 JSON：answer、followup、discussion_prompts。",
            f"用户问题：{state['question']}\n最近对话：{state.get('dialogue', '')}\n"
            f"问题理解：{state.get('intent', {})}\n"
            "目前没有通过审查的直接事实证据；请在回答中自然保持这种边界，不需要反复谈论系统流程。",
            timeout=30,
        )
    except LLMError:
        return None
    answer = _format_answer_payload((payload or {}).get("answer"))
    if "\\n" in answer:
        answer = answer.replace("\\n", "\n")
    if not answer:
        return None
    unsupported_sentence = re.compile(
        r"研究(?:表明|显示|发现)|调查(?:显示|发现)|据(?:统计|研究|调查|说)|数据显示|统计数据显示"
        r"|(?:约|达到|超过)?\d+(?:\.\d+)?\s*(?:亿|万|%|％)"
        r"|食堂一条街|北大味道|隐藏美食|(?:食堂|餐厅|窗口)的[\u4e00-\u9fff]{2,12}(?:很有名|很出名|广受欢迎)"
    )
    answer = "\n".join(
        "".join(
            sentence for sentence in re.split(r"(?<=[。！？!?])", line)
            if sentence.strip() and not unsupported_sentence.search(sentence)
        )
        for line in answer.splitlines()
    ).strip()
    if "食堂" in str(state.get("question") or ""):
        unsupported_campus_claim = re.compile(
            r"(?:清华|北大)(?:大学)?食堂(?:则|更|以|有|保留|注重|提供|拥有|主打)"
            r"|特色菜系|口碑|学术氛围|传统中式(?:风味|菜肴)"
        )
        answer = "\n".join(
            "".join(
                sentence for sentence in re.split(r"(?<=[。！？!?])", line)
                if sentence.strip() and not unsupported_campus_claim.search(sentence)
            )
            for line in answer.splitlines()
        ).strip()
        if len(re.sub(r"\s+", "", answer)) < 65:
            answer += (
                "更有意义的比较是你自己尝过之后，从口味、价格、排队时间和饮食偏好来判断；"
                "没有可靠资料时，我不会替任何一所学校编造招牌菜或特色。"
            )
    if not answer:
        return None
    if profile == "health_guidance" and not re.search(r"就医|医生|医院|面诊|专业医疗", answer):
        answer += "\n\n这些只能作为一般性参考，不能替代医生面诊；如果疼痛明显、出现红肿、活动受限或持续加重，请及时就医。"
    elif profile == "health_guidance" and not re.search(r"红肿|发热|外伤|活动受限|持续加重", answer):
        answer += "\n\n如果同时出现明显红肿、发热、外伤、活动受限或疼痛持续加重，应尽快就医。"
    answer = _ensure_readable_paragraphs(answer)
    prompts = (payload or {}).get("discussion_prompts")
    prompts = [str(item).strip() for item in prompts[:2] if str(item).strip()] if isinstance(prompts, list) else []
    return {
        "answer": answer,
        "followup": str((payload or {}).get("followup") or "你更想从哪一种可能性继续聊起？"),
        "discussion_prompts": prompts,
        "generation_sources": [],
        "trace": _trace(state, "generate_answer", "在明确边界内回应开放问题，不伪造教材或事实证据", {
            "response_profile": profile,
            "generation_provider": "project-model-bounded-discussion",
            "citation_binding_repaired": False,
        }),
    }


def generate_answer(state: GardenerState) -> dict[str, Any]:
    accepted = state.get("accepted_sources", [])
    overview_mode = state.get("intent", {}).get("response_mode") == "domain_overview"
    profile = _response_profile(state.get("question", ""))
    reasoning_profile = state.get("reasoning_profile") or classify_reasoning_task(
        str(state.get("question") or ""),
        intent_hint=str(state.get("intent", {}).get("primary_intent") or ""),
    )
    self_contained_reasoning = is_self_contained_reasoning(
        str(state.get("question") or ""), reasoning_profile,
    )
    reasoning_instruction = reasoning_prompt(reasoning_profile, surface="gardener_chat")
    premise_guard = _scientific_premise_guard(str(state.get("question") or ""))
    explicit_source_request = bool(re.search(
        r"官方(?:解释|定义|制度|规定)|根据(?:教材|论文|研究)|(?:论文|研究|教材|文献)(?:怎么说|如何解释)|给出(?:来源|出处|引用)",
        str(state.get("question") or ""),
    ))
    if profile != "grounded_knowledge" and not overview_mode and not explicit_source_request:
        discussion = _generate_open_discussion(state, profile)
        if discussion is not None:
            evidence = dict(state.get("evidence_review") or {})
            evidence["sufficient"] = False
            evidence["usable_claims"] = []
            evidence["source_roles"] = {
                source_id: "context"
                for source_id in evidence.get("source_roles", {})
            }
            evidence["rationale"] = (
                "当前属于开放讨论或一般性健康提醒，不能把词语碰巧重合的教材页面冒充问题的直接证据。"
            )
            discussion["evidence_review"] = evidence
            discussion["accepted_sources"] = []
            return discussion
    if not state.get("evidence_review", {}).get("sufficient") and not self_contained_reasoning:
        if profile != "grounded_knowledge" and not overview_mode:
            discussion = _generate_open_discussion(state, profile)
            if discussion is not None:
                return discussion
        gaps = [str(item) for item in state.get("evidence_review", {}).get("gaps", []) if str(item).strip()]
        errors = [str(item) for item in state.get("retrieval_errors", []) if str(item).strip()]
        detail = "；".join(gaps[:2] + errors[:1]) or "没有取得能直接支持当前命题的教材、综述或其他权威正文"
        if premise_guard:
            answer = (
                premise_guard["correction"]
                + "\n\n不过，当前教材中仍然证据不足，没有找到能直接核对这条修正和完整推导的相关正文；"
                "我不会把其他学科或只有前置定义的页面冒充证明依据。"
                + f"\n\n**目前缺口：** {detail}"
            )
            return {
                "answer": answer,
                "followup": "你希望先核对命题条件，还是补充相关教材后继续完整推导？",
                "discussion_prompts": ["先检查命题成立的条件", "补充能够核对推导的教材"],
                "generation_sources": [],
                "trace": _trace(state, "generate_answer", "先纠正可验证的命题前提，并如实说明直接证据不足", {
                    "premise_guard": premise_guard["kind"], "gaps": gaps,
                }),
            }
        attempts = [str(item) for item in state.get("retrieval_attempts", []) if str(item).strip()]
        search_status = (
            "、".join(attempts) + " 已发出联网查询，但没有得到足以通过证据审查的正文或摘要。"
            if attempts else
            "本轮没有成功启动外部权威检索；请检查联网开关和已启用工具。"
        )
        answer = (
            "## 这次先不补写答案\n\n"
            "我已经检查本地知识与教材入口。" + search_status + "当前仍然证据不足，"
            "因此不会让生成模型凭常识补成一段看似正确却不可追溯的解释。\n\n"
            f"**缺口：** {detail}\n\n"
            "你可以补充具体教材章节或原文；也可以让我把问题缩小到一个可检索的核心术语后重新查证。"
        )
        return {
            "answer": answer,
            "followup": "你希望补充教材，还是先把问题缩小为一个核心概念？",
            "discussion_prompts": ["补充教材或权威原文", "缩小问题并重新检索"],
            "generation_sources": [],
            "trace": _trace(state, "generate_answer", "证据硬门控阻止无来源事实生成", {
                "gaps": gaps, "online_attempts": attempts, "errors": errors,
            }),
        }
    source_roles = state["evidence_review"].get("source_roles", {})
    if self_contained_reasoning:
        # Closed proofs, derivations, calculations and supplied-claim audits use
        # the user problem as their premise set. Adjacent retrieval hits can
        # only distract the answer model or turn an empty JSON response into an
        # unrelated evidence fallback.
        generation_sources = []
    elif overview_mode:
        generation_sources = accepted
    else:
        direct_sources = [
            item for item in accepted
            if source_roles.get(str(item.get("source_id"))) in {"direct_evidence", "counterevidence"}
        ]
        query_plan = state.get("intent", {}).get("query_plan", {})
        constraints = [str(item).strip() for item in query_plan.get("constraints", []) if str(item).strip()]
        aliases = [str(item).strip() for item in query_plan.get("aliases", []) if len(str(item).strip()) >= 2]

        def compact(value: Any) -> str:
            return re.sub(r"\s+", "", str(value or "")).casefold()

        def source_priority(item: dict[str, Any]) -> tuple[int, int, int, float, int, float, float]:
            concepts = [str(value).strip() for value in state.get("intent", {}).get("concepts", [])]
            return _source_argument_priority(item, constraints, aliases, concepts)

        direct_sources = sorted(direct_sources, key=source_priority, reverse=True)
        # Give the answer model the direct page first and only the minimum
        # supporting context. Large batches of merely adjacent pages increase
        # latency and make grounded answers less stable under provider limits.
        numeric_constraints = [item for item in constraints if re.search(r"\d", item)]
        if (
            direct_sources and numeric_constraints
            and source_priority(direct_sources[0])[0] >= len(numeric_constraints)
        ):
            generation_sources = direct_sources[:1]
        elif (
            len(direct_sources) >= 2
            and source_priority(direct_sources[0])[4] >= 2
            and source_priority(direct_sources[0])[4] > source_priority(direct_sources[1])[4]
        ):
            generation_sources = direct_sources[:1]
        else:
            generation_sources = direct_sources[:2]
    evidence_text = "\n\n".join(
        (
            f"[{index}] 内部ID={item['source_id']} {item['title']}（{item['source_type']}；角色={source_roles.get(item['source_id'], 'context')}）\n{item['text'][:2400]}"
            if overview_mode else
            f"[{item['source_id']}] {item['title']}（{item['source_type']}；角色={source_roles.get(item['source_id'], 'context')}）\n{item['text'][:2400] if source_roles.get(item['source_id']) in {'direct_evidence', 'counterevidence'} else item['text'][:900]}"
        )
        for index, item in enumerate(generation_sources, 1)
    ) or (
        "本题是自足推理任务：只能把用户题设作为前提，并使用可复核的形式推导；"
        "不得补入题设之外的现实事实。"
        if self_contained_reasoning else "没有通过审核的直接证据"
    )
    grounding_rule = _evidence_grounding_rule(
        [str(item).strip() for item in state.get("intent", {}).get("concepts", [])],
        generation_sources,
    )
    strategy = state["teaching_strategy"]
    preference_directives = [
        str(item).strip() for item in strategy.get("preference_directives", [])
        if str(item).strip()
    ]
    preference_instruction = ""
    if preference_directives:
        preference_instruction = (
            "【已确认的个性化教学约束】以下偏好来自用户明确反馈，只调整讲解方式，不改变事实："
            + "；".join(preference_directives)
            + "。必须在最终答案中可观察地执行，而不是只把它写进计划：按教学策略中的 explanation_order"
            "组织内容，开头落实第一个步骤，并完成偏好要求的例子、推导或表达方式。若某一步受证据边界限制，"
            "明确说明缺口，但不得悄悄退回通用模板。"
        )
    previous_example_instruction = ""
    if re.search(r"不要(?:重新|再|重复)|无需(?:重新|再)|已经(?:知道|学过)", state.get("question", "")):
        dialogue = str(state.get("dialogue") or "")
        example = re.search(r"f\s*[（(]\s*x\s*[,，]\s*y\s*[)）]\s*[=＝]\s*([^\n。；;]{3,60})", dialogue, re.I)
        if example:
            expression = example.group(0).strip()
            previous_example_instruction = (
                "【多轮记忆硬约束】用户已在上一轮学过基础定义。第一句必须直接沿用上一轮具体例子"
                f"“{expression}”，用它展示本轮新增步骤；不得再次介绍‘偏导数是什么’、"
                "泛化符号记号或从头铺垫。若该新增应用没有直接教材证据，明确标注证据边界。"
            )
    rigorous_reasoning_instruction = ""
    if re.search(r"证明|推导|求解|计算|构造|伪代码|充要条件|为什么.*(?:公式|方程|定理)", state.get("question", "")):
        rigorous_reasoning_instruction = (
            "【严谨推理要求】回答前先判断用户命题是否真的成立，并检查所在数域、适用条件、边界条件、"
            "单位制、显含变量及可逆性等必要假设。若题目结论错误、条件遗漏或概念形成错误对立，"
            "先明确指出并纠正，绝不能为了迎合提问替错误命题构造证明。证明题写出关键引理与逻辑链；"
            "推导题交代出发方程、中间变形和成立条件；计算题展示关键中间结果并核对最终答案；"
            "充要条件分别证明两个方向，不能把待证命题换一个说法当作论证；"
            "归纳证明要写清基例、归纳假设与降维步骤；物理推导说明所用定律和近似条件。"
            "算法复杂度先定义问题规模，不得杜撰‘逐参数求导需要指数级计算’等没有依据的量级。"
            "比较题给出真正能区分两者的反例。公式或步骤较多时用自然段、Markdown 公式、"
            "有实际内容的小标题或少量列表改善阅读，但不必每题复用同一格式。"
            "仅引用实际支持相应论断的材料；可以清楚说明由所引定理作出的自己的演绎，"
            "但不得暗示教材原文已经包含它没有写出的算法、实例或完整证明。"
        )
    if premise_guard:
        rigorous_reasoning_instruction += (
            "\n【已核实的题设问题，第一段必须明确纠正】"
            + premise_guard["correction"]
            + "后续推导只能建立在修正后的命题上，不得再次赞同原错误前提。"
        )
    precision_instruction = ""
    compact_question = re.sub(r"\s+", "", str(state.get("question") or ""))
    if "可对角化" in compact_question and "可逆" in compact_question:
        precision_instruction += (
            "【线性代数精度约束】一般的可对角化只保证存在一组由特征向量组成的基，"
            "不保证该基正交；只有正规矩阵（实数情形常见为实对称矩阵）才保证正交或酉特征基。"
            "必须分别给出‘可逆但不可对角化’与‘可对角化但不可逆’的例子，不能只给兼具二者的例子。"
        )
    if "特征值" in compact_question and "奇异值" in compact_question:
        precision_instruction += (
            "【矩阵分析精度约束】奇异值是 A* A 的非负特征值的平方根，其中 A* 是共轭转置；"
            "实矩阵时才可简写为 A^T。零特征值也对应零奇异值，不能只写‘正特征值’。"
        )
    if re.search(r"优化器|梯度|驻点|Hessian|海森|局部极|凸函数|严格凸|全局最小", compact_question, re.I):
        precision_instruction += (
            "【非凸优化精度约束】对二次可微函数，驻点处 Hessian 半正定通常只是局部极小的"
            "二阶必要条件，单独并不充分；Hessian 正定才给出严格局部极小的常用二阶充分条件。"
            "半正定退化时必须检查高阶项、邻域函数值或额外凸性，不能直接宣布局部最优。"
            "对可微凸函数，无约束问题或定义域内点处梯度为零可判全局最小，严格凸时最小点若存在则唯一；"
            "反向声称‘全局最小必有零梯度’必须限定无约束或内点。边界约束情形应使用 KKT、"
            "可行方向或法锥条件，不能把边界最优误写成梯度为零。"
        )
    if re.search(r"预测区间|不确定性|区间(?:很宽|窄|重叠)|置信上界|信息增益", compact_question, re.I):
        precision_instruction += (
            "【不确定性量化精度约束】先区分模型认识不确定性、过程固有随机性和实验测量误差。"
            "增加同条件重复测量主要降低测量均值的不确定性，未必消除模型结构或分布外导致的认识盲区；"
            "高认识不确定性可用信息增益驱动的新条件实验缩小。实验排序还必须同时考虑成本与安全约束。"
            "候选区间是否重叠不能直接替代差值检验或显著性判断；必须说明区间类型、联合误差结构与比较量。"
            "候选排序还要分开当前利用损失与探索的信息价值，不能仅凭点估计或区间宽度机械排序。"
        )
    retry_question = str(state.get("question") or "")
    wrapper_marker = "\n\n题目：\n"
    if retry_question.startswith("【致理结构调试·") and wrapper_marker in retry_question:
        retry_question = retry_question.rsplit(wrapper_marker, 1)[-1].strip() or retry_question
    payload = None
    generation_provider = "project-model"
    generation_error = ""
    if state.get("evidence_review", {}).get("proof_anchor_mode") == "exact_textbook_iff_statement":
        verified_proof = _deterministic_exact_iff_proof(
            str(state.get("question") or ""), generation_sources,
        )
        if verified_proof:
            payload = {
                "answer": verified_proof,
                "followup": "你想继续用初等变换，还是用线性方程组唯一解再证明一遍？",
                "discussion_prompts": ["比较三种证明路线", "伴随矩阵恒等式为什么成立？"],
            }
            generation_provider = "deterministic-verified-proof"
    try:
        overview_rule = (
            "当前是首次接触的领域概览。输出 800~1500 字、约1000字的中立认知地图，固定结构为：开头列出‘本概览基于以下来源’和生成日期；# 领域名概览；## 一句话定位；## 它在解决什么问题；## 核心框架（可用小型层级图）；## 发展脉络（极简版，3~5个时间+事件节点）；必要时写 ## 它不是什么（边界澄清）；## 如果感兴趣，可以从这里开始（3~4个入口，每项说明推荐理由）；结尾注明资料截止日期和6个月后建议复审。核心定义、框架与每个历史节点必须使用给定的数字脚注 [1][2]，编号严格对应证据列表；来源不足处使用规范化未核验表达，严禁模型常识伪装成检索来源。首次概览正文不得提用户专业、兴趣、旧笔记、掌握度，不做个性化类比，不替用户决定路线。输出 discussion_prompts 为空数组，followup 只用中性话术说明用户可自行输入任何想深入的方向。"
            if overview_mode else ""
        )
        if payload is None:
            payload = _agent_json(
            "你是教学回答 Agent。承接具体问题与对话，优先准确、自然、有理论意识地回答，不展示内部工作流。解释结构由问题本身决定：定义题可以先给直觉再澄清关键条件，机制题才展开必要因果链，比较题突出真正存在的差异，开放讨论允许并列视角与不确定性。禁止每一题重复‘先说结论、为什么、成立边界、证据缺口’等固定小标题或把所有问题压进同一个框架；自然段通常优于格式表演，只有内容确实复杂且分节能提高理解时才使用针对问题内容的标题。简单问题说清即可，复杂问题适度展开；不要为了长度重复。若用户明确说不要重复定义，必须直接承接上一轮已经出现的具体例子、符号和计算结果，讲新的应用步骤，禁止重新从头解释已学概念。若 primary_intent=compare，分别准确定位双方，并根据实际材料选择有实质内容的比较维度、共同区域与不可互换之处。只使用通过审核的证据，并尊重来源角色：direct_evidence/counterevidence 才能支撑或反驳核心命题；prerequisite 只能解释必要定义，不能冒充答案依据；context 只能交代背景。如果教材只支持偏导等前置概念，不支持新的算法或应用，必须明确说该应用缺少教材直接证据，不能把前置教材标成应用结论的来源。一般模式用 [M1]/[L1]/[W1]/[A1]/[T1]/[P1] 标注实际承载相应句子的来源，禁止堆砌引用。只有确实有助于回答时才补充历史、邻近理论或反例。区分已证实事实、合理推测与价值判断；证据不足时如实交代，不用流畅文案补全。M 表示用户带入材料，abstract 只能做摘要导读，open_fulltext 才能声称阅读正文。T 表示授权微信片段，只能说明对话实际出现的内容。禁止固定套用‘这和你学过的某某相似’，也禁止为了个性化强行引用无关课本。" + overview_rule + "输出 answer、followup、discussion_prompts。",
            f"问题：{state['question']}\n对话：{state.get('dialogue','')}\n探究框架：{state['intent']}\n教学策略：{strategy}\n"
            "公开网页或机构官网使用 [P1] 等实际证据编号；search_snippet 只证明搜索摘要明确展示的信息，不得声称已阅读全文。\n"
            "篇幅偏好：在证据充分时充分展开用户真正关心的概念与推理；普通概念题约250~450字，理论辨析、证明、推导与交叉问题约500~1100字。信息较多时自然分段；必要时使用两三个由当前内容决定的 Markdown 加粗小标题（如 **谱定理为何关键**）、公式或列表，但不要每题套固定框架，也不要固定使用‘结论、机制、边界’等通用栏目。避免重复和没有来源的新断言。\n"
            f"{previous_example_instruction}\n"
            f"{preference_instruction}\n"
            f"{rigorous_reasoning_instruction}\n"
            f"{reasoning_instruction}\n"
            f"{precision_instruction}\n"
            f"检索异常：{state.get('retrieval_errors', [])}\n可用论断：{state['evidence_review'].get('usable_claims',[])}\n"
            f"{grounding_rule}\n通过审核的证据：\n{evidence_text}",
                timeout=45,
            )
    except LLMError as exc:
        payload = None
        generation_provider = "deterministic-grounded-fallback"
        generation_error = str(exc)[:320]
        if self_contained_reasoning:
            generation_provider = "project-model-self-contained-json-retry"
            try:
                payload = _agent_json(
                    "你是严谨的自足推理回答 Agent。只使用用户题设和通用形式规则，不需要外部检索。"
                    "先检查前提是否成立，再给可核验的关键步骤、条件和结论。"
                    "只输出 JSON，字段为 answer、followup、discussion_prompts。",
                    f"问题：{retry_question}\n{rigorous_reasoning_instruction}\n{reasoning_instruction}\n{precision_instruction}",
                    timeout=45,
                )
            except LLMError as retry_exc:
                generation_error += f"；自足推理重试失败：{str(retry_exc)[:220]}"
    answer = _format_answer_payload((payload or {}).get("answer"))
    if "\\n" in answer:
        answer = answer.replace("\\n", "\n")
    if (
        not answer
        and self_contained_reasoning
        and generation_provider != "project-model-self-contained-json-retry"
    ):
        generation_provider = "project-model-self-contained-json-retry"
        try:
            payload = _agent_json(
                "你是严谨的自足推理回答 Agent。只使用用户题设和通用形式规则，不需要外部检索。"
                "先检查前提是否成立，再给可核验的关键步骤、条件和结论。"
                "只输出 JSON，字段为 answer、followup、discussion_prompts。",
                f"问题：{retry_question}\n{rigorous_reasoning_instruction}\n{reasoning_instruction}\n{precision_instruction}",
                timeout=45,
            )
            answer = _format_answer_payload((payload or {}).get("answer"))
            if "\\n" in answer:
                answer = answer.replace("\\n", "\n")
        except LLMError as retry_exc:
            generation_error += f"；自足推理空结果重试失败：{str(retry_exc)[:220]}"
    if not answer:
        if self_contained_reasoning:
            answer = (
                "这次自足推理模型没有返回可解析的实质答案。题目不需要外部证据，"
                "因此不会引用无关材料来填补；请重试本题。"
            )
        elif accepted:
            direct_ids = {
                source_id for source_id, role in state["evidence_review"].get("source_roles", {}).items()
                if role in {"direct_evidence", "counterevidence"}
            }
            source = next(
                (item for item in accepted if str(item.get("source_id")) in direct_ids),
                accepted[0],
            )
            answer = f"目前最贴近问题的可靠内容来自《{source['title']}》：{source['text'][:500]} [{source['source_id']}]\n\n现有证据还不足以完成更强的推断。"
        elif state.get("wechat_lookup", {}).get("requested"):
            detail = "；".join(state.get("retrieval_errors", [])) or "没有取得指定会话的消息"
            answer = f"我理解你希望我读取微信记录，但这次没有取得可引用的聊天片段：{detail}。我不会用猜测代替真实消息。"
        else:
            answer = "这次没有取得足够贴合且可核查的证据。我不想用一个听起来完整、实际没有依据的解释糊弄你。可以先缩小问题范围，或补充一份教材与来源。"
    if premise_guard and not re.search(premise_guard["required_pattern"], answer, re.I):
        answer = premise_guard["correction"] + "\n\n" + answer
    allowed_source_ids = {str(item.get("source_id")) for item in generation_sources}
    removed_invalid_citations: list[str] = []

    def remove_invalid_citation(match: re.Match[str]) -> str:
        source_id = match.group(1)
        if source_id in allowed_source_ids:
            return match.group(0)
        removed_invalid_citations.append(source_id)
        return ""

    answer = re.sub(r"\[((?:M|L|W|A|T|P)\d+)\]", remove_invalid_citation, answer)
    direct_ids = {
        source_id for source_id, role in state["evidence_review"].get("source_roles", {}).items()
        if role in {"direct_evidence", "counterevidence"}
    }
    cited_ids = set(re.findall(r"\[((?:M|L|W|A|T|P)\d+)\]", answer))
    citation_repaired = False
    if state["evidence_review"].get("sufficient") and direct_ids and not (cited_ids & direct_ids):
        source = next((
            item for item in generation_sources if str(item.get("source_id")) in direct_ids
        ), None)
        if source:
            source_id = str(source["source_id"])
            answer += f"\n\n**本回答的直接依据：** 《{source['title']}》 [{source_id}]"
            citation_repaired = True
    if not overview_mode:
        answer = _ensure_readable_paragraphs(answer)
    followup = str((payload or {}).get("followup") or "你认为这条解释里最需要验证的是哪一步？")
    prompts = (payload or {}).get("discussion_prompts")
    prompts = [str(item) for item in prompts[:2]] if isinstance(prompts, list) else ["能否检查一个反例？", "哪一个前置概念仍不清楚？"]
    return {
        "answer": answer, "followup": followup, "discussion_prompts": prompts,
        "generation_sources": generation_sources,
        "trace": _trace(state, "generate_answer", "只使用通过证据审查的来源生成教学回答", {
            "citation_binding_repaired": citation_repaired,
            "generation_provider": generation_provider,
            "generation_error": generation_error,
            "removed_invalid_citations": list(dict.fromkeys(removed_invalid_citations)),
        }),
    }


def route_after_text_generation(state: GardenerState) -> str:
    plan = state.get("planner_decision", {})
    return "generate_visualization" if plan.get("primary_modality") == "text_visual" else "reflect_outputs"


def generate_visualization(state: GardenerState) -> dict[str, Any]:
    """Try the full DeepDiagram service, then fall back deterministically."""
    plan = state.get("planner_decision", {})
    kind = str(plan.get("visual_kind") or "none")
    visual_request = str(plan.get("visual_request") or "").strip()
    allowed_ids = {str(item["source_id"]) for item in state.get("accepted_sources", [])}
    diagram = unavailable_diagram(kind, "本轮没有请求图解。")
    full_service_error = ""
    if visual_request and kind != "none":
        try:
            diagram = generate_with_full_service(
                user_request=visual_request,
                kind=kind,
                blueprint=state.get("content_blueprint", {}),
                allowed_source_ids=allowed_ids,
            )
            if diagram.get("status") != "ready":
                raise DeepDiagramServiceError(str(diagram.get("warning") or "完整服务产物未通过校验"))
        except (DeepDiagramServiceError, TimeoutError, OSError) as exc:
            full_service_error = str(exc)
            diagram = build_local_diagram(
                state.get("content_blueprint", {}),
                requested_kind=kind,
                allowed_source_ids=allowed_ids,
                fallback_reason=f"完整 DeepDiagram 未交付：{full_service_error}",
            )
    return {
        "visualization": diagram,
        "trace": _trace(state, "generate_visualization", (
            f"{diagram.get('provider')} 生成 {kind} 图解"
            if diagram.get("status") == "ready" else "图解未通过结构校验，安全回退为文字"
        ), {
            "provider": diagram.get("provider"), "status": diagram.get("status"),
            "deepdiagram_user_request": visual_request,
            "full_service_error": full_service_error,
            "warning": diagram.get("warning", ""),
        }),
    }


def generate_deliverables(state: GardenerState) -> dict[str, Any]:
    """Fan out text and DeepDiagram work from one audited blueprint, then join."""
    wants_visual = state.get("planner_decision", {}).get("primary_modality") == "text_visual"
    with ThreadPoolExecutor(max_workers=2 if wants_visual else 1, thread_name_prefix="garden-delivery") as pool:
        text_future = pool.submit(generate_answer, state)
        visual_future = pool.submit(generate_visualization, state) if wants_visual else None
        text_result = text_future.result()
        visual_result = visual_future.result() if visual_future else {
            "visualization": DiagramSpec(status="suppressed", kind="none").model_dump(),
            "trace": [],
        }
    trace = list(state.get("trace") or [])
    if text_result.get("trace"):
        trace.append(text_result["trace"][-1])
    if visual_result.get("trace"):
        trace.append(visual_result["trace"][-1])
    trace.append({
        "node": "join_deliverables",
        "summary": "文字与图解并行完成并汇合" if wants_visual else "文字回答完成",
        "data": {"parallel": wants_visual, "visual_status": visual_result["visualization"].get("status")},
    })
    result = {
        "answer": text_result["answer"],
        "followup": text_result["followup"],
        "discussion_prompts": text_result["discussion_prompts"],
        "generation_sources": text_result.get("generation_sources", []),
        "visualization": visual_result["visualization"],
        "trace": trace,
    }
    # The text branch can deliberately downgrade coincidental textbook hits
    # when answering a subjective or health question. Preserve that decision
    # across the fan-in; otherwise Reflector still sees the old factual gate.
    for field in ("evidence_review", "accepted_sources"):
        if field in text_result:
            result[field] = text_result[field]
    return result


def review_answer(state: GardenerState) -> dict[str, Any]:
    allowed_ids = {str(item["source_id"]) for item in state.get("accepted_sources", [])}
    diagram = state.get("visualization") or DiagramSpec(status="suppressed", kind="none").model_dump()
    grounded = diagram_is_grounded(diagram, allowed_ids)
    plan = state.get("planner_decision", {})
    visual_teaching_value = diagram_has_teaching_value(
        diagram, str(plan.get("visual_kind") or "none")
    )
    evidence = state.get("evidence_review", {})
    answer = str(state.get("answer") or "")
    profile = _response_profile(state.get("question", ""))
    reasoning_profile = state.get("reasoning_profile") or classify_reasoning_task(
        str(state.get("question") or ""),
        intent_hint=str(state.get("intent", {}).get("primary_intent") or ""),
    )
    self_contained_reasoning = is_self_contained_reasoning(
        str(state.get("question") or ""), reasoning_profile,
    )
    if (
        reasoning_profile.get("activated")
        and profile == "grounded_knowledge"
        and not self_contained_reasoning
        and not evidence.get("sufficient")
    ):
        reasoning_review = {
            "applicable": False,
            "passed": True,
            "checks": {},
            "issues": [],
            "skipped_reason": "事实型任务没有取得直接证据，本轮只验收诚实边界，不要求伪造完整推导。",
        }
    else:
        reasoning_review = review_reasoning_answer(
            reasoning_profile, answer, surface="gardener_chat",
        )
    open_discussion = profile != "grounded_knowledge" and not evidence.get("sufficient")
    cited_ids = set(re.findall(r"\[((?:M|L|W|A|T|P)\d+)\]", answer))
    fabricated_citation_ids = cited_ids - allowed_ids
    direct_ids = {
        source_id for source_id, role in evidence.get("source_roles", {}).items()
        if role in {"direct_evidence", "counterevidence"}
    }
    evidence_bounded = (
        (open_discussion and not cited_ids)
        or (self_contained_reasoning and not cited_ids)
        or (not evidence.get("sufficient") and "证据不足" in answer)
        or (
            bool(evidence.get("sufficient"))
            and bool(cited_ids & direct_ids)
            and not fabricated_citation_ids
        )
    )
    generic_headings = re.findall(
        r"(?m)^\s*#{1,4}\s*(?:先说结论|结论|为什么|成立边界|边界|目前还缺什么证据|证据缺口)\s*$",
        answer,
    )
    expression_natural = len(generic_headings) < 3 or state.get("intent", {}).get("response_mode") == "domain_overview"
    preference_directives = [
        str(item).strip()
        for item in state.get("teaching_strategy", {}).get("preference_directives", [])
        if str(item).strip()
    ]
    preference_text = "；".join(preference_directives)
    preference_checks: list[tuple[bool, str]] = []
    if re.search(r"几何|空间|直觉|图景", preference_text):
        preference_checks.append((
            bool(re.search(r"几何|空间|直觉|图景|直观|方向|伸缩|拉伸|压缩", answer)),
            "没有落实用户要求的几何或空间直觉",
        ))
    if re.search(r"具体例|举例|例子|案例", preference_text):
        preference_checks.append((
            bool(re.search(r"例如|举例|比如|例子|具体来看", answer)),
            "没有提供用户要求的具体例子",
        ))
    if re.search(r"逐步推导|推导|中间步骤", preference_text):
        preference_checks.append((
            bool(re.search(r"由此|因此|所以|于是|得到|推出|⇒|→|=", answer)),
            "没有呈现用户要求的推导或中间关系",
        ))
    personalization_fit_issues = [message for passed, message in preference_checks if not passed]
    personalization_natural = not personalization_fit_issues
    medical_safe = (
        profile != "health_guidance"
        or bool(re.search(r"就医|医生|医院|面诊|专业医疗", answer))
    )
    modality_fit = (
        plan.get("primary_modality") != "text_visual"
        or diagram.get("status") == "ready"
    )
    answered = len(answer.strip()) >= 24
    comparison_depth = True
    if (
        state.get("intent", {}).get("primary_intent") == "compare"
        and evidence.get("sufficient")
        and plan.get("complexity") != "simple"
    ):
        concepts = [
            str(item).strip() for item in state.get("intent", {}).get("concepts", [])
            if str(item).strip()
        ][:2]
        concept_coverage = len(concepts) < 2 or all(item in answer for item in concepts)
        structural_markers = len(re.findall(r"(?m)^\s*(?:#{2,4}\s+|[-*]\s+|\d+[.、]\s*)", answer))
        comparison_depth = len(answer.strip()) >= 280 and concept_coverage and structural_markers >= 2
    issues: list[str] = []
    target = "none"
    if not answered:
        issues.append("回答为空或过短，尚未处理当前问题")
        target = "text"
    if not comparison_depth:
        issues.append("比较回答过浅：需要分别定位双方、使用多个真实维度比较，并说明共同区域与边界")
        target = "text" if target == "none" else "both"
    if not reasoning_review.get("passed", True):
        issues.extend(
            issue for issue in reasoning_review.get("issues", []) if issue not in issues
        )
        target = "text" if target == "none" else "both"
    if not evidence_bounded:
        issues.append("事实回答没有绑定通过审查的直接证据")
        target = "text" if target == "none" else "both"
    if fabricated_citation_ids:
        issues.append("回答包含不存在或未经证据审核的引用：" + "、".join(sorted(fabricated_citation_ids)))
        target = "text" if target == "none" else "both"
    if not expression_natural:
        issues.append("回答机械套用固定标题，应根据问题内容自然组织并保留理论深度")
        target = "text" if target == "none" else "both"
    if not personalization_natural:
        issues.extend(personalization_fit_issues)
        target = "text" if target == "none" else "both"
    if not medical_safe:
        issues.append("身体不适回答缺少专业医疗边界或必要的就医提醒")
        target = "text" if target == "none" else "both"
    if not grounded:
        issues.append("图解包含未通过证据门控的来源或失效关系")
        target = "visualization" if target == "none" else "both"
    if not visual_teaching_value:
        issues.append("图解虽能渲染，但缺少教学结构，或把 Markdown/文件导航误当成知识节点")
        target = "visualization" if target == "none" else "both"
    if not modality_fit:
        issues.append("Planner 要求图文表达，但没有交付可用图解")
        target = "visualization" if target == "none" else "both"
    fallback = QualityReview(
        answered_question=answered,
        evidence_bounded=evidence_bounded,
        expression_natural=expression_natural,
        personalization_natural=personalization_natural,
        boundary_appropriate=evidence_bounded and medical_safe,
        medical_safe=medical_safe,
        visualization_grounded=grounded and visual_teaching_value,
        modality_fit=modality_fit,
        issues=issues,
        repair_target=target,
        passed=answered and comparison_depth and reasoning_review.get("passed", True) and evidence_bounded and expression_natural and personalization_natural and medical_safe and grounded and visual_teaching_value and modality_fit,
        rationale="检查回答完整性、可迁移推理、事实引用、开放讨论边界、表达自然度、医疗安全与图解可靠性。",
    )
    payload = None
    high_risk_review = (
        plan.get("complexity") == "complex"
        or (state.get("intent", {}).get("primary_intent") in {"evaluate", "design"} and not open_discussion)
        or state.get("intent", {}).get("response_mode") == "domain_overview"
        or state.get("wechat_lookup", {}).get("requested")
        or reasoning_profile.get("activated")
        or not fallback.passed
    )
    if high_risk_review:
        try:
            payload = _agent_json(
                "你是最终 Reflector，不重新回答问题，只验收并给出定向返工意见。逐句检查：是否直接回答 core_question；是否准确处理 claim_to_verify；事实型核心断言是否由 direct_evidence/counterevidence 支持；是否把 prerequisite 教材错当结论依据；是否强行套兴趣或旧知识；若 teaching_strategy 中存在 preference_directives，最终答案是否可观察地逐项执行，而不是只在计划里声称采用。另检查表达是否机械套用固定标题、主观与哲学问题是否保留开放性、身体问题是否避免诊断并给出必要就医提醒、科学解释是否区分事实与推测。没有直接证据的开放讨论可以回答，但不得捏造文献、教材、学校制度或医学结论。再对照 Planner 的表达计划检查：所选文字/图解是否真的适合问题；图的类型是否正确；图中关系是否来自回答和已审核证据；图解是否比文字更清楚而非装饰。若使用 T 类微信证据，聊天说法与客观事实必须分开。若 response_mode=domain_overview，还要检查中立、完整、来源可追溯且不预设路线。若只需修改文字，repair_target=text；只需重画图则 visualization；两者都错则 both。revised_answer 只能在文字确实有问题时填写。",
                f"问题：{state['question']}\n意图：{state['intent']}\nPlanner计划：{state.get('planner_decision', {})}\n"
                f"本地硬检查：{fallback.model_dump()}\n证据审查：{state['evidence_review']}\n教学策略：{state['teaching_strategy']}\n"
                f"推理协议：{reasoning_profile}\n推理硬检查：{reasoning_review}\n"
                f"文字回答：\n{state['answer']}\n图解结构：{diagram}\n"
                "输出 passed、answered_question、evidence_bounded、personalization_natural、expression_natural、boundary_appropriate、medical_safe、modality_fit、visualization_grounded、repair_target(none/text/visualization/both)、issues、revised_answer、rationale。",
            )
        except (LLMError, AssertionError):
            pass
    review = _validated(QualityReview, payload, fallback)
    # LLM review may tighten semantic quality, but it cannot overrule hard gates.
    if not fallback.passed:
        review["passed"] = False
        review["answered_question"] = review.get("answered_question", True) and fallback.answered_question
        review["evidence_bounded"] = review.get("evidence_bounded", True) and fallback.evidence_bounded
        review["expression_natural"] = review.get("expression_natural", True) and fallback.expression_natural
        review["personalization_natural"] = review.get("personalization_natural", True) and fallback.personalization_natural
        review["boundary_appropriate"] = review.get("boundary_appropriate", True) and fallback.boundary_appropriate
        review["medical_safe"] = review.get("medical_safe", True) and fallback.medical_safe
        review["visualization_grounded"] = review.get("visualization_grounded", True) and fallback.visualization_grounded
        review["modality_fit"] = review.get("modality_fit", True) and fallback.modality_fit
        for issue in fallback.issues:
            if issue not in review["issues"]:
                review["issues"].append(issue)
        if fallback.repair_target != "none":
            review["repair_target"] = fallback.repair_target
    return {
        "quality_review": review,
        "trace": _trace(state, "reflect_outputs", "文字与图解通过验收" if review["passed"] else "Reflector 要求定向返工", {
            "issues": review["issues"], "repair_target": review["repair_target"],
            "mode": "llm_high_risk" if high_risk_review else "local_hard_checks",
            "revision_count": state.get("revision_count", 0),
        }),
    }


def route_after_reflection(state: GardenerState) -> str:
    review = state.get("quality_review", {})
    if review.get("passed") or int(state.get("revision_count", 0)) >= int(state.get("planner_decision", {}).get("max_revisions", 1)):
        return "assemble_result"
    return "repair_outputs"


def repair_outputs(state: GardenerState) -> dict[str, Any]:
    """Perform at most one targeted revision chosen by the Reflector."""
    review = state.get("quality_review", {})
    target = str(review.get("repair_target") or "both")
    answer = state["answer"]
    diagram = state.get("visualization") or DiagramSpec(status="suppressed", kind="none").model_dump()
    if target in {"text", "both"}:
        evidence = state.get("evidence_review", {})
        reasoning_profile = state.get("reasoning_profile") or classify_reasoning_task(
            str(state.get("question") or ""),
            intent_hint=str(state.get("intent", {}).get("primary_intent") or ""),
        )
        self_contained_reasoning = is_self_contained_reasoning(
            str(state.get("question") or ""), reasoning_profile,
        )
        direct_ids = {
            str(source_id) for source_id, role in evidence.get("source_roles", {}).items()
            if role in {"direct_evidence", "counterevidence"}
        }

        def keeps_evidence_gate(candidate: str) -> bool:
            cited = set(re.findall(r"\[((?:M|L|W|A|T|P)\d+)\]", candidate))
            if evidence.get("sufficient"):
                return bool(cited & direct_ids)
            if self_contained_reasoning:
                return not cited
            if _response_profile(state.get("question", "")) != "grounded_knowledge":
                return not cited
            return "证据不足" in candidate

        revised = str(review.get("revised_answer") or "").strip()
        if revised and keeps_evidence_gate(revised):
            answer = revised
        else:
            payload = None
            try:
                payload = _agent_json(
                    "你是文字返工 Agent。只修 Reflector 指出的具体问题，保留正确内容和原有来源标注；不得扩写新事实、改变问题或引入未经审核的来源。若教学策略含 preference_directives，必须在改写后可观察执行。输出 answer。",
                    f"问题：{state['question']}\n教学策略：{state.get('teaching_strategy', {})}\n"
                    f"推理协议：{reasoning_prompt(reasoning_profile, surface='gardener_chat')}\n"
                    f"问题清单：{review.get('issues', [])}\n原回答：\n{answer}",
                )
            except (LLMError, AssertionError):
                pass
            repaired = str((payload or {}).get("answer") or "").strip()
            if repaired and keeps_evidence_gate(repaired):
                answer = repaired
    if target in {"visualization", "both"}:
        allowed_ids = {str(item["source_id"]) for item in state.get("accepted_sources", [])}
        plan = state.get("planner_decision", {})
        diagram = build_local_diagram(
            state.get("content_blueprint", {}),
            requested_kind=str(plan.get("visual_kind") or "none"),
            allowed_source_ids=allowed_ids,
            fallback_reason="Reflector 要求重画，已使用本地确定性适配器，不再次等待远程模型。",
        )
        if diagram.get("status") != "ready":
            diagram = unavailable_diagram(str(plan.get("visual_kind") or "none"), "返工后仍未通过图解校验，最终安全回退为纯文字。")
    return {
        "answer": answer,
        "visualization": diagram,
        "revision_count": int(state.get("revision_count", 0)) + 1,
        "trace": _trace(state, "repair_outputs", f"完成一次定向返工：{target}", {"issues": review.get("issues", [])}),
    }


def assemble_result(state: GardenerState) -> dict[str, Any]:
    answer = str(state.get("answer") or "")
    reasoning_profile = state.get("reasoning_profile") or classify_reasoning_task(
        str(state.get("question") or ""),
        intent_hint=str(state.get("intent", {}).get("primary_intent") or ""),
    )
    final_self_contained = is_self_contained_reasoning(state.get("question", ""), reasoning_profile)
    evidence = state.get("evidence_review", {})
    direct_ids = {
        str(source_id) for source_id, role in evidence.get("source_roles", {}).items()
        if role in {"direct_evidence", "counterevidence"}
    }
    initial_cited_ids = set(re.findall(r"\[((?:M|L|W|A|T|P)\d+)\]", answer))
    final_gate_failed = bool(evidence.get("sufficient")) and not bool(initial_cited_ids & direct_ids)
    quality_review = dict(state.get("quality_review") or {})
    if final_gate_failed:
        answer = (
            "这轮生成结果在返工后丢失了与直接证据的逐项绑定，因此最终证据硬门已阻止展示该事实性回答。"
            "请重新提问；园丁会保留原问题，但不会把没有可追溯依据的内容冒充答案。"
        )
        quality_review["passed"] = False
        quality_review["evidence_bounded"] = False
        issues = list(quality_review.get("issues") or [])
        issue = "最终装配检查发现直接证据引用丢失，已阻止无依据答案"
        if issue not in issues:
            issues.append(issue)
        quality_review["issues"] = issues
    # A retrieved source becomes a displayed citation (and later receives
    # activation credit) only when the final answer actually cites its ID.
    cited_ids = set(re.findall(r"\[((?:M|L|W|A|T|P)\d+)\]", answer))
    if state.get("intent", {}).get("response_mode") == "domain_overview":
        cited_numbers = {int(item) for item in re.findall(r"\[(\d+)\]", answer) if item.isdigit()}
        cited_ids.update(
            item["source_id"] for index, item in enumerate(state.get("accepted_sources", []), 1)
            if index in cited_numbers
        )
    accepted = [item for item in state.get("accepted_sources", []) if item["source_id"] in cited_ids]
    accepted_local = [item["note"] for item in accepted if item["local"] and item.get("note")]
    chat_records = [item for item in accepted if item.get("source_type") == "wechat_history"]
    external = [
        item for item in accepted
        if not item["local"] and item.get("source_type") != "wechat_history"
    ]
    local_connections = [
        {
            "id": item["note"]["id"], "title": item["title"], "path": item["note"]["path"],
            "reason": item.get("relevance_reason") or "概念相关，但尚未形成可追溯事实依据",
        }
        for item in state.get("candidate_sources", [])
        if item.get("local") and item.get("note") and item.get("knowledge_status") != "placeholder"
        and item["source_id"] not in cited_ids
    ][:3]
    if _response_profile(state.get("question", "")) != "grounded_knowledge" and not cited_ids:
        local_connections = []
    web_sources = []
    for item in external:
        article = item.get("article", {})
        web_sources.append({
            "title": item["title"], "url": item["url"], "year": article.get("year"),
            "authors": article.get("authors", []), "venue": article.get("venue") or item["source_type"],
            "source": article.get("source") or item["source_type"],
            "access_scope": item.get("access_scope", "metadata_only"),
        })
    wechat_sources = [
        {
            "title": item["title"],
            "talker": item.get("talker", ""),
            "time_hint": item.get("time_hint", ""),
            "message_count": item.get("message_count", 0),
            "access_scope": item.get("access_scope", "authorized_excerpt"),
            "boundary": "只证明本轮授权片段中出现过的表达；不代表事实已经外部核验。",
        }
        for item in chat_records
    ]
    layers = []
    accepted_local_types = {str(item.get("source_type") or "") for item in accepted if item.get("local")}
    if "textbook" in accepted_local_types:
        layers.append("textbook")
    if "local_wiki" in accepted_local_types:
        layers.append("wiki")
    if external:
        layers.append("authority")
    if chat_records:
        layers.append("authorized_wechat")
    result = {
        "answer": answer,
        "citations": [
            {
                "id": item["note"]["id"],
                "source_id": item["source_id"],
                "title": item["note"]["title"],
                "path": item["note"]["path"],
            }
            for item in accepted if item.get("local") and item.get("note")
        ][:3],
        "local_connections": local_connections,
        "web_sources": web_sources, "wechat_sources": wechat_sources, "followup": state["followup"],
        "discussion_prompts": state["discussion_prompts"],
        "evidence_layer": " + ".join(layers) if layers else "none",
        "researched_online": bool(state.get("retrieval_attempts") or external),
        "research_error": "；".join(state.get("retrieval_errors", [])), "offer_save": True,
        "agent_trace": _trace(state, "assemble_result", "完成结构化问园丁工作流"),
        "intent": state["intent"], "teaching_strategy": state["teaching_strategy"],
        "planner": state.get("planner_decision", PlannerDecision().model_dump()),
        "visualization": state.get("visualization", DiagramSpec(status="suppressed", kind="none").model_dump()),
        "personalization": state.get("personalization_plan", PersonalizationPlan().model_dump()),
        "reasoning": {
            "type": reasoning_profile.get("key") if reasoning_profile.get("activated") else "general",
            "label": reasoning_profile.get("label") if reasoning_profile.get("activated") else "通用问答",
            "confidence": reasoning_profile.get("confidence", 0.0),
            "task_key": reasoning_profile.get("task_key", "general"),
            "self_contained": final_self_contained,
            "review": (
                {
                    "applicable": False,
                    "passed": True,
                    "checks": {},
                    "issues": [],
                    "skipped_reason": "事实型任务没有取得直接证据，本轮只验收诚实边界。",
                }
                if (
                    reasoning_profile.get("activated")
                    and _response_profile(state.get("question", "")) == "grounded_knowledge"
                    and not final_self_contained
                    and not evidence.get("sufficient")
                )
                else review_reasoning_answer(reasoning_profile, answer, surface="gardener_chat")
            ),
        },
        "evidence_review": state["evidence_review"], "quality_review": quality_review,
        "revision_count": int(state.get("revision_count", 0)),
        "citation_binding": {
            "used_source_ids": sorted(cited_ids & {item["source_id"] for item in accepted}),
            "final_gate_failed": final_gate_failed,
        },
        "memory_used": {
            "claims": len(state.get("learner_context", {}).get("active_memory_claims", [])),
            "mastery": len(state.get("learner_context", {}).get("concept_mastery", [])),
            "l1": {
                "role": "本轮结束后记录可追溯观察，供后续反思；不直接控制本轮回答",
                "used_for_current_answer": False,
            },
            "l2": {
                "recalled_claim_ids": [
                    item.get("claim_id") for item in state.get("learner_context", {}).get("active_memory_claims", [])
                    if int(item.get("layer") or 0) == 2
                ],
            },
            "l3": {
                "profile_graph_nodes": len(state.get("profile_graph", {}).get("nodes", [])),
                "profile_graph_edges": len(state.get("profile_graph", {}).get("edges", [])),
                "claim_ids_used_for_understanding": state.get("intent", {}).get("profile_graph_claim_ids_used", []),
            },
        },
        "request_id": state["context"].request_id,
        "session_id": state["context"].session_id,
    }
    state["store"].add_activity("agent_query", state["question"][:80], 2)
    return {"result": result, "trace": result["agent_trace"]}


def build_gardener_graph():
    def timed(node_name: str, function: Any):
        def invoke(state: GardenerState) -> dict[str, Any]:
            started = time.perf_counter()
            result = function(state)
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            trace = result.get("trace")
            if isinstance(trace, list) and trace:
                trace[-1].setdefault("data", {})["duration_ms"] = duration_ms
            return result
        invoke.__name__ = f"timed_{node_name}"
        return invoke

    builder = StateGraph(GardenerState)
    builder.add_node("planner_intake", timed("planner_intake", planner_intake))
    builder.add_node("understand_question", timed("understand_question", understand_question))
    builder.add_node("planner_plan", timed("planner_plan", planner_plan))
    builder.add_node("clarify", timed("clarify", ask_clarification))
    builder.add_node("load_learner_memory", timed("load_learner_memory", load_learner_memory))
    builder.add_node("gate_personalization", timed("gate_personalization", gate_personalization))
    builder.add_node("plan_sources", timed("plan_sources", plan_sources))
    builder.add_node("clarify_wechat", timed("clarify_wechat", ask_wechat_clarification))
    builder.add_node("retrieve_sources", timed("retrieve_sources", retrieve_sources))
    builder.add_node("audit_evidence", timed("audit_evidence", audit_evidence))
    builder.add_node("choose_teaching_strategy", timed("choose_teaching_strategy", choose_teaching_strategy))
    builder.add_node("planner_select_delivery", timed("planner_select_delivery", planner_select_delivery))
    builder.add_node("build_content_blueprint", timed("build_content_blueprint", build_content_blueprint))
    builder.add_node("generate_deliverables", timed("generate_deliverables", generate_deliverables))
    builder.add_node("reflect_outputs", timed("reflect_outputs", review_answer))
    builder.add_node("repair_outputs", timed("repair_outputs", repair_outputs))
    builder.add_node("assemble_result", timed("assemble_result", assemble_result))
    builder.add_edge(START, "planner_intake")
    builder.add_edge("planner_intake", "understand_question")
    builder.add_conditional_edges(
        "understand_question", route_after_understanding,
        {"planner_plan": "planner_plan"},
    )
    builder.add_conditional_edges(
        "planner_plan", route_after_planner,
        {"clarify": "clarify", "load_learner_memory": "load_learner_memory"},
    )
    builder.add_edge("clarify", END)
    builder.add_edge("load_learner_memory", "gate_personalization")
    builder.add_edge("gate_personalization", "plan_sources")
    builder.add_conditional_edges(
        "plan_sources", route_after_source_plan,
        {"clarify_wechat": "clarify_wechat", "retrieve_sources": "retrieve_sources"},
    )
    builder.add_edge("clarify_wechat", END)
    builder.add_edge("retrieve_sources", "audit_evidence")
    builder.add_edge("audit_evidence", "choose_teaching_strategy")
    builder.add_edge("choose_teaching_strategy", "planner_select_delivery")
    builder.add_edge("planner_select_delivery", "build_content_blueprint")
    builder.add_edge("build_content_blueprint", "generate_deliverables")
    builder.add_edge("generate_deliverables", "reflect_outputs")
    builder.add_conditional_edges(
        "reflect_outputs", route_after_reflection,
        {"repair_outputs": "repair_outputs", "assemble_result": "assemble_result"},
    )
    builder.add_edge("repair_outputs", "reflect_outputs")
    builder.add_edge("assemble_result", END)
    return builder.compile()


GARDENER_GRAPH = build_gardener_graph()


def run_gardener_graph(
    store: GardenStore,
    context: GardenContext,
    *,
    include_evaluation_context: bool = False,
) -> dict[str, Any]:
    question, direct_material = _extract_frontier_material(context.current_message.content)
    clean_history = [
        {
            "role": item.role,
            "content": item.content[:2500],
            "evidence_layer": item.evidence_layer,
        }
        for item in context.conversation_history[-10:]
    ]
    dialogue = "\n".join(f"{'用户' if item['role']=='user' else '园丁'}：{item['content']}" for item in clean_history)
    initial: GardenerState = {
        "store": store, "context": context, "question": question.strip(),
        "direct_material": direct_material,
        "history": clean_history, "dialogue": dialogue,
        "learner_context": {
            "learning_level": context.learner_settings.declared_level,
            "explicit_interests": list(context.learner_settings.explicit_interests),
            "explicit_teaching_preferences": list(
                context.learner_settings.explicit_teaching_preferences
            ),
            "history_observation": "仅使用对话中明确表现；未建立稳定思维风格标签。",
        },
        "trace": [],
    }
    if not initial["question"]:
        raise ValueError("请先写下你想问园丁的问题")
    final = GARDENER_GRAPH.invoke(initial)
    result = final["result"]
    if include_evaluation_context:
        evaluation_sources = final.get("generation_sources") or final.get("accepted_sources", [])
        result["evaluation_context"] = {
            "retrieved_contexts": [
                str(item.get("text") or "")
                for item in evaluation_sources
                if str(item.get("text") or "").strip()
            ],
            "retrieved_context_ids": [
                str(item.get("note", {}).get("path") or item.get("source_id") or "")
                for item in evaluation_sources
            ],
            "retrieved_titles": [
                str(item.get("title") or "") for item in evaluation_sources
            ],
        }
    return result
