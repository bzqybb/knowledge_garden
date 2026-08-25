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
from core.query_understanding import build_query_plan
from core.retrieval import relevance_gate, search_notes
from core.storage import GardenStore
from core.tracememo import TraceMemoClient, TraceMemoError, tracememo_config
from core.web_research import fetch_open_access_pdf_text, search_academic_articles


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
        "official_docs", "wechat_history",
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
    if deep_planning_needed and not simple_fast_path:
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
                "llm_planner" if deep_planning_needed and not simple_fast_path
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
    text = re.sub(r"^(?:什么是|何为|请问什么是|请解释什么是)", "", text).strip(" ：:，,。？！? ")
    quoted = re.search(r"[《“\"]([^》”\"]{2,40})[》”\"]", text)
    if quoted:
        return quoted.group(1).strip()
    prefix = re.split(
        r"是不是|是否|是什么|指什么|为什么|为何|怎么|如何|的定义|的历史|的起源|的核心",
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
    profile_graph = LearningMemoryService(state["store"]).l3_profile_graph()
    profile_patterns = profile_graph.get("applicable_patterns", [])
    dialogue = state.get("dialogue", "")
    recent_user = next((
        str(item.get("content", "")).strip() for item in reversed(state.get("history", []))
        if item.get("role") == "user" and str(item.get("content", "")).strip()
    ), "")
    needs_resolution = bool(re.search(r"^(?:那|那么|这个|它|上述|刚才|其中)|这(?:个|种|一)" , question.strip()))
    resolved_fallback_question = f"{recent_user}\n当前追问：{question}" if recent_user and needs_resolution else question
    overview_mode = question.lstrip().startswith("【领域概览】")
    fallback_intent = "design" if re.search(r"构建|设计|结合", question) else "apply" if re.search(r"怎么用|如何实现|多少|求(?:出|解)?|计算|若|当", question) else "compare" if re.search(r"区别|比较", question) else "explain_mechanism" if re.search(r"为什么|机制|原理|基于什么", question) else "define"
    fallback = IntentResult(
        primary_intent=fallback_intent, task_demand="analyze" if fallback_intent in {"compare", "explain_mechanism"} else "understand",
        possible_obstacle="causal_gap" if fallback_intent == "explain_mechanism" else "unknown",
        evidence="离线规则只依据当前问题中的明确问法；未推断情绪或长期思维风格。",
        research_object=re.sub(r"^【严谨探究】", "", question).strip()[:120],
        core_question=resolved_fallback_question,
        claim_to_verify=(question if "【严谨探究】" in question else ""),
        response_mode="domain_overview" if overview_mode else "standard",
        first_exposure_evidence="用户主动从灵感跃迁请求建立领域概览。" if overview_mode else "",
    )
    payload = None
    understanding_provider = "deterministic-fallback"
    simple_payload = _simple_definition_payload(question)
    if simple_payload is not None:
        payload = simple_payload
        understanding_provider = "deterministic-simple-definition"
    else:
        if needs_resolution and recent_user:
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
            "保留数字、单位、范围、否定、前提和比较对象；只有歧义会改变答案时才澄清，不得猜测。"
            f"{few_shot}"
            "仅输出：primary_intent,research_object,concepts,needs_clarification,"
            "clarification_question,explicit_constraints,ambiguities,confidence。",
            f"已有对话（仅用于解析指代）：\n{dialogue or '无'}\n\n当前问题：{question}",
            )
        except LLMError:
            pass
    if payload is None and simple_payload is None:
        payload = _contextual_understanding_fallback(question, recent_user, fallback)
    intent = _validated(IntentResult, _normalize_understanding_payload(payload, fallback), fallback)
    intent["core_question"] = str(intent.get("core_question") or question).strip()
    research_object = str(intent.get("research_object") or "").strip()
    if (
        not research_object or len(research_object) > 40
        or re.search(r"[？?。！!]|是不是|是否|为什么|如何|怎么", research_object)
        or re.search(r"^(?:什么是|何为|请问什么是|请解释什么是)", research_object)
        or re.sub(r"\s+", "", research_object) == re.sub(r"\s+", "", question)
    ):
        intent["research_object"] = _question_subject(question)
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
        )
    ]))
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
    return {
        "intent": intent,
        "profile_graph": profile_graph,
        "trace": _trace(state, "understand_question", f"识别为 {intent['primary_intent']}，任务要求 {intent['task_demand']}", {
            "evidence": intent["evidence"],
            "understanding_provider": understanding_provider,
            "canonical_subject_candidate": canonical_subject,
            "candidate_aliases": candidate_aliases,
            "l3_profile_claim_ids_available": sorted(available_profile_ids),
            "l3_profile_claim_ids_used": intent["profile_graph_claim_ids_used"],
            "query_plan": query_plan,
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
    recalled = LearningMemoryService(state["store"]).active_memory_context(
        concepts,
        surface="gardener_chat",
        task_keys=[str(intent.get("primary_intent", "")), str(intent.get("task_demand", ""))],
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
    task_key = str(intent.get("primary_intent") or intent.get("task_demand") or "general")
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
    if intent["primary_intent"] in {"define", "clarify"}:
        fallback_types.append("encyclopedia")
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
    )
    use_hybrid = interactive_semantic or foundational_hybrid
    local_hits = search_notes(
        store, retrieval_question,
        kinds={"concept", "moc", "bridge", "knowledge", "course", "textbook"}, limit=8,
        query_plan=query_plan,
        # Direct definitions stay on the millisecond lexical path.  Composite
        # foundational questions automatically use BGE/FAISS + reranking;
        # other interactive questions may opt in through the environment flag.
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
    if not network_enabled and any(kind in plan["source_types"] for kind in ("encyclopedia", "review", "research_paper")):
        errors.append("联网检索已被 GARDEN_DISABLE_NETWORK 关闭")
    if "encyclopedia" in plan["source_types"] and "wikipedia" not in mounted_tools:
        errors.append("Wikipedia 工具未启用")
    if any(kind in plan["source_types"] for kind in ("review", "research_paper")) and "academic_search" not in mounted_tools:
        errors.append("学术检索工具未启用")
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
        }),
    }


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

    def comparison_subjects() -> list[str]:
        if intent.get("primary_intent") != "compare":
            return []
        question_text = str(intent.get("core_question") or question)
        patterns = (
            r"(.+?)\s*(?:与|和|跟|vs\.?|VS\.?)\s*(.+?)(?:的(?:核心)?(?:区别|差异|关系)|有什么(?:区别|差异)|相比|$)",
            r"比较\s*(.+?)\s*(?:与|和|跟)\s*(.+)",
        )
        for pattern in patterns:
            match = re.search(pattern, question_text, re.I)
            if match:
                values = [match.group(1), match.group(2)]
                cleaned = [re.sub(r"^(?:请|比较|解释)\s*", "", item).strip("？?。；;：: ") for item in values]
                if all(len(item) >= 2 for item in cleaned):
                    return cleaned[:2]
        concepts = [str(item).strip() for item in intent.get("concepts", []) if len(str(item).strip()) >= 2]
        return list(dict.fromkeys(concepts))[:2]

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
        note = item.get("note", {})
        reranker_score = float(note.get("reranker_score", 0.0))
        reranker_rank = int(note.get("reranker_rank", 0) or 0)
        semantic_passed = reranker_score >= 0.5 and 0 < reranker_rank <= 6
        # Preserve this auditable decision for the role-assignment phase. A
        # foundational exercise often composes several textbook terms, so the
        # full research-object phrase may never occur verbatim on the page.
        item["semantic_passed"] = semantic_passed
        item["relevance_score"] = max(float(item.get("relevance_score", 0.0)), relevance["score"])
        item["matched_terms"] = list(dict.fromkeys([*(item.get("matched_terms") or []), *relevance["matched_terms"]]))
        if not actual_text or access_scope == "metadata_only":
            hard_rejections.append({"source_id": source_id, "reason": "仅取得题录或标题，没有可核验内容"})
        elif (
            not relevance["passed"] and not semantic_passed and not item.get("explicitly_selected")
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
            item.get("source_type") == "textbook"
            and (
                object_grounded
                or (
                    item.get("semantic_passed") and bool(item.get("audit_matched_terms"))
                )
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
    fallback_ids = list(fallback_roles)[:5]
    fallback = EvidenceDecision(
        accepted_ids=fallback_ids,
        sufficient=any(role == "direct_evidence" for role in fallback_roles.values()),
        gaps=[] if any(role == "direct_evidence" for role in fallback_roles.values()) else ["只有前置或背景材料，没有取得直接支持待核验命题的证据"],
        rationale="离线审查只接纳具有实际内容的教材、本地知识、学术来源或用户授权聊天片段；聊天只证明对话中出现过什么说法，百科只作为术语入口。",
        source_roles=fallback_roles,
    )
    payload = None
    high_risk_audit = (
        state.get("planner_decision", {}).get("complexity") == "complex"
        or state["intent"].get("primary_intent") in {"evaluate", "design"}
        or state.get("wechat_lookup", {}).get("requested")
        or bool(re.search(r"适用边界|成立边界|适用条件|同时说明.*(?:边界|条件)|证据是否", question))
    )
    if candidates and high_risk_audit:
        source_text = "\n\n".join(
            f"[{item['source_id']}] 类型={item['source_type']} 权威={item['authority']} 标题={item['title']}\n{item['text'][:1200]}"
            for item in candidates
        )
        try:
            payload = _agent_json(
                "你是独立证据审查 Agent，不负责生成答案。先审查每个来源与 core_question/claim_to_verify 的论证关系，而不只看关键词相似。给通过来源标注唯一角色：direct_evidence（直接支持或反驳命题）、prerequisite（只解释必要前置）、counterevidence（反例或替代解释）、context（背景/历史）。教材和本地 Wiki 只有在明确支撑某个推理步骤时才可进入；不能因为用户学过它就把它生搬硬套成依据。逐条判断摘要是否真的支持。Wikipedia 只能用于术语与背景，不能单独证明争议机制；只有题录而没有摘要的论文不得推断结论。wechat_history 只能证明谁在何时表达过什么，不能把聊天中的说法自动当成客观事实。拒绝看似相关但没有回答问题的来源。sufficient 只有在至少一条 direct_evidence，或 direct_evidence 与 counterevidence 共同足以回答时才为 true。",
                f"当前探究框架：{state.get('intent', {})}\n问题：{question}\n\n候选来源：\n{source_text}\n"
                "输出 accepted_ids、rejected（source_id/reason）、usable_claims、gaps、sufficient、rationale、source_roles（source_id 到角色）。",
            )
        except LLMError:
            pass
    review = _validated(EvidenceDecision, payload, fallback)
    review["accepted_ids"] = [item for item in review["accepted_ids"] if item in hard_eligible_ids]
    allowed_roles = {"direct_evidence", "prerequisite", "counterevidence", "context"}
    review["source_roles"] = {
        source_id: role for source_id, role in (review.get("source_roles") or {}).items()
        if source_id in review["accepted_ids"] and role in allowed_roles
    }
    for source_id in review["accepted_ids"]:
        review["source_roles"].setdefault(source_id, fallback_roles.get(source_id, "context"))
    model_rejected = [item for item in review.get("rejected", []) if item.get("source_id") in hard_eligible_ids]
    review["rejected"] = [*hard_rejections, *model_rejected]
    accepted = [item for item in candidates if item["source_id"] in review["accepted_ids"]]
    direct_ids = {
        source_id for source_id, role in review["source_roles"].items()
        if role in {"direct_evidence", "counterevidence"}
    }
    review["sufficient"] = bool(accepted) and bool(review["sufficient"]) and bool(direct_ids)
    compared = comparison_subjects()
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
        "trace": _trace(state, "audit_evidence", f"通过 {len(accepted)}/{len(candidates)} 条候选证据", {"gaps": review["gaps"], "rationale": review["rationale"]}),
    }


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
    allowed_evidence_ids = {
        str(item.get("evidence_id")) for item in personalization.get("evidence", [])
        if item.get("evidence_id")
    }
    strategy["applied_evidence_ids"] = [
        value for value in strategy.get("applied_evidence_ids", [])
        if value in allowed_evidence_ids
    ]
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

    def comparison_subjects() -> list[str]:
        if intent.get("primary_intent") != "compare":
            return []
        question_text = str(intent.get("core_question") or state["question"])
        match = re.search(
            r"(.+?)\s*(?:与|和|跟|vs\.?|VS\.?)\s*(.+?)(?:的(?:核心)?(?:区别|差异|关系)|有什么(?:区别|差异)|相比|$)",
            question_text, re.I,
        )
        if match:
            values = [re.sub(r"^(?:请|比较|解释)\s*", "", item).strip("？?。；;：: ") for item in match.groups()]
            if all(values):
                return values[:2]
        return list(dict.fromkeys(
            str(item).strip() for item in intent.get("concepts", []) if str(item).strip()
        ))[:2]

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
        "comparison_subjects": comparison_subjects(),
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


def generate_answer(state: GardenerState) -> dict[str, Any]:
    accepted = state.get("accepted_sources", [])
    overview_mode = state.get("intent", {}).get("response_mode") == "domain_overview"
    if not state.get("evidence_review", {}).get("sufficient"):
        gaps = [str(item) for item in state.get("evidence_review", {}).get("gaps", []) if str(item).strip()]
        errors = [str(item) for item in state.get("retrieval_errors", []) if str(item).strip()]
        detail = "；".join(gaps[:2] + errors[:1]) or "没有取得能直接支持当前命题的教材、综述或其他权威正文"
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
    if overview_mode:
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

        def source_priority(item: dict[str, Any]) -> tuple[int, int, float]:
            corpus = f"{item.get('title', '')}\n{item.get('text', '')}"
            compact_corpus = compact(corpus)
            constraint_hits = sum(compact(value) in compact_corpus for value in constraints)
            alias_hits = sum(value.casefold() in corpus.casefold() for value in aliases)
            reranker_score = float(item.get("note", {}).get("reranker_score", 0.0) or 0.0)
            return constraint_hits, alias_hits, reranker_score

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
            and source_priority(direct_sources[0])[1] >= 2
            and source_priority(direct_sources[0])[1] > source_priority(direct_sources[1])[1]
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
    ) or "没有通过审核的直接证据"
    strategy = state["teaching_strategy"]
    payload = None
    generation_provider = "project-model"
    generation_error = ""
    try:
        overview_rule = (
            "当前是首次接触的领域概览。输出 800~1500 字、约1000字的中立认知地图，固定结构为：开头列出‘本概览基于以下来源’和生成日期；# 领域名概览；## 一句话定位；## 它在解决什么问题；## 核心框架（可用小型层级图）；## 发展脉络（极简版，3~5个时间+事件节点）；必要时写 ## 它不是什么（边界澄清）；## 如果感兴趣，可以从这里开始（3~4个入口，每项说明推荐理由）；结尾注明资料截止日期和6个月后建议复审。核心定义、框架与每个历史节点必须使用给定的数字脚注 [1][2]，编号严格对应证据列表；来源不足处使用规范化未核验表达，严禁模型常识伪装成检索来源。首次概览正文不得提用户专业、兴趣、旧笔记、掌握度，不做个性化类比，不替用户决定路线。输出 discussion_prompts 为空数组，followup 只用中性话术说明用户可自行输入任何想深入的方向。"
            if overview_mode else ""
        )
        payload = _agent_json(
            "你是教学回答 Agent。严格执行给定教学策略并承接对话，不展示内部工作流。第一句先直接回应 core_question，并准确复述待核验命题；随后建立清晰的因果链：前提→机制→结果→边界。排版也是教学的一部分：复杂回答使用 3~5 个简短 Markdown 小节，优先采用‘## 先说结论’‘## 为什么’‘## 成立边界’‘## 目前还缺什么证据’等语义标题；机制有多个环节时使用短列表。每段最多 3~4 句，禁止把“结论：机制：边界：证据缺口：”连续塞进一个大段落。简单定义题可以只用 2~3 个短段，不必机械凑齐所有标题。若问题只问一个数值、元件、对象或物理原理，控制在 100~220 字、最多两个小节，只写结论和必要推导；除非直接证据明确写出，否则不要添加高频效应、实际器件、例外条件或延伸机制。若 primary_intent=compare，不能只给一句抽象差异：先分别给出双方的准确定位，再使用 3~5 个有实质内容且由证据支持的比较维度（如研究对象、目标、方法、作用层级、应用或边界，按当前学科灵活选择），明确写出共同区域与不可互换之处，并给一个具体案例或反例；非简单比较题通常写 500~900 个汉字，但信息密度优先，不为凑字数重复。只使用通过审核的证据，并尊重来源角色：direct_evidence/counterevidence 才能支撑或反驳核心命题；prerequisite 只能解释必要定义，不能冒充答案依据；context 只能交代背景。一般模式用 [M1]/[L1]/[W1]/[A1]/[T1] 标注实际承载相应句子的来源，禁止在段末随意堆引用。若探究框架确实包含纵向问题，就用起源或演化解释为什么形成今天的机制；若包含横向问题，就比较真正相关的替代解释或邻近理论。不要把普通问答扩写成完整行业报告。先摆事实，再给判断；推测必须标明。M 表示用户带入材料，abstract 只能做摘要导读，open_fulltext 才能声称阅读正文。T 表示授权微信片段，只能说明对话实际出现的内容。证据不足要单独指出缺失环节，不用流畅文案补全。禁止固定套用‘这和你学过的某某相似’，也禁止为了个性化强行引用课本。" + overview_rule + "输出 answer、followup、discussion_prompts。",
            f"问题：{state['question']}\n对话：{state.get('dialogue','')}\n探究框架：{state['intent']}\n教学策略：{strategy}\n"
            f"检索异常：{state.get('retrieval_errors', [])}\n可用论断：{state['evidence_review'].get('usable_claims',[])}\n通过审核的证据：\n{evidence_text}",
            timeout=45,
        )
    except LLMError as exc:
        payload = None
        generation_provider = "deterministic-grounded-fallback"
        generation_error = str(exc)[:320]
    answer = str((payload or {}).get("answer") or "").strip()
    if "\\n" in answer:
        answer = answer.replace("\\n", "\n")
    if not answer:
        if accepted:
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
    direct_ids = {
        source_id for source_id, role in state["evidence_review"].get("source_roles", {}).items()
        if role in {"direct_evidence", "counterevidence"}
    }
    cited_ids = set(re.findall(r"\[((?:M|L|W|A|T)\d+)\]", answer))
    citation_repaired = False
    if state["evidence_review"].get("sufficient") and direct_ids and not (cited_ids & direct_ids):
        source = next((
            item for item in accepted if str(item.get("source_id")) in direct_ids
        ), None)
        if source:
            source_id = str(source["source_id"])
            answer += f"\n\n**本回答的直接依据：** 《{source['title']}》 [{source_id}]"
            citation_repaired = True
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
    return {
        "answer": text_result["answer"],
        "followup": text_result["followup"],
        "discussion_prompts": text_result["discussion_prompts"],
        "generation_sources": text_result.get("generation_sources", []),
        "visualization": visual_result["visualization"],
        "trace": trace,
    }


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
    cited_ids = set(re.findall(r"\[((?:M|L|W|A|T)\d+)\]", answer))
    direct_ids = {
        source_id for source_id, role in evidence.get("source_roles", {}).items()
        if role in {"direct_evidence", "counterevidence"}
    }
    evidence_bounded = (
        (not evidence.get("sufficient") and "证据不足" in answer)
        or (bool(evidence.get("sufficient")) and bool(cited_ids & direct_ids))
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
    if not evidence_bounded:
        issues.append("事实回答没有绑定通过审查的直接证据")
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
        visualization_grounded=grounded and visual_teaching_value,
        modality_fit=modality_fit,
        issues=issues,
        repair_target=target,
        passed=answered and comparison_depth and evidence_bounded and grounded and visual_teaching_value and modality_fit,
        rationale="先执行确定性的问答完整性、引用绑定、图结构与表达形态硬检查。",
    )
    payload = None
    high_risk_review = (
        plan.get("complexity") == "complex"
        or state.get("intent", {}).get("primary_intent") in {"evaluate", "design"}
        or state.get("intent", {}).get("response_mode") == "domain_overview"
        or state.get("wechat_lookup", {}).get("requested")
        or not fallback.passed
    )
    if high_risk_review:
        try:
            payload = _agent_json(
                "你是最终 Reflector，不重新回答问题，只验收并给出定向返工意见。逐句检查：是否直接回答 core_question；是否准确处理 claim_to_verify；每条核心断言是否由 direct_evidence/counterevidence 支持；是否把 prerequisite 教材错当结论依据；是否强行套兴趣或旧知识。再对照 Planner 的表达计划检查：所选文字/图解是否真的适合问题；图的类型是否正确；图中关系是否来自回答和已审核证据；图解是否比文字更清楚而非装饰。若使用 T 类微信证据，聊天说法与客观事实必须分开。若 response_mode=domain_overview，还要检查中立、完整、来源可追溯且不预设路线。若只需修改文字，repair_target=text；只需重画图则 visualization；两者都错则 both。revised_answer 只能在文字确实有问题时填写。",
                f"问题：{state['question']}\n意图：{state['intent']}\nPlanner计划：{state.get('planner_decision', {})}\n"
                f"本地硬检查：{fallback.model_dump()}\n证据审查：{state['evidence_review']}\n教学策略：{state['teaching_strategy']}\n"
                f"文字回答：\n{state['answer']}\n图解结构：{diagram}\n"
                "输出 passed、answered_question、evidence_bounded、personalization_natural、modality_fit、visualization_grounded、repair_target(none/text/visualization/both)、issues、revised_answer、rationale。",
            )
        except (LLMError, AssertionError):
            pass
    review = _validated(QualityReview, payload, fallback)
    # LLM review may tighten semantic quality, but it cannot overrule hard gates.
    if not fallback.passed:
        review["passed"] = False
        review["answered_question"] = review.get("answered_question", True) and fallback.answered_question
        review["evidence_bounded"] = review.get("evidence_bounded", True) and fallback.evidence_bounded
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
        direct_ids = {
            str(source_id) for source_id, role in evidence.get("source_roles", {}).items()
            if role in {"direct_evidence", "counterevidence"}
        }

        def keeps_evidence_gate(candidate: str) -> bool:
            cited = set(re.findall(r"\[((?:M|L|W|A|T)\d+)\]", candidate))
            if evidence.get("sufficient"):
                return bool(cited & direct_ids)
            return "证据不足" in candidate

        revised = str(review.get("revised_answer") or "").strip()
        if revised and keeps_evidence_gate(revised):
            answer = revised
        else:
            payload = None
            try:
                payload = _agent_json(
                    "你是文字返工 Agent。只修 Reflector 指出的具体问题，保留正确内容和原有来源标注；不得扩写新事实、改变问题或引入未经审核的来源。输出 answer。",
                    f"问题：{state['question']}\n问题清单：{review.get('issues', [])}\n原回答：\n{answer}",
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
    evidence = state.get("evidence_review", {})
    direct_ids = {
        str(source_id) for source_id, role in evidence.get("source_roles", {}).items()
        if role in {"direct_evidence", "counterevidence"}
    }
    initial_cited_ids = set(re.findall(r"\[((?:M|L|W|A|T)\d+)\]", answer))
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
    cited_ids = set(re.findall(r"\[((?:M|L|W|A|T)\d+)\]", answer))
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
        "citations": [{"id": item["id"], "title": item["title"], "path": item["path"]} for item in accepted_local[:3]],
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
        {"role": item.role, "content": item.content[:2500]}
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
