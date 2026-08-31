from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import multiprocessing as mp
import os
import re
import statistics
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from openai import AsyncOpenAI

from core.config import DB_PATH, llm_config
from core.inspiration import explore_inspiration
from evals.adapter import load_cases, temporary_store
from evals.adversarial_foundations_eval import answer_symbolic_grounding
from evals.boundary_eval import _run_turn
from evals.science_code_execution import execute_answer_in_docker
from evals.science_runtime_repair import repair_answer_with_retries
from evals.judge_config import judge_api_key, judge_base_url, judge_independence, judge_label, judge_model, judge_request_options


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "evals" / "datasets" / "zhili_dual_surface_5_v1.jsonl"
SYMBOLIC_DATASET = ROOT / "evals" / "datasets" / "zhili_symbolic_checks_v1.json"
REPORT_DIR = ROOT / "evals" / "reports"
RUBRIC_SCHEMA_VERSION = "dynamic-reference-rubric-v2"
JUDGE_PROMPT_VERSION = "heterogeneous-independent-blind-v5-runtime-trace"
SYMBOLIC_CHECKER_VERSION = "target-bound-sympy-v3"
DETERMINISTIC_ORACLE_VERSION = "science-math-static-v2"
SYMBOLIC_TIMEOUT_SECONDS = max(1.0, min(30.0, float(os.getenv("GARDEN_SYMBOLIC_TIMEOUT_SECONDS", "20"))))
SURFACE_TIMEOUT_SECONDS = max(30.0, min(300.0, float(os.getenv("GARDEN_EVAL_SURFACE_TIMEOUT_SECONDS", "180"))))
JUDGE_BATCH_TIMEOUT_SECONDS = max(30.0, min(300.0, float(os.getenv("GARDEN_EVAL_JUDGE_TIMEOUT_SECONDS", "210"))))
INFRASTRUCTURE_RETRIES = max(0, min(2, int(os.getenv("GARDEN_EVAL_INFRA_RETRIES", "1"))))
INFRASTRUCTURE_RETRY_BACKOFF_SECONDS = max(
    1.0, min(60.0, float(os.getenv("GARDEN_EVAL_INFRA_BACKOFF_SECONDS", "8"))),
)


CAPABILITY_DIMENSIONS = (
    "correctness", "derivation_rigor", "mechanism_discrimination",
    "uncertainty_calibration", "naturalness", "followup_value",
)

INFRASTRUCTURE_ERROR_RE = re.compile(
    r"(?:error code|status(?:_code)?|http)\s*[:=]?\s*(?:401|402|403|408|409|429|500|502|503|504)\b|"
    r"authentication(?:error)?|unauthorized|forbidden|quota|billing|rate.?limit|"
    r"timed?\s*out|timeout|outer_timeout|connection(?: error| reset| refused)?|dns|name resolution|"
    r"令牌.*(?:过期|不正确)|额度.*(?:耗尽|不足)|欠费|计费",
    re.IGNORECASE,
)
FATAL_INFRASTRUCTURE_ERROR_RE = re.compile(
    r"(?:error code|status(?:_code)?|http)\s*[:=]?\s*(?:401|402|403)\b|"
    r"authentication(?:error)?|unauthorized|forbidden|quota|billing|"
    r"令牌.*(?:过期|不正确)|额度.*(?:耗尽|不足)|欠费|计费",
    re.IGNORECASE,
)


def endpoint_host(url: str) -> str:
    return (urlparse(str(url)).hostname or "").casefold()


def classify_surface_infrastructure(result: dict[str, Any]) -> dict[str, Any]:
    """Separate provider/runtime outages from model capability failures."""
    if not bool(result.get("generation_failed")):
        return {"detected": False, "fatal": False, "category": "", "error_excerpt": ""}
    evidence = json.dumps({
        "trace": result.get("agent_trace", result.get("trace", [])),
        "diagnostics": result.get("generation_diagnostics", {}),
    }, ensure_ascii=False)
    match = INFRASTRUCTURE_ERROR_RE.search(evidence)
    if not match:
        return {"detected": False, "fatal": False, "category": "", "error_excerpt": ""}
    fatal = bool(FATAL_INFRASTRUCTURE_ERROR_RE.search(evidence))
    lower = evidence.casefold()
    if re.search(r"\b401\b|令牌.*(?:过期|不正确)|authentication|unauthorized", evidence, re.I):
        category = "credential"
    elif re.search(r"\b402\b|quota|billing|额度|欠费|计费", evidence, re.I):
        category = "quota"
    elif re.search(r"\b429\b|rate.?limit", evidence, re.I):
        category = "rate_limit"
    elif "timeout" in lower or "timed out" in lower:
        category = "timeout"
    else:
        category = "connection"
    excerpt_start = max(0, match.start() - 100)
    return {
        "detected": True, "fatal": fatal, "category": category,
        "error_excerpt": evidence[excerpt_start:match.end() + 260],
    }


def refresh_row_infrastructure(row: dict[str, Any]) -> None:
    """Backfill infrastructure labels for legacy reports before rescoring."""
    for surface in ("gardener", "inspiration"):
        result = row.setdefault(surface, {})
        result["infrastructure_failure"] = classify_surface_infrastructure(result)


def assert_expected_runtime() -> dict[str, str]:
    """Fail before generation if the benchmark lane was silently reconfigured."""
    generator = llm_config()
    actual = {
        "generator_model": generator.model,
        "generator_host": endpoint_host(generator.base_url),
        "judge_model": judge_model(),
        "judge_host": endpoint_host(judge_base_url(judge_model())),
    }
    expected = {
        "generator_model": os.getenv("EVAL_EXPECTED_GENERATOR_MODEL", "").strip(),
        "generator_host": os.getenv("EVAL_EXPECTED_GENERATOR_HOST", "").strip().casefold(),
        "judge_model": os.getenv("EVAL_EXPECTED_JUDGE_MODEL", "").strip(),
        "judge_host": os.getenv("EVAL_EXPECTED_JUDGE_HOST", "").strip().casefold(),
    }
    mismatches = [
        f"{key}: expected={value!r}, actual={actual[key]!r}"
        for key, value in expected.items() if value and actual[key] != value
    ]
    if mismatches:
        raise RuntimeError("评测模型/端点断言失败；拒绝启动：" + "; ".join(mismatches))
    return actual


def normalize_capability_case(case: dict[str, Any]) -> dict[str, Any]:
    """Add a deterministic frozen envelope for task-only benchmark rows."""
    normalized = dict(case)
    question = str(normalized.get("question") or "").strip()
    if not question:
        raise ValueError(f"{normalized.get('id')}: question 不能为空")
    normalized.setdefault("reference", (
        f"任务目标是：{question}。"
        "合格回答应从明确假设和适用条件出发完成原理推导。"
        "应给出可执行代码或明确的科学工具调用方案，并报告依赖与输入。"
        "应使用数值、符号、单元测试、守恒律或交叉实现中的至少一种验证结果。"
        "应区分实际执行结果、预期结果与未执行部分，最后反思模型边界和可能误差。"
    ))
    normalized.setdefault("rubric", [
        "针对题目指定对象给出正确、连贯且写明假设的原理推导，不以泛泛综述替代。",
        "提供可执行的代码或工具调用步骤，明确输入、依赖、参数与输出；不得把未执行代码宣称为已运行。",
        "给出可审计的结果验证，至少包含数值、符号、测试、守恒律或独立实现中的一种。",
        "解释结果的物理或数学意义，并反思近似、适用边界、失败模式与下一步改进。",
        "完整满足题目中点名的算法、模型、数据对象和目标量；缺少必要输入时应显式说明并给出可复现实例。",
    ])
    normalized.setdefault("reference_risk", "medium")
    normalized.setdefault(
        "requires_tool_execution",
        str(normalized.get("id") or "").startswith("SCI-"),
    )
    if str(normalized.get("id") or "").startswith("SCI-"):
        normalized.setdefault(
            "forbidden_routes",
            ["prepare_model_knowledge", "plan_sources", "retrieve_sources", "audit_sources"],
        )
    return normalized


def deterministic_answer_oracle(case_id: str, answer: str) -> dict[str, Any]:
    """Catch executable-code and known mathematical defects without an LLM judge."""
    text = str(answer or "")
    issues: list[str] = []
    python_blocks = re.findall(r"```(?:python|py)\s*\n(.*?)```", text, flags=re.I | re.S)
    if str(case_id).startswith("SCI-MATH-"):
        if not python_blocks:
            issues.append("NO_AUDITABLE_PYTHON_BLOCK")
        for index, block in enumerate(python_blocks, 1):
            try:
                ast.parse(block)
            except SyntaxError as exc:
                issues.append(f"PYTHON_SYNTAX_ERROR_BLOCK_{index}_LINE_{exc.lineno}")
    if case_id == "SCI-MATH-02" and (
        "n ∈ {1,2,3,5,7,9" in text and "阶数确实决定" in text
    ):
        issues.append("GROUP_ORDER_9_FALSE_UNIQUENESS_CLAIM")
    if case_id == "SCI-MATH-03" and "np.linalg.norm(v)" in text and (
        "双曲" in text or "hyperbol" in text.casefold()
    ):
        issues.append("HYPERBOLIC_TANGENT_USES_EUCLIDEAN_NORM")
    if case_id == "SCI-MATH-03" and re.search(
        r"bounds\s*=\s*\([^\)]*3\s*\*\s*np\.pi", text,
    ):
        issues.append("SPHERE_SHOOTING_INTERVAL_INCLUDES_NONSHORTEST_BRANCHES")
    if case_id == "SCI-MATH-03" and "np.dot(A, B)" in text and (
        "闵氏" in text or "Minkowski" in text or "Lorentz" in text
    ):
        issues.append("HYPERBOLIC_PROJECTION_USES_EUCLIDEAN_DOT")
    if case_id == "SCI-MATH-08":
        if "Delta*" in text:
            issues.append("UNDEFINED_DELTA_EXPRESSION")
        if "tau[m*n]" in text and re.search(r"default_prec\s*=\s*60", text):
            issues.append("TAU_COEFFICIENT_RANGE_CAN_EXCEED_PRECISION")
    if case_id == "SCI-MATH-10" and "tour.append(tour)" in text:
        issues.append("SELF_REFERENTIAL_TOUR_APPEND")
    return {
        "version": DETERMINISTIC_ORACLE_VERSION,
        "applicable": str(case_id).startswith("SCI-MATH-"),
        "python_blocks": len(python_blocks),
        "issues": issues,
        "passed": not issues if str(case_id).startswith("SCI-MATH-") else None,
    }


def local_surface_checks(answer: str, *, surface: str, case_id: str = "") -> dict[str, Any]:
    text = str(answer or "")
    return {
        "nonempty": bool(text.strip()),
        "substantive": len("".join(text.split())) >= 180,
        "has_equation": any(mark in text for mark in ("=", "\\(", "$$")),
        "has_reasoning_link": any(mark in text for mark in ("因此", "所以", "由", "从而", "故")),
        "has_limits": any(mark in text for mark in ("条件", "假设", "仅当", "边界", "需要", "并不")),
        "not_defensive_refusal": not any(mark in text for mark in (
            "无法回答", "不能回答", "缺少资料无法", "请提供教材",
            "没有返回可解析", "请重试本题", "先不补写答案",
        )),
        "inspiration_has_branches": None if surface == "gardener" else True,
        # Model-written code is not execution evidence.  This remains false until
        # the answer surface exposes a trusted sandbox/tool trace to the evaluator.
        "tool_execution_verified": False,
        "deterministic_oracle": deterministic_answer_oracle(case_id, text),
    }


def attach_runtime_code_execution(row: dict[str, Any]) -> None:
    """Attach trusted Docker traces without exposing model code to the host."""
    timeout_seconds = max(
        1.0, min(120.0, float(os.getenv("GARDEN_EVAL_CODE_TIMEOUT_SECONDS", "20")))
    )
    max_output_bytes = max(
        4096, min(131_072, int(os.getenv("GARDEN_EVAL_CODE_MAX_OUTPUT_BYTES", "32768")))
    )
    traces: dict[str, Any] = {}
    for surface in ("gardener", "inspiration"):
        execution = execute_answer_in_docker(
            str(row.get(surface, {}).get("answer") or ""),
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
        trace = {
            "status": execution.get("status"),
            "reason": execution.get("reason"),
            "executed": bool(execution.get("executed")),
            "backend": execution.get("backend"),
            "image": execution.get("image"),
            "exit_code": execution.get("exit_code"),
            "duration_seconds": execution.get("duration_seconds"),
            "timed_out": bool(execution.get("timed_out")),
            "output_limited": bool(execution.get("output_limited")),
            "blocks_combined": execution.get("blocks_combined", 0),
            "code_sha256": execution.get("combined_code_sha256") or execution.get("code_sha256"),
            "stdout": str(execution.get("stdout") or "")[:max_output_bytes],
            "stderr": str(execution.get("stderr") or "")[:max_output_bytes],
            "audit_decision": execution.get("answer_audit", {}).get("decision"),
        }
        local_checks = row.setdefault(surface, {}).setdefault("local_checks", {})
        local_checks["tool_execution"] = trace
        local_checks["tool_execution_verified"] = trace["status"] == "passed"
        traces[surface] = trace
    append_audit_event(row, "sandbox_code_execution_completed", {
        "backend": "docker", "timeout_seconds": timeout_seconds,
        "max_output_bytes": max_output_bytes, "surfaces": traces,
    })


def repair_failed_runtime_answers(row: dict[str, Any]) -> None:
    """Attempt one model repair, accepting it only after runtime and oracle gates."""

    timeout_seconds = max(
        1.0, min(120.0, float(os.getenv("GARDEN_EVAL_CODE_TIMEOUT_SECONDS", "20")))
    )
    max_attempts = max(
        1, min(3, int(os.getenv("GARDEN_EVAL_CODE_REPAIR_ATTEMPTS", "3")))
    )
    outcomes: dict[str, Any] = {}
    for surface in ("gardener", "inspiration"):
        surface_data = row.setdefault(surface, {})
        initial_answer = str(surface_data.get("answer") or "")
        initial_execution = surface_data.get("local_checks", {}).get("tool_execution", {})
        existing_repair = surface_data.get("runtime_repair", {})
        prior_attempts = list(existing_repair.get("attempts") or [])
        repair_answer = initial_answer
        repair_execution = initial_execution
        if prior_attempts and not existing_repair.get("accepted"):
            latest = prior_attempts[-1]
            latest_answer = str(latest.get("candidate_answer") or "").strip()
            latest_execution = latest.get("candidate_execution") or {}
            if latest_answer and latest_execution.get("reason") == "nonzero_exit":
                repair_answer = latest_answer
                repair_execution = latest_execution
            else:
                prior_attempts = []
        remaining_attempts = max_attempts - len(prior_attempts)
        if remaining_attempts <= 0:
            outcome = existing_repair
        else:
            outcome = repair_answer_with_retries(
                question=str(row.get("question") or ""),
                answer=repair_answer,
                execution=repair_execution,
                timeout_seconds=timeout_seconds,
                max_attempts=remaining_attempts,
            )
            if prior_attempts:
                new_attempts = list(outcome.get("attempts") or [])
                for index, attempt in enumerate(new_attempts, len(prior_attempts) + 1):
                    attempt["attempt"] = index
                outcome["attempts"] = prior_attempts + new_attempts
                outcome["max_attempts"] = max_attempts
                outcome["initial_answer_sha256"] = existing_repair.get(
                    "initial_answer_sha256", outcome.get("initial_answer_sha256")
                )
                outcome["initial_execution"] = existing_repair.get(
                    "initial_execution", outcome.get("initial_execution")
                )
        candidate = str(outcome.get("candidate_answer") or "")
        if outcome.get("accepted") and candidate:
            candidate_checks = local_surface_checks(
                candidate, surface=surface, case_id=str(row.get("id") or ""),
            )
            candidate_oracle = candidate_checks.get("deterministic_oracle", {})
            outcome["candidate_deterministic_oracle"] = candidate_oracle
            if candidate_oracle.get("applicable") and not candidate_oracle.get("passed"):
                outcome["accepted"] = False
                outcome["reason"] = "candidate_deterministic_oracle_failed"
            elif not candidate_checks.get("not_defensive_refusal", True):
                outcome["accepted"] = False
                outcome["reason"] = "candidate_meta_refusal"
            else:
                surface_data["answer_before_runtime_repair"] = initial_answer
                surface_data["answer"] = candidate
                surface_data["local_checks"] = candidate_checks
                candidate_execution = outcome.get("candidate_execution", {})
                surface_data["local_checks"]["tool_execution"] = candidate_execution
                surface_data["local_checks"]["tool_execution_verified"] = True
                surface_data["revision_count"] = int(surface_data.get("revision_count", 0)) + 1
        surface_data["runtime_repair"] = outcome
        outcomes[surface] = {
            "eligible": bool(outcome.get("eligible")),
            "attempted": bool(outcome.get("attempted")),
            "accepted": bool(outcome.get("accepted")),
            "reason": outcome.get("reason"),
        }
    append_audit_event(row, "runtime_code_repair_completed", {
        "max_attempts_per_surface": max_attempts,
        "surfaces": outcomes,
    })


def freeze_rubric(case: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    criteria = [str(item).strip() for item in case.get("rubric", []) if str(item).strip()]
    if not 3 <= len(criteria) <= 5:
        raise ValueError(f"{case.get('id')}: scoring rubric 必须在回答前冻结为 3-5 项")
    weight = round(1 / len(criteria), 6)
    reference_atoms = [
        item.strip() for item in re.split(r"[。；;]\s*", str(case.get("reference") or ""))
        if len(item.strip()) >= 6
    ]
    rubric = [
        {
            "id": f"R{index}", "criterion": criterion, "weight": weight,
            "reference_anchor": reference_atoms[min(index - 1, len(reference_atoms) - 1)] if reference_atoms else "",
            "provenance": "reference_decomposition_before_answer",
        }
        for index, criterion in enumerate(criteria, 1)
    ]
    rubric[-1]["weight"] = round(1 - sum(item["weight"] for item in rubric[:-1]), 6)
    envelope = {
        "schema_version": RUBRIC_SCHEMA_VERSION,
        "case_id": str(case.get("id") or ""),
        "question_sha256": hashlib.sha256(str(case.get("question") or "").encode("utf-8")).hexdigest(),
        "reference_sha256": hashlib.sha256(str(case.get("reference") or "").encode("utf-8")).hexdigest(),
        "rubric": rubric,
    }
    digest = hashlib.sha256(
        json.dumps(envelope, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return rubric, digest


def verify_frozen_rubric(row: dict[str, Any]) -> None:
    expected_rubric, expected_hash = freeze_rubric(row)
    if row.get("scoring_rubric") != expected_rubric or row.get("rubric_hash") != expected_hash:
        raise ValueError(f"{row.get('id')}: rubric envelope 缺失或已被篡改，拒绝重裁")


def audit_event(run_id: str, case_id: str, seq: int, event: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run_id, "case_id": case_id, "seq": seq,
        "timestamp": datetime.now().astimezone().isoformat(),
        "event": event, "data": data,
    }


def append_audit_event(row: dict[str, Any], event: str, data: dict[str, Any]) -> None:
    events = row.setdefault("audit_events", [])
    events.append(audit_event(
        str(row.get("run_id") or "unknown-run"), str(row.get("id") or "unknown-case"),
        len(events) + 1, event, data,
    ))


def build_blind_judge_payload(row: dict[str, Any], surface: str) -> dict[str, Any]:
    """Build the only payload visible to one judge call; never include peer identity/answer."""
    if surface not in {"gardener", "inspiration"}:
        raise ValueError(f"unknown surface: {surface}")
    return {
        "case_id": row["id"], "discipline": row["discipline"], "question": row["question"],
        "reference": row["reference"], "reference_risk": row["reference_risk"],
        "scoring_rubric": row["scoring_rubric"], "rubric_hash": row["rubric_hash"],
        "anonymous_answer": row[surface]["answer"],
        "symbolic_grounding": row[surface].get("symbolic_grounding", {}),
        "requires_tool_execution": bool(row.get("requires_tool_execution", False)),
        "tool_execution_verified": bool(
            row[surface].get("local_checks", {}).get("tool_execution_verified", False)
        ),
        "deterministic_oracle": row[surface].get("local_checks", {}).get("deterministic_oracle", {}),
        "trusted_tool_execution": row[surface].get("local_checks", {}).get("tool_execution", {}),
    }


def enforce_hard_failure_gates(row: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Make deterministic execution failures authoritative over semantic judging."""
    adjusted = result
    failures_by_surface = adjusted.setdefault("failures", {})
    for surface in ("gardener", "inspiration"):
        dimensions = adjusted.setdefault(f"{surface}_dimensions", {})
        failures = failures_by_surface.get(surface)
        if not isinstance(failures, list):
            failures = [] if failures is None else [str(failures)]
            failures_by_surface[surface] = failures
        infrastructure = row.get(surface, {}).get("infrastructure_failure", {})
        if infrastructure.get("detected"):
            adjusted[f"{surface}_verdict"] = "unscorable"
            adjusted[f"{surface}_score"] = None
            adjusted[f"{surface}_dimensions"] = {}
            marker = f"INFRASTRUCTURE_FAILURE_UNSCORABLE:{infrastructure.get('category') or 'unknown'}"
            if marker not in failures:
                failures.append(marker)
            continue
        if row.get(surface, {}).get("generation_failed"):
            adjusted[f"{surface}_verdict"] = "fail"
            adjusted[f"{surface}_score"] = 0.0
            for key in CAPABILITY_DIMENSIONS:
                dimensions[key] = 1
            failures.append("GENERATION_FAILED_HARD_GATE")
        symbolic = row.get(surface, {}).get("symbolic_grounding", {})
        if symbolic.get("applicable") and symbolic.get("passed") is False:
            adjusted[f"{surface}_verdict"] = "fail"
            adjusted[f"{surface}_score"] = min(float(adjusted.get(f"{surface}_score", 0.0)), 50.0)
            dimensions["correctness"] = 1
            failures.append("SYMBOLIC_GROUNDING_FAILED_HARD_GATE")
        execution_verified = bool(
            row.get(surface, {}).get("local_checks", {}).get("tool_execution_verified", False)
        )
        if row.get("requires_tool_execution") and not execution_verified:
            execution_trace = row.get(surface, {}).get("local_checks", {}).get("tool_execution", {})
            execution_ids = {
                str(item.get("id")) for item in row.get("scoring_rubric", [])
                if any(mark in str(item.get("criterion") or "") for mark in (
                    "可执行", "工具调用", "实际执行", "结果验证", "运行结果",
                ))
            }
            for item in adjusted.get("rubric_results", []):
                if str(item.get("rubric_id")) in execution_ids:
                    score_key = f"{surface}_score"
                    item[score_key] = min(float(item.get(score_key, 0.0)), 1.0)
            weighted = 0.0
            for rubric in row.get("scoring_rubric", []):
                judged = next((
                    item for item in adjusted.get("rubric_results", [])
                    if str(item.get("rubric_id")) == str(rubric.get("id"))
                ), {})
                weighted += float(judged.get(f"{surface}_score", 0.0)) * float(rubric.get("weight", 0.0))
            adjusted[f"{surface}_score"] = min(
                float(adjusted.get(f"{surface}_score", 0.0)),
                round(weighted * 50.0, 2),
            )
            adjusted[f"{surface}_verdict"] = "fail"
            adjusted[f"{surface}_score"] = min(
                float(adjusted.get(f"{surface}_score", 0.0)), 60.0,
            )
            marker = "TOOL_EXECUTION_NOT_VERIFIED_HARD_GATE"
            if marker not in failures:
                failures.append(marker)
            if execution_trace.get("executed") and execution_trace.get("status") == "failed":
                dimensions["correctness"] = 1
                adjusted[f"{surface}_score"] = min(
                    float(adjusted.get(f"{surface}_score", 0.0)), 50.0,
                )
                detail = f"SANDBOX_EXECUTION_FAILED:{execution_trace.get('reason') or 'unknown'}"
                if detail not in failures:
                    failures.append(detail)
        oracle = row.get(surface, {}).get("local_checks", {}).get("deterministic_oracle", {})
        if oracle.get("applicable") and oracle.get("passed") is False:
            adjusted[f"{surface}_verdict"] = "fail"
            adjusted[f"{surface}_score"] = min(float(adjusted.get(f"{surface}_score", 0.0)), 50.0)
            dimensions["correctness"] = 1
            marker = "DETERMINISTIC_ANSWER_ORACLE_FAILED"
            if marker not in failures:
                failures.append(marker)
            for issue in oracle.get("issues", []):
                detail = f"ORACLE:{issue}"
                if detail not in failures:
                    failures.append(detail)
    adjusted["comparison"] = (
        f"硬门控后程序聚合分：问园丁 {adjusted.get('gardener_score', 0.0)}，"
        f"灵感检测 {adjusted.get('inspiration_score', 0.0)}；生成失败计 0，符号失败最高 50，"
        "基础设施故障不计入能力分母；要求实际执行但无可信工具日志时 verdict=fail、总分最高 60；"
        "确定性 oracle 失败最高 50。"
    )
    return adjusted


def attach_symbolic_checks(
    case: dict[str, Any], symbolic_dataset: Path = SYMBOLIC_DATASET,
) -> dict[str, Any]:
    if "symbolic_checks" in case:
        return {**case, "symbolic_checks": list(case.get("symbolic_checks") or [])}
    checks_by_id = json.loads(symbolic_dataset.read_text(encoding="utf-8"))
    return {**case, "symbolic_checks": list(checks_by_id.get(str(case.get("id")), []))}


def _symbolic_worker(send_conn: Any, case: dict[str, Any], answer: str) -> None:
    try:
        send_conn.send({"ok": True, "result": answer_symbolic_grounding(case, answer)})
    except BaseException as exc:
        send_conn.send({"ok": False, "error_type": exc.__class__.__name__, "error": str(exc)[:500]})
    finally:
        send_conn.close()


def bounded_symbolic_grounding(
    case: dict[str, Any], answer: str, timeout_seconds: float = SYMBOLIC_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run SymPy in a killable subprocess so one hostile formula cannot stall a batch."""
    recv_conn, send_conn = mp.Pipe(duplex=False)
    process = mp.Process(target=_symbolic_worker, args=(send_conn, case, answer), daemon=True)
    process.start()
    send_conn.close()
    payload: dict[str, Any] | None = None
    try:
        if recv_conn.poll(timeout_seconds):
            payload = recv_conn.recv()
    finally:
        recv_conn.close()
    if process.is_alive():
        process.terminate()
    process.join(timeout=2)
    if payload and payload.get("ok"):
        return dict(payload["result"])
    checks = [
        {
            "id": check.get("id"), "status": "CHECK_TIMEOUT" if payload is None else "CHECK_ERROR",
            "target_lhs": check.get("target_lhs"), "expected": str(check.get("rhs")),
        }
        for check in case.get("symbolic_checks", [])
    ]
    row = {
        "extracted": [], "parse_errors": [] if payload is None else [{
            "error": str(payload.get("error") or "symbolic worker failed")[:500],
            "type": str(payload.get("error_type") or "WorkerError"),
        }],
        "checks": checks, "applicable": bool(checks),
        "passed": False if checks else None,
        "checker_error": "timeout" if payload is None else "worker_error",
        "timeout_seconds": timeout_seconds,
    }
    if os.getenv("GARDEN_EVAL_EXECUTE_CODE", "").strip() == "1":
        attach_runtime_code_execution(row)
    return row


def _surface_worker(
    send_conn: Any, surface: str, case: dict[str, Any], answer_question: str,
) -> None:
    started = time.perf_counter()
    try:
        with temporary_store(DB_PATH) as store:
            if surface == "gardener":
                surface_case = {
                    "id": case["id"], "question": answer_question,
                    "category": case["discipline"],
                }
                result = _run_turn(
                    store, surface_case, session_id=f"dual-g-{uuid.uuid4().hex[:8]}",
                )
            else:
                result = explore_inspiration(
                    store, answer_question, session_id=f"dual-i-{uuid.uuid4().hex[:8]}",
                )
        send_conn.send({
            "ok": True, "result": result,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        })
    except BaseException as exc:
        send_conn.send({
            "ok": False, "error_type": exc.__class__.__name__,
            "error": str(exc)[:500],
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        })
    finally:
        send_conn.close()


def bounded_surface_pair(
    case: dict[str, Any], answer_question: str,
    timeout_seconds: float = SURFACE_TIMEOUT_SECONDS,
) -> dict[str, tuple[dict[str, Any], float]]:
    workers: dict[str, dict[str, Any]] = {}
    for surface in ("gardener", "inspiration"):
        recv_conn, send_conn = mp.Pipe(duplex=False)
        process = mp.Process(
            target=_surface_worker,
            args=(send_conn, surface, case, answer_question),
            daemon=True,
        )
        process.start()
        send_conn.close()
        workers[surface] = {"recv": recv_conn, "process": process, "payload": None}
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline and any(item["payload"] is None for item in workers.values()):
        for item in workers.values():
            if item["payload"] is None and item["recv"].poll(0.05):
                try:
                    item["payload"] = item["recv"].recv()
                except EOFError:
                    item["payload"] = {"ok": False, "error_type": "WorkerEOF", "error": "worker closed pipe"}
        time.sleep(0.02)
    results: dict[str, tuple[dict[str, Any], float]] = {}
    for surface, item in workers.items():
        process = item["process"]
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
        item["recv"].close()
        payload = item["payload"]
        if payload and payload.get("ok"):
            results[surface] = (dict(payload["result"]), float(payload["latency_ms"]))
            continue
        timed_out = payload is None
        reason = "outer_timeout" if timed_out else f"{payload.get('error_type')}: {payload.get('error')}"
        results[surface] = ({
            "answer": "该入口未在评测墙钟上限内返回完整回答，已终止本题调用；没有用模板或伪造证据补写。",
            "generation_failed": True,
            "agent_trace": [{
                "node": "evaluation_outer_timeout" if timed_out else "evaluation_worker_failed",
                "summary": "评测外层闸门终止未完成调用",
                "data": {"surface": surface, "reason": reason, "timeout_seconds": timeout_seconds},
            }],
        }, float(payload.get("latency_ms", timeout_seconds * 1000) if payload else timeout_seconds * 1000))
    return results


def run_surfaces(
    case: dict[str, Any], symbolic_dataset: Path = SYMBOLIC_DATASET,
) -> dict[str, Any]:
    case = normalize_capability_case(case)
    case = attach_symbolic_checks(case, symbolic_dataset)
    symbolic_targets = [
        str(check.get("target_lhs") or "").strip()
        for check in case.get("symbolic_checks", [])
        if str(check.get("target_lhs") or "").strip()
    ]
    answer_question = str(case["question"])
    answer_question += (
        "\n\n统一评测要求：请依次给出（1）原理推导与假设；（2）可执行代码或明确工具调用；"
        "（3）结果验证；（4）理论反思与适用边界。若当前回答环境不能实际运行某个工具，"
        "必须明确标注‘未执行’，区分实测结果、预期结果和示例输出，不得伪造运行日志。"
    )
    if symbolic_targets:
        labels = "、".join(f"${target}$" for target in symbolic_targets)
        answer_question += (
            f"\n\n确定性审计格式：请把核心结论另起一行，严格写成以 {labels} 为左侧、"
            "最终简式为右侧的独立公式。该审计行不得含第二个等号/不等号、不得写中间定义或条件特例；"
            "这里只规定公式标签，不暗示结果。"
        )
    scoring_rubric, rubric_hash = freeze_rubric(case)
    run_id = f"dual-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    events: list[dict[str, Any]] = [audit_event(run_id, case["id"], 1, "request_received", {
        "question": case["question"], "discipline": case["discipline"],
        "question_sha256": hashlib.sha256(case["question"].encode("utf-8")).hexdigest(),
        "symbolic_target_lhs": symbolic_targets,
    }), audit_event(run_id, case["id"], 2, "rubric_frozen", {
        "schema_version": RUBRIC_SCHEMA_VERSION,
        "rubric_hash": rubric_hash, "scoring_rubric": scoring_rubric,
    })]
    surface_results = bounded_surface_pair(case, answer_question)
    gardener, gardener_ms = surface_results["gardener"]
    inspiration, inspiration_ms = surface_results["inspiration"]
    gardener_infrastructure = classify_surface_infrastructure(gardener)
    inspiration_infrastructure = classify_surface_infrastructure(inspiration)
    gardener_symbolic = bounded_symbolic_grounding(case, gardener.get("answer", ""))
    inspiration_symbolic = bounded_symbolic_grounding(case, inspiration.get("answer", ""))
    events.extend([
        audit_event(run_id, case["id"], 3, "gardener_answered", {
            "answer": gardener.get("answer", ""), "latency_ms": gardener_ms,
            "generation_failed": bool(gardener.get("generation_failed", False)),
        }),
        audit_event(run_id, case["id"], 4, "inspiration_answered", {
            "answer": inspiration.get("answer", ""), "latency_ms": inspiration_ms,
            "generation_failed": bool(inspiration.get("generation_failed", False)),
        }),
        audit_event(run_id, case["id"], 5, "symbolic_checked", {
            "checker_version": SYMBOLIC_CHECKER_VERSION,
            "gardener": gardener_symbolic, "inspiration": inspiration_symbolic,
        }),
    ])
    return {
        "run_id": run_id,
        "generator_model": llm_config().model,
        "generator_base_host": endpoint_host(llm_config().base_url),
        "id": case["id"], "discipline": case["discipline"], "topic": case["topic"],
        "question": case["question"], "reference": case["reference"], "rubric": case["rubric"],
        "symbolic_checks": list(case.get("symbolic_checks", [])),
        "scoring_rubric": scoring_rubric, "rubric_hash": rubric_hash,
        "reference_risk": case.get("reference_risk", "unknown"),
        "expected_route": case.get("expected_route"),
        "forbidden_routes": list(case.get("forbidden_routes", [])),
        "requires_tool_execution": bool(case.get("requires_tool_execution", False)),
        "gardener": {"answer": gardener.get("answer", ""), "latency_ms": gardener_ms,
                     "citations": gardener.get("citations", []), "evidence_layer": gardener.get("evidence_layer"),
                     "reasoning": gardener.get("reasoning", {}), "agent_trace": gardener.get("agent_trace", []),
                     "repair_degraded": bool(gardener.get("repair_degraded", False)),
                     "repair_diagnostics": gardener.get("repair_diagnostics", {}),
                     "revision_count": int(gardener.get("revision_count", 0)),
                     "generation_failed": bool(gardener.get("generation_failed", False)),
                     "generation_diagnostics": gardener.get("generation_diagnostics", {}),
                     "infrastructure_failure": gardener_infrastructure,
                     "symbolic_grounding": gardener_symbolic,
                     "local_checks": local_surface_checks(gardener.get("answer", ""), surface="gardener", case_id=case["id"])},
        "inspiration": {"answer": inspiration.get("answer", ""), "latency_ms": inspiration_ms,
                       "anchors": inspiration.get("anchors", []), "branches": inspiration.get("branches", []),
                       "claims": inspiration.get("claims", []), "reasoning": inspiration.get("reasoning", {}),
                       "agent_trace": inspiration.get("agent_trace", inspiration.get("trace", [])),
                       "generation_failed": bool(inspiration.get("generation_failed", False)),
                       "generation_diagnostics": inspiration.get("generation_diagnostics", {}),
                       "infrastructure_failure": inspiration_infrastructure,
                       "symbolic_grounding": inspiration_symbolic,
                       "local_checks": {**local_surface_checks(inspiration.get("answer", ""), surface="inspiration", case_id=case["id"]),
                                         "inspiration_has_branches": 2 <= len(inspiration.get("branches", [])) <= 4}},
        "audit_events": events,
    }


async def judge_row(row: dict[str, Any]) -> dict[str, Any]:
    model = judge_model()
    if not judge_api_key(model):
        raise RuntimeError(f"独立裁判 {model} 没有可用凭据")
    prompt = (
        "你是异构红队裁判。你每次只会看到一个匿名回答，不会看到另一个入口，必须独立评分。"
        "先审计 reference 是否有数学、量纲、系数或逻辑错误，再按 scoring_rubric 的 id 逐项给 0/1/2 分，"
        "并引用匿名回答中的短证据。确定性 symbolic_grounding 优先；若认为不适用必须明确说明，不得静默覆盖。"
        "若 requires_tool_execution=true 但 tool_execution_verified=false，模型写出的代码、伪造的控制台输出或口头声称已运行"
        "都不能算作实际执行证据；涉及实际执行与运行结果验证的 rubric 项最高只能给 1 分，并在 failures 中记录。"
        "trusted_tool_execution 来自无网络、无宿主挂载、非 root、只读根目录的 Docker 沙箱；"
        "其非零退出、超时、stderr 及 stdout 与回答声称结果的矛盾，必须优先于文本自述并降低 correctness。"
        "输出严格 JSON：reference_audit{status,issues,corrected_reference},"
        "rubric_results[{rubric_id,score,evidence,reason}],"
        "dimensions{correctness,derivation_rigor,mechanism_discrimination,uncertainty_calibration,naturalness,followup_value}，"
        "其中 dimensions 每项必须使用 1–5 分量尺（1最低、5最高），不得沿用 rubric 的 0–2 分量尺；"
        "verdict(pass/warn/fail),failures。"
    )
    client = AsyncOpenAI(api_key=judge_api_key(model), base_url=judge_base_url(model), timeout=150, max_retries=1)
    async def close_client() -> None:
        try:
            await asyncio.wait_for(client.close(), timeout=10)
        except BaseException:
            pass
    verify_frozen_rubric(row)
    refresh_row_infrastructure(row)
    symbolic_case = attach_symbolic_checks(row)
    for surface in ("gardener", "inspiration"):
        existing_execution = row[surface].get("local_checks", {}).get("tool_execution")
        refreshed_checks = local_surface_checks(
            row[surface].get("answer", ""), surface=surface, case_id=str(row.get("id") or ""),
        )
        refreshed_checks.pop("inspiration_has_branches", None)
        row[surface].setdefault("local_checks", {}).update(refreshed_checks)
        if existing_execution:
            row[surface]["local_checks"]["tool_execution"] = existing_execution
            row[surface]["local_checks"]["tool_execution_verified"] = (
                existing_execution.get("status") == "passed"
            )
        row[surface]["symbolic_grounding"] = bounded_symbolic_grounding(
            symbolic_case, row[surface].get("answer", ""),
        )
    append_audit_event(row, "symbolic_rechecked_before_judge", {
        "checker_version": SYMBOLIC_CHECKER_VERSION,
        "gardener": row["gardener"]["symbolic_grounding"],
        "inspiration": row["inspiration"]["symbolic_grounding"],
    })
    started = time.perf_counter()
    rubric_ids = [item["id"] for item in row["scoring_rubric"]]

    def validate_blind_result(result: dict[str, Any]) -> dict[str, Any]:
        items = result.get("rubric_results")
        if not isinstance(items, list):
            raise ValueError("rubric_results 缺失")
        ids = [str(item.get("rubric_id") or "") for item in items]
        if ids != rubric_ids or len(set(ids)) != len(ids):
            raise ValueError(f"rubric id 不完整或乱序：{ids}")
        for item in items:
            score = item.get("score")
            if not isinstance(score, (int, float)) or float(score) not in {0.0, 1.0, 2.0}:
                raise ValueError(f"非法 rubric 分值：{score}")
            if not str(item.get("reason") or "").strip():
                raise ValueError("rubric reason 为空")
        dimensions = result.get("dimensions")
        if not isinstance(dimensions, dict) or any(
            not 1 <= float(dimensions.get(key, 0)) <= 5
            for key in ("correctness", "derivation_rigor", "mechanism_discrimination", "uncertainty_calibration", "naturalness", "followup_value")
        ):
            raise ValueError("dimensions 缺失或越界")
        return result

    async def judge_surface(surface: str) -> dict[str, Any]:
        payload = build_blind_judge_payload(row, surface)
        payload_hash = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        for attempt, max_tokens in enumerate((3600, 6000), 1):
            append_audit_event(row, "judge_requested", {
                "surface": surface, "attempt": attempt, "model": model,
                "base_host": endpoint_host(judge_base_url(model)),
                "prompt_version": JUDGE_PROMPT_VERSION, "payload_sha256": payload_hash,
                "payload_fields": sorted(payload),
                "judge_visible_surface_identity": False,
                "peer_answer_included": False,
                "max_tokens": max_tokens,
            })
            try:
                response = await client.chat.completions.create(
                    model=model, response_format={"type": "json_object"}, max_tokens=max_tokens,
                    messages=[{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                    **judge_request_options(model),
                )
                content = str(response.choices[0].message.content or "").strip()
                parsed = validate_blind_result(json.loads(content)) if content else None
                append_audit_event(row, "judge_response_received", {
                    "surface": surface, "attempt": attempt, "nonempty": bool(content),
                    "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest() if content else "",
                    "schema_valid": parsed is not None,
                })
                if parsed is not None:
                    return parsed
            except Exception as exc:
                append_audit_event(row, "judge_attempt_failed", {
                    "surface": surface, "attempt": attempt,
                    "error": f"{exc.__class__.__name__}: {exc}"[:400],
                })
        raise ValueError(f"{surface}: 独立裁判 {model} 两次均未返回严格盲审结果")

    try:
        gardener_judge, inspiration_judge = await asyncio.wait_for(asyncio.gather(
            judge_surface("gardener"), judge_surface("inspiration"),
        ), timeout=JUDGE_BATCH_TIMEOUT_SECONDS)
    except BaseException:
        await close_client()
        raise
    by_surface = {"gardener": gardener_judge, "inspiration": inspiration_judge}
    merged = []
    totals: dict[str, float] = {}
    for surface, judged in by_surface.items():
        score_map = {item["rubric_id"]: item for item in judged["rubric_results"]}
        totals[surface] = round(sum(
            float(score_map[item["id"]]["score"]) * float(item["weight"])
            for item in row["scoring_rubric"]
        ) / 2 * 100, 2)
    for rubric in row["scoring_rubric"]:
        gid = {item["rubric_id"]: item for item in gardener_judge["rubric_results"]}[rubric["id"]]
        iid = {item["rubric_id"]: item for item in inspiration_judge["rubric_results"]}[rubric["id"]]
        merged.append({
            "rubric_id": rubric["id"], "criterion": rubric["criterion"],
            "gardener_score": gid["score"], "gardener_evidence": gid.get("evidence", ""),
            "inspiration_score": iid["score"], "inspiration_evidence": iid.get("evidence", ""),
        })
    result = {
        "model": model, "model_label": judge_label(model),
        "base_host": endpoint_host(judge_base_url(model)),
        "independence": judge_independence(model),
        "role": "independent_blind_auxiliary_judge",
        "prompt_version": JUDGE_PROMPT_VERSION,
        "rubric_results": merged,
        "gardener_dimensions": gardener_judge["dimensions"],
        "inspiration_dimensions": inspiration_judge["dimensions"],
        "gardener_verdict": gardener_judge.get("verdict"),
        "inspiration_verdict": inspiration_judge.get("verdict"),
        "gardener_score": totals["gardener"], "inspiration_score": totals["inspiration"],
        "reference_audits": {"gardener_blind": gardener_judge.get("reference_audit"), "inspiration_blind": inspiration_judge.get("reference_audit")},
        "failures": {"gardener": gardener_judge.get("failures", []), "inspiration": inspiration_judge.get("failures", [])},
        "comparison": f"程序聚合分：问园丁 {totals['gardener']}，灵感检测 {totals['inspiration']}；两份答案先独立盲评，未在裁判 Prompt 中互见。",
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    result = enforce_hard_failure_gates(row, result)
    await close_client()
    return result


def route_matches_expectation(row: dict[str, Any]) -> bool | None:
    expected = row.get("expected_route")
    if expected is None:
        return None
    expected_routes = {str(item) for item in (expected if isinstance(expected, list) else [expected])}
    trace = row.get("gardener", {}).get("agent_trace", [])
    nodes = {str(event.get("node") or "") for event in trace}
    forbidden = {str(item) for item in row.get("forbidden_routes", [])}
    if nodes & forbidden:
        return False
    if "NO_RETRIEVAL" in expected_routes:
        return not bool(nodes & {"plan_sources", "retrieve_sources", "audit_sources"})
    return bool(nodes & expected_routes)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    routing_results = [route_matches_expectation(row) for row in rows]
    routing_evaluated = sum(result is not None for result in routing_results)
    routing_correct = sum(result is True for result in routing_results)
    summary: dict[str, Any] = {
        "cases": len(rows),
        "judged": sum(bool(r.get("auxiliary_judge", {}).get("rubric_results")) for r in rows),
        "rubric_non_null": sum(bool(r.get("scoring_rubric")) for r in rows),
        "judge_models": sorted({
            str(r.get("auxiliary_judge", {}).get("model")) for r in rows
            if r.get("auxiliary_judge", {}).get("model")
        }),
        "judge_base_hosts": sorted({
            str(r.get("auxiliary_judge", {}).get("base_host")) for r in rows
            if r.get("auxiliary_judge", {}).get("base_host")
        }),
        "judge_independence": sorted({
            str(r.get("auxiliary_judge", {}).get("independence")) for r in rows
            if r.get("auxiliary_judge", {}).get("independence")
        }),
        "generator_models": sorted({
            str(r.get("generator_model")) for r in rows if r.get("generator_model")
        }),
        "generator_base_hosts": sorted({
            str(r.get("generator_base_host")) for r in rows if r.get("generator_base_host")
        }),
        "tool_execution_required": sum(bool(r.get("requires_tool_execution")) for r in rows),
        "routing_evaluated": routing_evaluated,
        "routing_correct": routing_correct,
        "routing_accuracy": round(routing_correct / routing_evaluated, 4) if routing_evaluated else None,
        "gardener_false_refusals": sum(
            not r.get("gardener", {}).get("local_checks", {}).get("not_defensive_refusal", False)
            for r in rows
        ),
        "gardener_generation_failures": sum(
            bool(r.get("gardener", {}).get("generation_failed")) for r in rows
        ),
        "gardener_infrastructure_failures": sum(
            bool(r.get("gardener", {}).get("infrastructure_failure", {}).get("detected")) for r in rows
        ),
        "gardener_capability_generation_failures": sum(
            bool(r.get("gardener", {}).get("generation_failed"))
            and not bool(r.get("gardener", {}).get("infrastructure_failure", {}).get("detected"))
            for r in rows
        ),
        "inspiration_false_refusals": sum(
            not r.get("inspiration", {}).get("local_checks", {}).get("not_defensive_refusal", False)
            for r in rows
        ),
        "inspiration_generation_failures": sum(
            bool(r.get("inspiration", {}).get("generation_failed")) for r in rows
        ),
        "inspiration_infrastructure_failures": sum(
            bool(r.get("inspiration", {}).get("infrastructure_failure", {}).get("detected")) for r in rows
        ),
        "inspiration_capability_generation_failures": sum(
            bool(r.get("inspiration", {}).get("generation_failed"))
            and not bool(r.get("inspiration", {}).get("infrastructure_failure", {}).get("detected"))
            for r in rows
        ),
        "gardener_repair_trigger_rate": round(
            sum(int(r.get("gardener", {}).get("revision_count", 0)) > 0 for r in rows) / max(1, len(rows)),
            4,
        ),
    }
    for surface in ("gardener", "inspiration"):
        scorable_rows = [
            r for r in rows
            if not bool(r.get(surface, {}).get("infrastructure_failure", {}).get("detected"))
        ]
        latencies = [float(r[surface]["latency_ms"]) for r in rows]
        hard_gated_scores = [
            float(r.get("auxiliary_judge", {}).get(f"{surface}_score"))
            for r in scorable_rows if r.get("auxiliary_judge", {}).get(f"{surface}_score") is not None
        ]
        judged_dimensions = [
            r.get("auxiliary_judge", {}).get(f"{surface}_dimensions", {}) for r in scorable_rows
        ]
        keys = sorted({k for d in judged_dimensions if isinstance(d, dict) for k in d})
        dimension_values: dict[str, list[float]] = {key: [] for key in keys}
        success_values: dict[str, list[float]] = {key: [] for key in keys}
        for row, dimensions in zip(scorable_rows, judged_dimensions):
            failed = bool(row.get(surface, {}).get("generation_failed"))
            for key in keys:
                value = dimensions.get(key) if isinstance(dimensions, dict) else None
                if failed:
                    dimension_values[key].append(0.0)
                elif value is not None:
                    dimension_values[key].append(float(value))
                    success_values[key].append(float(value))
        summary[surface] = {
            "average_latency_seconds": round(statistics.fmean(latencies) / 1000, 2) if latencies else None,
            "capability_scorable": len(scorable_rows),
            "infrastructure_excluded": len(rows) - len(scorable_rows),
            "average_hard_gated_score": round(statistics.fmean(hard_gated_scores), 2) if hard_gated_scores else None,
            "hard_gated_score_denominator": len(hard_gated_scores),
            "symbolic_applicable": sum(bool(r[surface].get("symbolic_grounding", {}).get("applicable")) for r in scorable_rows),
            "symbolic_pass": sum(r[surface].get("symbolic_grounding", {}).get("passed") is True for r in scorable_rows),
            "tool_execution_verified": sum(
                bool(r[surface].get("local_checks", {}).get("tool_execution_verified"))
                for r in scorable_rows if r.get("requires_tool_execution")
            ),
            "deterministic_oracle_failed": sum(
                r[surface].get("local_checks", {}).get("deterministic_oracle", {}).get("passed") is False
                for r in scorable_rows
            ),
            "dimension_averages": {
                key: round(statistics.fmean(values), 2)
                for key, values in dimension_values.items() if values
            },
            "dimension_success_only_averages": {
                key: round(statistics.fmean(values), 2)
                for key, values in success_values.items() if values
            },
            "dimension_denominators": {key: len(values) for key, values in dimension_values.items()},
        }
    return summary


def write_report(rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = REPORT_DIR / f"dual-surface-capability-{stamp}.json"
    md_path = REPORT_DIR / f"dual-surface-capability-{stamp}.md"
    summary = summarize(rows)
    json_path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 问园丁 × 灵感检测：科学能力报告", "", f"- 样本：{len(rows)}；独立裁判完成：{summary['judged']}/{len(rows)}",
             f"- 实际被测模型：{', '.join(summary['generator_models']) or '未记录'}",
             f"- 被测端点：{', '.join(summary['generator_base_hosts']) or '未记录'}",
             f"- 实际裁判模型：{', '.join(summary['judge_models']) or '无'}",
             f"- 裁判端点：{', '.join(summary['judge_base_hosts']) or '未记录'}",
             f"- 裁判独立性：{', '.join(summary['judge_independence']) or '未完成裁判'}",
             f"- 基础设施故障（排除出能力分母）：问园丁 {summary['gardener_infrastructure_failures']}；灵感检测 {summary['inspiration_infrastructure_failures']}",
             f"- 能力可评分样本：问园丁 {summary['gardener']['capability_scorable']}/{len(rows)}；灵感检测 {summary['inspiration']['capability_scorable']}/{len(rows)}",
             f"- 要求工具执行：{summary['tool_execution_required']} 题；问园丁已验证：{summary['gardener']['tool_execution_verified']}；灵感检测已验证：{summary['inspiration']['tool_execution_verified']}",
             f"- 确定性符号检查覆盖：问园丁 {summary['gardener']['symbolic_applicable']}/{len(rows)}；灵感检测 {summary['inspiration']['symbolic_applicable']}/{len(rows)}",
             f"- 确定性答案 oracle 失败：问园丁 {summary['gardener']['deterministic_oracle_failed']}；灵感检测 {summary['inspiration']['deterministic_oracle_failed']}",
             f"- 程序硬门控总分均值：问园丁 {summary['gardener']['average_hard_gated_score']}；灵感检测 {summary['inspiration']['average_hard_gated_score']}",
             f"- 问园丁平均耗时：{summary['gardener']['average_latency_seconds']}秒", f"- 灵感检测平均耗时：{summary['inspiration']['average_latency_seconds']}秒", "",
             "## 全样本惩罚均分（成功答案按 1–5；生成失败按 0）", "", "```json", json.dumps({"问园丁": summary["gardener"]["dimension_averages"], "灵感检测": summary["inspiration"]["dimension_averages"]}, ensure_ascii=False, indent=2), "```", "",
             "## 成功样本维度均分（1–5，不含生成失败）", "", "```json", json.dumps({"问园丁": summary["gardener"]["dimension_success_only_averages"], "灵感检测": summary["inspiration"]["dimension_success_only_averages"]}, ensure_ascii=False, indent=2), "```", ""]
    for row in rows:
        lines += [f"## {row['id']} · {row['discipline']} · {row['topic']}", "", f"- Reference 风险：{row['reference_risk']}",
                  f"- Rubric hash：`{row.get('rubric_hash', '')}`",
                  f"- 问园丁耗时：{row['gardener']['latency_ms']/1000:.2f}s；灵感检测耗时：{row['inspiration']['latency_ms']/1000:.2f}s", "",
                  "### 确定性符号校验", "", "```json", json.dumps({"问园丁": row['gardener'].get('symbolic_grounding', {}), "灵感检测": row['inspiration'].get('symbolic_grounding', {})}, ensure_ascii=False, indent=2), "```", "",
                  "### 问园丁回答", "", row['gardener']['answer'], "", "### 灵感检测回答", "", row['inspiration']['answer'], "",
                  "### DeepSeek-V4 辅裁", "", "```json", json.dumps(row.get("auxiliary_judge", row.get("judge_error", {})), ensure_ascii=False, indent=2), "```", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def write_checkpoint(
    path: Path, rows: list[dict[str, Any]], *, complete: bool = False,
    interruption: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "complete": complete, "updated_at": datetime.now().astimezone().isoformat(),
        "summary": summarize(rows), "rows": rows,
    }
    if interruption:
        payload["interruption"] = interruption
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def append_judge_event(row: dict[str, Any]) -> None:
    judge = row.get("auxiliary_judge")
    error = row.get("judge_error")
    append_audit_event(row, "judge_completed" if judge else "judge_failed", judge if judge else error or {})
    if judge:
        append_audit_event(row, "judge_disagreement_recorded", {
            "gardener_verdict": judge.get("gardener_verdict"),
            "inspiration_verdict": judge.get("inspiration_verdict"),
            "comparison": judge.get("comparison"),
        })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--ids", default="")
    parser.add_argument("--input-report", type=Path)
    parser.add_argument("--reuse-existing-judge", action="store_true")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--symbolic-dataset", type=Path, default=SYMBOLIC_DATASET)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--rejudge-completed", action="store_true")
    parser.add_argument("--execute-code", action="store_true")
    parser.add_argument("--repair-code", action="store_true")
    args = parser.parse_args()
    if args.repair_code and not args.execute_code:
        parser.error("--repair-code requires --execute-code")
    os.environ["GARDEN_DISABLE_NETWORK"] = "1"
    if args.execute_code:
        os.environ["GARDEN_EVAL_EXECUTE_CODE"] = "1"
    runtime = assert_expected_runtime()
    print(f"runtime={json.dumps(runtime, ensure_ascii=False)}", flush=True)
    cases = [normalize_capability_case(case) for case in load_cases(args.dataset)]
    selected = {item.strip() for item in args.ids.split(",") if item.strip()}
    if selected: cases = [case for case in cases if case["id"] in selected]
    if args.limit: cases = cases[:args.limit]
    if args.input_report:
        payload = json.loads(args.input_report.read_text(encoding="utf-8"))
        rows = list(payload.get("rows", []))
        if selected:
            rows = [row for row in rows if str(row.get("id")) in selected]
        if args.limit:
            rows = rows[:args.limit]
        for index, row in enumerate(rows, 1):
            print(f"[rejudge {index}/{len(rows)}] {row.get('id')}", flush=True)
            refresh_row_infrastructure(row)
            if args.execute_code:
                attach_runtime_code_execution(row)
            if args.repair_code:
                repair_failed_runtime_answers(row)
            for surface in ("gardener", "inspiration"):
                existing_execution = row[surface].get("local_checks", {}).get("tool_execution")
                refreshed_checks = local_surface_checks(
                    row[surface].get("answer", ""), surface=surface,
                    case_id=str(row.get("id") or ""),
                )
                refreshed_checks.pop("inspiration_has_branches", None)
                row[surface].setdefault("local_checks", {}).update(refreshed_checks)
                if existing_execution:
                    row[surface]["local_checks"]["tool_execution"] = existing_execution
                    row[surface]["local_checks"]["tool_execution_verified"] = (
                        existing_execution.get("status") == "passed"
                    )
            if args.reuse_existing_judge:
                existing = row.get("auxiliary_judge")
                if not isinstance(existing, dict) or not existing.get("rubric_results"):
                    raise RuntimeError(f"{row.get('id')}: 没有可复用的既有裁判结果")
                row["auxiliary_judge"] = enforce_hard_failure_gates(row, existing)
                append_audit_event(row, "hard_gates_reapplied_without_external_call", {
                    "oracle_version": DETERMINISTIC_ORACLE_VERSION,
                    "external_request_made": False,
                })
            elif args.skip_judge:
                symbolic_case = attach_symbolic_checks(row, args.symbolic_dataset)
                for surface in ("gardener", "inspiration"):
                    row[surface]["symbolic_grounding"] = bounded_symbolic_grounding(
                        symbolic_case, row[surface].get("answer", ""),
                    )
                append_audit_event(row, "symbolic_rechecked_without_judge", {
                    "checker_version": SYMBOLIC_CHECKER_VERSION,
                    "gardener": row["gardener"]["symbolic_grounding"],
                    "inspiration": row["inspiration"]["symbolic_grounding"],
                })
            else:
                try: row["auxiliary_judge"] = asyncio.run(judge_row(row)); row.pop("judge_error", None)
                except Exception as exc: row["judge_error"] = {"type": exc.__class__.__name__, "message": str(exc)}
                append_judge_event(row)
        paths = write_report(rows)
        summary = summarize(rows)
        complete = bool(args.skip_judge or summary["judged"] == len(rows))
        print(json.dumps({"complete": complete, "summary": summary, "reports": [str(p) for p in paths]}, ensure_ascii=False, indent=2))
        if not complete:
            raise SystemExit(f"重裁不完整：仅 {summary['judged']}/{len(rows)} 题完成独立裁判")
        return
    if args.resume_checkpoint:
        checkpoint_path = args.resume_checkpoint
        checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        rows = list(checkpoint_payload.get("rows", []))
        if not args.skip_judge:
            for index, row in enumerate(rows, 1):
                if (
                    row.get("auxiliary_judge", {}).get("rubric_results")
                    and not args.rejudge_completed
                ):
                    continue
                print(f"[resume rejudge {index}/{len(rows)}] {row.get('id')}", flush=True)
                try:
                    row["auxiliary_judge"] = asyncio.run(judge_row(row))
                    row.pop("judge_error", None)
                except Exception as exc:
                    row["judge_error"] = {"type": exc.__class__.__name__, "message": str(exc)}
                append_judge_event(row)
            write_checkpoint(checkpoint_path, rows)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_path = REPORT_DIR / f"dual-surface-checkpoint-{stamp}.json"
        rows = []
    completed_ids = {str(row.get("id")) for row in rows}
    remaining = [case for case in cases if str(case.get("id")) not in completed_ids]
    print(f"checkpoint={checkpoint_path}", flush=True)
    for i, case in enumerate(remaining, 1):
        print(f"[{i}/{len(remaining)}] running {case['id']} on both surfaces", flush=True)
        row = None
        for infrastructure_attempt in range(INFRASTRUCTURE_RETRIES + 1):
            candidate = run_surfaces(case, args.symbolic_dataset)
            failures = {
                surface: candidate.get(surface, {}).get("infrastructure_failure", {})
                for surface in ("gardener", "inspiration")
            }
            fatal = {surface: detail for surface, detail in failures.items() if detail.get("fatal")}
            if fatal:
                interruption = {
                    "case_id": case["id"], "kind": "fatal_infrastructure_failure",
                    "attempt": infrastructure_attempt + 1, "surfaces": fatal,
                    "message": "检测到鉴权、额度或计费故障；已停止评测，修复凭据后可从本 checkpoint 续跑。",
                }
                write_checkpoint(checkpoint_path, rows, interruption=interruption)
                raise SystemExit(
                    f"{case['id']}: 检测到致命基础设施故障，已在污染后续题目前停止："
                    + json.dumps(fatal, ensure_ascii=False)
                )
            transient = {surface: detail for surface, detail in failures.items() if detail.get("detected")}
            if transient and infrastructure_attempt < INFRASTRUCTURE_RETRIES:
                print(
                    f"[infra retry {infrastructure_attempt + 1}/{INFRASTRUCTURE_RETRIES}] "
                    f"{case['id']} {json.dumps(transient, ensure_ascii=False)}",
                    flush=True,
                )
                time.sleep(INFRASTRUCTURE_RETRY_BACKOFF_SECONDS * (infrastructure_attempt + 1))
                continue
            row = candidate
            break
        assert row is not None
        if args.execute_code:
            attach_runtime_code_execution(row)
        if args.repair_code:
            repair_failed_runtime_answers(row)
        if not args.skip_judge:
            try: row["auxiliary_judge"] = asyncio.run(judge_row(row))
            except Exception as exc: row["judge_error"] = {"type": exc.__class__.__name__, "message": str(exc)}
            append_judge_event(row)
        rows.append(row)
        write_checkpoint(checkpoint_path, rows)
    summary = summarize(rows)
    complete = bool(args.skip_judge or summary["judged"] == len(rows))
    paths = write_report(rows)
    write_checkpoint(checkpoint_path, rows, complete=complete)
    print(json.dumps({"complete": complete, "summary": summary, "reports": [str(p) for p in paths]}, ensure_ascii=False, indent=2))
    if not complete:
        raise SystemExit(f"评测不完整：仅 {summary['judged']}/{len(rows)} 题完成独立裁判")


if __name__ == "__main__":
    main()
