from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from core.llm import chat
from evals.science_code_execution import execute_answer_in_docker


RUNTIME_REPAIR_VERSION = "science-runtime-repair-v2"
REPAIRABLE_REASONS = {"nonzero_exit"}


def runtime_failure_is_repairable(execution: dict[str, Any] | None) -> bool:
    """Only repair concrete code failures; never rewrite missing-code or infra cases."""

    execution = execution or {}
    return bool(
        execution.get("executed")
        and execution.get("status") == "failed"
        and execution.get("reason") in REPAIRABLE_REASONS
        and not execution.get("timed_out")
        and not execution.get("output_limited")
    )


def build_runtime_repair_request(
    *, question: str, answer: str, execution: dict[str, Any],
) -> tuple[str, str]:
    system = (
        "你是科学计算回答的 Runtime Repair Agent。你收到用户原题、完整首稿和可信 Docker "
        "运行错误。请输出修正后的完整最终回答，不是补丁、审校意见或道歉。保留首稿中正确的"
        "原理推导、验证设计、理论反思与边界，只修改导致执行失败的代码和受其影响的结果表述。"
        "不得声称未实际得到的运行结果；修复后的代码必须自包含，禁止联网、读写宿主文件、"
        "启动子进程或要求用户另行安装依赖。只输出完整回答正文。"
    )
    trusted_trace = {
        "image": execution.get("image"),
        "reason": execution.get("reason"),
        "exit_code": execution.get("exit_code"),
        "stdout": str(execution.get("stdout") or "")[-4000:],
        "stderr": str(execution.get("stderr") or "")[-6000:],
    }
    user = (
        f"【用户原题】\n{question}\n\n"
        f"【完整首稿】\n{answer}\n\n"
        "【可信隔离运行记录】\n"
        + json.dumps(trusted_trace, ensure_ascii=False, indent=2)
    )
    return system, user


def repair_answer_once(
    *,
    question: str,
    answer: str,
    execution: dict[str, Any],
    timeout_seconds: float = 60.0,
    chat_fn: Callable[..., str | None] = chat,
    execute_fn: Callable[..., dict[str, Any]] = execute_answer_in_docker,
) -> dict[str, Any]:
    """Ask the generator for one full-answer repair and verify it in Docker."""

    result: dict[str, Any] = {
        "version": RUNTIME_REPAIR_VERSION,
        "eligible": runtime_failure_is_repairable(execution),
        "attempted": False,
        "accepted": False,
        "initial_answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        "initial_execution": execution,
    }
    if not result["eligible"]:
        result["reason"] = "runtime_failure_not_repairable"
        return result

    result["attempted"] = True
    system, user = build_runtime_repair_request(
        question=question, answer=answer, execution=execution,
    )
    try:
        candidate = str(chat_fn(
            system, user, temperature=0.1, json_mode=False,
            timeout=120, max_retries=1,
        ) or "").strip()
    except Exception as exc:
        result["reason"] = "repair_model_failed"
        result["error"] = f"{exc.__class__.__name__}: {exc}"[:1000]
        return result
    if not candidate:
        result["reason"] = "repair_model_empty"
        return result

    candidate_execution = execute_fn(candidate, timeout_seconds=timeout_seconds)
    result.update({
        "candidate_answer": candidate,
        "candidate_answer_sha256": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
        "candidate_execution": candidate_execution,
    })
    if candidate_execution.get("status") != "passed":
        result["reason"] = "candidate_execution_failed"
        return result
    result["accepted"] = True
    result["reason"] = "candidate_execution_passed"
    return result


def repair_answer_with_retries(
    *,
    question: str,
    answer: str,
    execution: dict[str, Any],
    timeout_seconds: float = 60.0,
    max_attempts: int = 2,
    chat_fn: Callable[..., str | None] = chat,
    execute_fn: Callable[..., dict[str, Any]] = execute_answer_in_docker,
) -> dict[str, Any]:
    """Run a bounded traceback-driven loop, retaining every candidate as evidence."""

    max_attempts = max(1, min(3, int(max_attempts)))
    overall: dict[str, Any] = {
        "version": RUNTIME_REPAIR_VERSION,
        "eligible": runtime_failure_is_repairable(execution),
        "attempted": False,
        "accepted": False,
        "max_attempts": max_attempts,
        "attempts": [],
        "initial_answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        "initial_execution": execution,
    }
    if not overall["eligible"]:
        overall["reason"] = "runtime_failure_not_repairable"
        return overall

    current_answer = answer
    current_execution = execution
    for attempt_index in range(1, max_attempts + 1):
        attempt = repair_answer_once(
            question=question,
            answer=current_answer,
            execution=current_execution,
            timeout_seconds=timeout_seconds,
            chat_fn=chat_fn,
            execute_fn=execute_fn,
        )
        attempt["attempt"] = attempt_index
        overall["attempts"].append(attempt)
        overall["attempted"] = True
        for key in (
            "candidate_answer", "candidate_answer_sha256", "candidate_execution",
            "error", "reason",
        ):
            if key in attempt:
                overall[key] = attempt[key]
        if attempt.get("accepted"):
            overall["accepted"] = True
            overall["reason"] = "candidate_execution_passed"
            return overall
        if attempt.get("reason") != "candidate_execution_failed":
            return overall
        next_answer = str(attempt.get("candidate_answer") or "").strip()
        next_execution = attempt.get("candidate_execution") or {}
        if not next_answer or not runtime_failure_is_repairable(next_execution):
            return overall
        current_answer = next_answer
        current_execution = next_execution
    overall["reason"] = "repair_attempts_exhausted"
    return overall
