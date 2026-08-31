from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from evals.dual_surface_capability_eval import classify_surface_infrastructure


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / "evals" / "training"


def _execution_failure_class(trace: dict[str, Any]) -> str | None:
    status = str(trace.get("status") or "")
    reason = str(trace.get("reason") or "")
    stderr = str(trace.get("stderr") or "")
    if status == "passed":
        return None
    if status == "no_python_block":
        return "missing_code"
    if status == "blocked":
        return "static_safety_rejection"
    if reason == "timeout":
        return "timeout_or_excessive_compute"
    if reason == "output_limit":
        return "excessive_output"
    if "Unable to find image" in stderr or "pull access denied" in stderr:
        return "runner_image_unavailable"
    if "ModuleNotFoundError" in stderr:
        return "dependency_missing"
    if "FileNotFoundError" in stderr:
        return "external_input_missing"
    if status == "failed":
        return "generated_code_error"
    return "execution_not_verified"


def _surface_experience(row: dict[str, Any], surface: str) -> dict[str, Any]:
    result = dict(row.get(surface) or {})
    judge = dict(row.get("auxiliary_judge") or {})
    infrastructure = result.get("infrastructure_failure") or classify_surface_infrastructure(result)
    generation_failed = bool(result.get("generation_failed"))
    local_checks = dict(result.get("local_checks") or {})
    oracle = dict(local_checks.get("deterministic_oracle") or {})
    tool_required = bool(row.get("requires_tool_execution"))
    tool_verified = bool(local_checks.get("tool_execution_verified"))
    tool_trace = dict(local_checks.get("tool_execution") or {})
    execution_failure_class = _execution_failure_class(tool_trace)
    verdict = str(judge.get(f"{surface}_verdict") or "")
    score = judge.get(f"{surface}_score")

    labels: list[str] = []
    disposition = "candidate_requires_verification"
    trainable = False
    if infrastructure.get("detected"):
        labels.append(f"infrastructure:{infrastructure.get('category') or 'unknown'}")
        disposition = "exclude_infrastructure"
    elif generation_failed:
        labels.append("generation_failure")
        disposition = "exclude_generation_failure"
    else:
        if tool_required and not tool_verified:
            labels.append("tool_execution_not_verified")
            if execution_failure_class:
                labels.append(f"execution:{execution_failure_class}")
            if execution_failure_class == "missing_code":
                disposition = "execution_missing_code"
            elif execution_failure_class == "static_safety_rejection":
                disposition = "execution_blocked_static"
            elif execution_failure_class in {"timeout_or_excessive_compute", "excessive_output"}:
                disposition = "execution_resource_failure"
            elif execution_failure_class in {
                "generated_code_error", "dependency_missing", "external_input_missing",
                "runner_image_unavailable",
            }:
                disposition = "execution_failed"
            else:
                disposition = "execution_required"
        if oracle.get("passed") is False:
            labels.extend(f"oracle:{issue}" for issue in oracle.get("issues", []))
            disposition = "repair_required_verified"
        elif verdict in {"fail", "warn"} and disposition == "candidate_requires_verification":
            labels.append(f"judge_verdict:{verdict}")
            disposition = "semantic_review_required"

    rubric_results = []
    for item in judge.get("rubric_results") or []:
        rubric_results.append({
            "rubric_id": item.get("rubric_id"),
            "criterion": item.get("criterion"),
            "score": item.get(f"{surface}_score"),
            "evidence": item.get(f"{surface}_evidence"),
        })
    return {
        "surface": surface,
        "answer": str(result.get("answer") or ""),
        "generation_failed": generation_failed,
        "infrastructure": infrastructure,
        "tool_execution": {
            "required": tool_required, "verified": tool_verified,
            "failure_class": execution_failure_class,
            "trace": tool_trace,
        },
        "deterministic_oracle": oracle,
        "judge": {
            "model": judge.get("model"), "base_host": judge.get("base_host"),
            "verdict": verdict, "score": score,
            "dimensions": judge.get(f"{surface}_dimensions", {}),
            "rubric_results": rubric_results,
            "failures": (judge.get("failures") or {}).get(surface, []),
        },
        "failure_labels": labels,
        "disposition": disposition,
        # No sample becomes SFT/DPO material until a corrected answer and real
        # tool evidence have been independently verified.
        "trainable": trainable,
        "required_next_action": {
            "exclude_infrastructure": "修复凭据后重新生成，不把当前回退文本用于训练。",
            "exclude_generation_failure": "重新生成并查明解析/格式失败原因。",
            "execution_required": "在隔离工具环境中运行代码，保存退出码、stdout、依赖和产物哈希。",
            "execution_missing_code": "补充可执行代码或明确工具调用，再在隔离环境中验证。",
            "execution_blocked_static": "移除文件、网络、进程或动态执行风险后重新生成并验证。",
            "execution_resource_failure": "缩小样例规模或优化算法，使其在受限沙箱内完成。",
            "execution_failed": "依据 stderr 与输入/依赖诊断修复代码，再重跑并独立复核。",
            "repair_required_verified": "依据确定性 oracle 修正答案和代码，再独立复核。",
            "semantic_review_required": "由独立专家或可验证 oracle 审核后生成修正版。",
            "candidate_requires_verification": "完成事实与工具验证后才可晋升为正样本。",
        }[disposition],
    }


def build_experience(report: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in report.get("rows") or []:
        for surface in ("gardener", "inspiration"):
            records.append({
                "schema_version": "science-training-experience-v2",
                "case_id": row.get("id"), "discipline": row.get("discipline"),
                "topic": row.get("topic"), "question": row.get("question"),
                "reference": row.get("reference"), "scoring_rubric": row.get("scoring_rubric"),
                "rubric_hash": row.get("rubric_hash"),
                "generator": {
                    "model": row.get("generator_model"), "base_host": row.get("generator_base_host"),
                },
                **_surface_experience(row, surface),
            })
    dispositions = Counter(str(item["disposition"]) for item in records)
    labels = Counter(label for item in records for label in item["failure_labels"])
    summary = {
        "schema_version": "science-training-experience-summary-v2",
        "created_at": datetime.now().astimezone().isoformat(),
        "cases": len({str(item["case_id"]) for item in records}),
        "surface_records": len(records),
        "trainable_now": sum(bool(item["trainable"]) for item in records),
        "dispositions": dict(sorted(dispositions.items())),
        "failure_labels": dict(sorted(labels.items())),
        "policy": "所有记录先用于错误挖掘与回归；只有经修正、工具执行和独立验证后才能进入 SFT/DPO。",
    }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    records, summary = build_experience(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    jsonl_path = args.output_dir / f"science100-experience-{stamp}.jsonl"
    summary_path = args.output_dir / f"science100-experience-{stamp}.summary.json"
    jsonl_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "records": len(records), "jsonl": str(jsonl_path),
        "summary": str(summary_path), "counts": summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
