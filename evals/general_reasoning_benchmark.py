from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from core.config import DB_PATH
from core.reasoning_capability import (
    CATEGORY_ALIASES,
    classify_reasoning_task,
    review_reasoning_answer,
)
from evals.adapter import load_cases, run_graph_case, temporary_store


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "evals" / "datasets" / "general_reasoning_15_v1.jsonl"
DEFAULT_REPORT_DIR = ROOT / "evals" / "reports"
REQUIRED_FIELDS = {
    "id", "category", "subcategory", "difficulty", "question", "reference",
    "assumptions", "reasoning_trace", "key_insights", "common_errors", "evaluation_points",
}


def validate_case(case: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(REQUIRED_FIELDS - set(case))
    points = case.get("evaluation_points") if isinstance(case.get("evaluation_points"), list) else []
    weights = [float(item.get("weight") or 0.0) for item in points if isinstance(item, dict)]
    criteria = [str(item.get("criterion") or "").strip() for item in points if isinstance(item, dict)]
    expected_key = CATEGORY_ALIASES.get(str(case.get("category") or "").strip(), "")
    routed = classify_reasoning_task(str(case.get("question") or ""))
    return {
        "schema_valid": not missing,
        "missing_fields": missing,
        "weights_sum": round(sum(weights), 6),
        "weights_valid": bool(weights) and abs(sum(weights) - 1.0) <= 1e-6,
        "criteria_complete": bool(criteria) and all(criteria),
        "expected_reasoning_type": expected_key,
        "predicted_reasoning_type": routed.get("key") if routed.get("activated") else "general",
        "routing_confidence": routed.get("confidence", 0.0),
        "routing_correct": bool(expected_key) and routed.get("activated") and routed.get("key") == expected_key,
        "matched_signals": routed.get("matched_signals", []),
    }


def validate_dataset(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    duplicate_ids = {
        item for item, count in Counter(str(case.get("id") or "") for case in cases).items()
        if item and count > 1
    }
    rows = []
    for case in cases:
        checks = validate_case(case)
        rows.append({
            "id": case.get("id"),
            "category": case.get("category"),
            "subcategory": case.get("subcategory"),
            "difficulty": case.get("difficulty"),
            "question": case.get("question"),
            "duplicate_id": str(case.get("id") or "") in duplicate_ids,
            "dataset_checks": checks,
        })
    return rows


def run_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run questions only; references and rubrics never enter the answer context."""
    rows: list[dict[str, Any]] = []
    with temporary_store(DB_PATH) as store:
        for index, case in enumerate(cases, 1):
            row = run_graph_case(store, case)
            reasoning = row.get("reasoning") if isinstance(row.get("reasoning"), dict) else {}
            profile = classify_reasoning_task(str(case["question"]))
            local_review = review_reasoning_answer(profile, str(row.get("answer") or ""))
            rows.append({
                **row,
                "reference": case.get("reference", ""),
                "assumptions": case.get("assumptions", []),
                "reasoning_trace": case.get("reasoning_trace", []),
                "key_insights": case.get("key_insights", []),
                "common_errors": case.get("common_errors", []),
                "evaluation_points": case.get("evaluation_points", []),
                "dataset_checks": validate_case(case),
                "reasoning_review": reasoning.get("review") or local_review,
            })
            print(
                f"[{index}/{len(cases)}] {case['id']} "
                f"route={rows[-1]['dataset_checks']['predicted_reasoning_type']} "
                f"local={'pass' if rows[-1]['reasoning_review'].get('passed') else 'warn'}",
                flush=True,
            )
    return rows


def summarize(rows: list[dict[str, Any]], *, executed: bool) -> dict[str, Any]:
    routing = [row.get("dataset_checks", {}) for row in rows]
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"cases": 0, "routing_correct": 0, "local_pass": 0})
    for row in rows:
        category = str(row.get("category") or "未分类")
        by_category[category]["cases"] += 1
        by_category[category]["routing_correct"] += int(bool(row.get("dataset_checks", {}).get("routing_correct")))
        if executed:
            by_category[category]["local_pass"] += int(bool(row.get("reasoning_review", {}).get("passed")))
    return {
        "benchmark": "general_reasoning_15_v1",
        "executed": executed,
        "cases": len(rows),
        "schema_valid": sum(bool(item.get("schema_valid")) for item in routing),
        "weights_valid": sum(bool(item.get("weights_valid")) for item in routing),
        "routing_correct": sum(bool(item.get("routing_correct")) for item in routing),
        "routing_accuracy": round(
            sum(bool(item.get("routing_correct")) for item in routing) / max(1, len(rows)), 4,
        ),
        "local_reasoning_pass": (
            sum(bool(row.get("reasoning_review", {}).get("passed")) for row in rows)
            if executed else None
        ),
        "semantic_score": None,
        "semantic_score_note": (
            "本日志的本地检查只验证可观察结构，不冒充语义正确率；语义正确性需由隔离的独立裁判或人工复核评分。"
        ),
        "by_category": dict(sorted(by_category.items())),
    }


def write_reports(
    rows: list[dict[str, Any]],
    *,
    report_dir: Path,
    stamp: str,
    executed: bool,
) -> dict[str, str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows, executed=executed)
    stem = f"general-reasoning-{'run' if executed else 'validation'}-{stamp}"
    json_path = report_dir / f"{stem}.json"
    jsonl_path = report_dir / f"{stem}.jsonl"
    markdown_path = report_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    jsonl_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# 通用推理 Agent Benchmark 日志", "",
        f"- 时间：{stamp}",
        f"- 模式：{'完整回答回归' if executed else '数据与路由校验'}",
        f"- 样本数：{summary['cases']}",
        f"- Schema 通过：{summary['schema_valid']}/{summary['cases']}",
        f"- Rubric 权重通过：{summary['weights_valid']}/{summary['cases']}",
        f"- 类型路由正确：{summary['routing_correct']}/{summary['cases']}（{summary['routing_accuracy']:.1%}）",
        f"- 本地可观察推理检查通过：{summary['local_reasoning_pass'] if executed else '未运行回答'}",
        "- 重要边界：本地规则不评判答案语义正确性；参考答案不会传给被测 Agent。", "",
        "## 分板块", "",
        "| 板块 | 题数 | 路由正确 | 本地检查通过 |", "|---|---:|---:|---:|",
    ]
    for category, item in summary["by_category"].items():
        lines.append(
            f"| {category} | {item['cases']} | {item['routing_correct']} | "
            f"{item['local_pass'] if executed else '-'} |"
        )
    lines.extend(["", "## 逐题日志", ""])
    for row in rows:
        checks = row.get("dataset_checks", {})
        lines.extend([
            f"### {row.get('id')} · {row.get('category')}", "",
            f"- 路由：{checks.get('predicted_reasoning_type')}；期望：{checks.get('expected_reasoning_type')}；"
            f"结果：{'通过' if checks.get('routing_correct') else '不匹配'}",
            f"- 命中信号：{'、'.join(checks.get('matched_signals', [])) or '无'}",
        ])
        if executed:
            review = row.get("reasoning_review", {})
            lines.extend([
                f"- 本地推理检查：{'通过' if review.get('passed') else '需复核'}",
                f"- 问题：{'；'.join(review.get('issues', [])) or '无'}", "",
                "**Agent 回答**", "", str(row.get("answer") or "（空）"), "",
                "<details><summary>人工/独立裁判参考答案与 rubric</summary>", "",
                str(row.get("reference") or ""), "",
                *[
                    f"- {item.get('criterion')}（{float(item.get('weight') or 0):.0%}）"
                    for item in row.get("evaluation_points", []) if isinstance(item, dict)
                ], "", "</details>", "",
            ])
        else:
            lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "json": str(json_path),
        "jsonl": str(jsonl_path),
        "markdown": str(markdown_path),
    }


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Run the transferable general-reasoning benchmark")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--run", action="store_true", help="Call the project Agent; default only validates data and routing")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="")
    args = parser.parse_args()
    if not args.allow_network:
        os.environ["GARDEN_DISABLE_NETWORK"] = "1"
    cases = load_cases(args.dataset)
    selected_ids = {item.strip() for item in args.ids.split(",") if item.strip()}
    if selected_ids:
        cases = [case for case in cases if str(case.get("id")) in selected_ids]
    if args.limit:
        cases = cases[:args.limit]
    rows = run_cases(cases) if args.run else validate_dataset(cases)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    paths = write_reports(rows, report_dir=args.report_dir, stamp=stamp, executed=args.run)
    print(json.dumps({"summary": summarize(rows, executed=args.run), "reports": paths}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
