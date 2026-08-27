from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from core.config import DB_PATH
from core.reasoning_capability import classify_reasoning_task
from evals.adapter import load_cases, run_graph_case, temporary_store


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "evals" / "datasets" / "zhili_structural_challenge_28_v1.jsonl"
REPORT_DIR = ROOT / "evals" / "reports"
VARIANT_LABELS = {
    "A": "同构换皮/直接推导",
    "B": "表面相似但条件不同",
    "C": "缺条件",
    "D": "错误前提",
    "E": "证据不足",
    "F": "错误回答诊断",
    "T": "作者可见测试题",
}
EXPECTED_GROUP_SIZES = {
    "数学基础：方向导数、连续与可微": 6,
    "物理基础：麦克斯韦方程与波动方程": 6,
    "化学基础：热力学判据": 6,
    "AI4S：材料性能预测": 6,
    "X-idea：大胆想法评价": 6,
    "书院成长：专业选择": 6,
}


def dataset_audit(cases: list[dict[str, Any]]) -> dict[str, Any]:
    development = [case for case in cases if case.get("split") == "development"]
    tests = [case for case in cases if case.get("split") == "test"]
    groups: dict[str, list[str]] = defaultdict(list)
    for case in development:
        groups[str(case.get("structure_group") or "未分组")].append(str(case.get("variant") or ""))
    missing_by_group = {
        group: sorted(set("ABCDEF") - set(groups.get(group, [])))
        for group in EXPECTED_GROUP_SIZES
    }
    return {
        "actual_total": len(cases),
        "actual_development": len(development),
        "actual_validation": sum(case.get("split") == "validation" for case in cases),
        "actual_test": len(tests),
        "claimed_development": 36,
        "claimed_validation": 6,
        "claimed_test": 2,
        "missing_development": 36 - len(development),
        "missing_validation": 6,
        "missing_test": 2 - len(tests),
        "missing_variants_by_group": missing_by_group,
        "duplicate_ids": [item for item, count in Counter(str(case.get("id")) for case in cases).items() if count > 1],
        "strict_blind_test": False,
        "strict_blind_test_note": "测试题及参考答案已在用户材料中公开给实现者，因此只能作为作者可见测试，不能证明严格盲测泛化。",
    }


def observable_checks(case: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    answer = str(row.get("answer") or "")
    compact = re.sub(r"\s+", "", answer)
    variant = str(case.get("variant") or "T")
    routed = row.get("reasoning") if isinstance(row.get("reasoning"), dict) else {}
    route_type = str(routed.get("type") or "general")
    self_contained = bool(routed.get("self_contained"))
    refusal = bool(re.search(r"先不补写答案|当前仍然证据不足|请补充.*(?:教材|原文)", answer))
    common = {
        "substantive": len(compact) >= 120,
        "reasoning_route_active": route_type != "general",
        "contains_key_inference": bool(re.search(r"因此|所以|由此|若|则|不能推出|意味着|取决于|=|⇒|→", answer)),
        "states_condition_or_boundary": bool(re.search(r"条件|前提|假设|边界|不确定|仅当|还需|缺失|取决于|不能仅凭", answer)),
        "unexpected_refusal": refusal and self_contained,
    }
    variant_checks = {
        "A": bool(re.search(r"推导|构造|验证|步骤|首先|其次|①|=|∇|Δ", answer)),
        "B": bool(re.search(r"不同|区别|相比|不一定|仍可能|取决于|条件", answer)),
        "C": bool(re.search(r"缺少|缺失|还需|需要了解|无法确定|不能仅凭|不完整|追问", answer)),
        "D": bool(re.search(r"不正确|错误|不成立|不能证明|前提.*(?:错|不充分)", answer)),
        "E": bool(re.search(r"证据不足|不充分|不可靠|不确定|预测区间|实验验证|还需", answer)),
        "F": bool(re.search(r"错误|问题在于|错在", answer)) and bool(re.search(r"正确|应当|应该|修正", answer)),
        "T": bool(re.search(r"不准确|不能简单|并非|框架|范式|条件|复杂", answer)),
    }
    common["variant_requirement"] = variant_checks.get(variant, False)
    passed = (
        common["substantive"]
        and common["contains_key_inference"]
        and common["variant_requirement"]
        and not common["unexpected_refusal"]
    )
    return {
        "passed": passed,
        "variant": variant,
        "variant_label": VARIANT_LABELS.get(variant, variant),
        "route_type": route_type,
        "route_label": routed.get("label", ""),
        "self_contained": self_contained,
        "checks": common,
        "issues": [
            label for key, label in (
                ("substantive", "回答过短，未形成可复核分析"),
                ("reasoning_route_active", "未激活专门推理类型"),
                ("contains_key_inference", "缺少可见推导连接"),
                ("states_condition_or_boundary", "未说明条件或边界"),
                ("variant_requirement", f"未满足{VARIANT_LABELS.get(variant, variant)}的可观察要求"),
            )
            if not common[key]
        ] + (["题设已足以推理，却被错误地按外部证据不足拒答"] if common["unexpected_refusal"] else []),
    }


def summarize(rows: list[dict[str, Any]], audit: dict[str, Any], *, executed: bool) -> dict[str, Any]:
    by_variant: dict[str, dict[str, int]] = defaultdict(lambda: {"cases": 0, "passed": 0})
    by_group: dict[str, dict[str, int]] = defaultdict(lambda: {"cases": 0, "passed": 0})
    by_capability: dict[str, dict[str, int]] = defaultdict(lambda: {"cases": 0, "passed": 0})
    for row in rows:
        passed = bool(row.get("observable_checks", {}).get("passed")) if executed else False
        variant = str(row.get("variant") or "T")
        group = str(row.get("structure_group") or "未分组")
        by_variant[variant]["cases"] += 1
        by_variant[variant]["passed"] += int(passed)
        by_group[group]["cases"] += 1
        by_group[group]["passed"] += int(passed)
        for capability in row.get("agent_capability", []):
            by_capability[str(capability)]["cases"] += 1
            by_capability[str(capability)]["passed"] += int(passed)
    executed_rows = [row for row in rows if row.get("observable_checks")]
    return {
        "benchmark": "zhili_structural_reasoning_v1",
        "executed": executed,
        "cases": len(rows),
        "observable_pass": sum(bool(row.get("observable_checks", {}).get("passed")) for row in executed_rows),
        "observable_pass_rate": round(
            sum(bool(row.get("observable_checks", {}).get("passed")) for row in executed_rows)
            / max(1, len(executed_rows)), 4,
        ) if executed else None,
        "active_reasoning_routes": sum(
            row.get("observable_checks", {}).get("route_type") != "general" for row in executed_rows
        ),
        "unexpected_self_contained_refusals": sum(
            bool(row.get("observable_checks", {}).get("checks", {}).get("unexpected_refusal"))
            for row in executed_rows
        ),
        "semantic_score": None,
        "semantic_boundary": "可观察检查不是答案正确率；逐题语义正确性仍需人工或隔离裁判依据 reference/common_failures 评分。",
        "dataset_audit": audit,
        "by_variant": dict(sorted(by_variant.items())),
        "by_group": dict(sorted(by_group.items())),
        "by_capability": dict(sorted(by_capability.items())),
    }


def write_report(rows: list[dict[str, Any]], audit: dict[str, Any], stamp: str, *, executed: bool) -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows, audit, executed=executed)
    stem = f"test-log-2-zhili-structural-{stamp}"
    json_path = REPORT_DIR / f"{stem}.json"
    md_path = REPORT_DIR / f"{stem}.md"
    json_path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 测试日志 2：致理结构对照与失败驱动推理", "",
        f"- 时间：{stamp}", f"- 实际题数：{len(rows)}",
        f"- 运行模式：{'完整 Agent 回答' if executed else '仅数据审计'}",
        f"- 可观察结构通过：{summary['observable_pass']}/{len(rows) if executed else 0}",
        f"- 专门推理路由激活：{summary['active_reasoning_routes']}/{len(rows) if executed else 0}",
        f"- 自足题错误拒答：{summary['unexpected_self_contained_refusals']}",
        "- 语义正确率：未评分；本地结构规则不冒充独立裁判。", "",
        "## 数据完整性审计", "",
        f"- 正文实际：开发 {audit['actual_development']}、验证 {audit['actual_validation']}、测试 {audit['actual_test']}。",
        f"- 文末声称：开发 {audit['claimed_development']}、验证 {audit['claimed_validation']}、测试 {audit['claimed_test']}。",
        f"- 缺口：开发 {audit['missing_development']}、验证 {audit['missing_validation']}、测试 {audit['missing_test']}。",
        f"- 严格盲测：否。{audit['strict_blind_test_note']}", "",
        "| 结构组 | 缺少的 A-F 变体 |", "|---|---|",
        *[
            f"| {group} | {', '.join(items) if items else '无'} |"
            for group, items in audit["missing_variants_by_group"].items()
        ], "", "## 分组表现", "",
        "| 结构组 | 题数 | 可观察通过 |", "|---|---:|---:|",
        *[f"| {group} | {item['cases']} | {item['passed'] if executed else '-'} |" for group, item in summary["by_group"].items()],
        "", "## 能力缺口视图", "",
        "| 目标能力 | 题数 | 可观察通过 |", "|---|---:|---:|",
        *[f"| {capability} | {item['cases']} | {item['passed'] if executed else '-'} |" for capability, item in summary["by_capability"].items()],
        "", "## 逐题日志", "",
    ]
    for row in rows:
        checks = row.get("observable_checks", {})
        lines.extend([
            f"### {row['id']} · {row.get('structure_group')} · {VARIANT_LABELS.get(str(row.get('variant')), row.get('variant'))}", "",
            f"- 目标能力：{'、'.join(row.get('agent_capability', []))}",
            f"- 推理路由：{checks.get('route_label') or checks.get('route_type') or '未运行'}",
            f"- 可观察检查：{'通过' if checks.get('passed') else '需复核'}",
            f"- 问题：{'；'.join(checks.get('issues', [])) or '无'}", "",
        ])
        if executed:
            lines.extend([
                "**Agent 回答**", "", str(row.get("answer") or "（空）"), "",
                "<details><summary>人工/独立裁判参考</summary>", "",
                str(row.get("reference") or ""), "",
                *[f"- 常见失败：{item}" for item in row.get("common_failures", [])],
                "", "</details>", "",
            ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Run test log 2 for the Zhili structural reasoning set")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="")
    args = parser.parse_args()
    if not args.allow_network:
        os.environ["GARDEN_DISABLE_NETWORK"] = "1"
    cases = load_cases(args.dataset)
    audit = dataset_audit(cases)
    wanted = {item.strip() for item in args.ids.split(",") if item.strip()}
    if wanted:
        cases = [case for case in cases if str(case.get("id")) in wanted]
    if args.limit:
        cases = cases[:args.limit]
    rows: list[dict[str, Any]] = []
    if args.run:
        with temporary_store(DB_PATH) as store:
            for index, case in enumerate(cases, 1):
                try:
                    result = run_graph_case(store, case)
                    row = {**case, **result}
                except Exception as exc:
                    row = {**case, "answer": "", "run_error": f"{type(exc).__name__}: {str(exc)[:400]}"}
                row["observable_checks"] = observable_checks(case, row)
                rows.append(row)
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                write_report(rows, audit, stamp, executed=True)
                print(
                    f"[{index}/{len(cases)}] {case['id']} "
                    f"route={row['observable_checks']['route_type']} "
                    f"observable={'pass' if row['observable_checks']['passed'] else 'review'}",
                    flush=True,
                )
    else:
        rows = cases
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path, md_path = write_report(rows, audit, stamp, executed=args.run)
    print(json.dumps({"summary": summarize(rows, audit, executed=args.run), "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
