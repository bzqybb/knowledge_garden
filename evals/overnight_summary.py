"""Consolidate overnight RAG evaluations without making any network requests."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "evals" / "reports"


def load_report(value: str) -> tuple[Path, dict[str, Any]]:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path, json.loads(path.read_text(encoding="utf-8"))


def verdict_text(report: dict[str, Any]) -> str:
    verdicts = report.get("summary", {}).get("verdicts", {})
    return " / ".join(f"{name} {count}" for name, count in verdicts.items()) or "未评分"


def report_link(path: Path) -> str:
    markdown = path.with_suffix(".md")
    return f"[{markdown.name}]({markdown.name})"


def yes_no(value: Any) -> str:
    return "是" if value else "否"


def render(
    *,
    boundary_before: tuple[Path, dict[str, Any]],
    boundary_after: tuple[Path, dict[str, Any]],
    reasoning_before: tuple[Path, dict[str, Any]],
    reasoning_after: tuple[Path, dict[str, Any]],
    full: tuple[Path, dict[str, Any]],
    sample: tuple[Path, dict[str, Any]],
    latest_local: tuple[Path, dict[str, Any]] | None = None,
    regression_tests: int = 0,
) -> str:
    boundary_old_path, boundary_old = boundary_before
    boundary_new_path, boundary_new = boundary_after
    proof_old_path, proof_old = reasoning_before
    proof_new_path, proof_new = reasoning_after
    full_path, full_report = full
    sample_path, sample_report = sample
    latest_path, latest_report = latest_local or sample
    old_boundary = boundary_old.get("summary", {})
    new_boundary = boundary_new.get("summary", {})
    old_proof = proof_old.get("summary", {})
    new_proof = proof_new.get("summary", {})
    complete = full_report.get("summary", {})
    sampled = sample_report.get("summary", {})
    full_rows = sorted(full_report.get("rows", []), key=lambda row: row.get("id", ""))
    sample_by_id = {
        str(row.get("id", "")): row for row in sample_report.get("rows", [])
    }
    latest_by_id = {
        str(row.get("id", "")): row for row in latest_report.get("rows", [])
    }
    personalization = Counter(
        str(row.get("personalization", {}).get("status", "unknown"))
        for row in boundary_new.get("rows", [])
    )
    revised_traps = sum(
        bool(latest_by_id.get(str(row.get("id")), sample_by_id.get(str(row.get("id")), row))
             .get("local_checks", {}).get("premise_identified"))
        for row in full_rows if str(row.get("premise_status", "valid")) != "valid"
    )
    durations = [
        float(row.get("latency_ms", 0)) / 1000
        for row in full_rows if float(row.get("latency_ms", 0)) > 0
    ]
    warm_durations = durations[3:] if len(durations) > 3 else durations
    warning_rows = [
        row for row in sample_report.get("rows", [])
        if str(row.get("judge", {}).get("verdict", "")) in {"warn", "fail"}
    ]
    output = [
        "# 知识花园：夜间三组测试与迭代结果",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}。入口：http://127.0.0.1:8765/",
        "",
        "## 一眼看结论",
        "",
        f"- 边界与开放问题：{new_boundary.get('cases', 0)} 题经过 Kimi 独立评审，"
        f"结论 {verdict_text(boundary_new)}；发现的幻觉从 "
        f"{old_boundary.get('hallucinations', 0)} 处降低到 {new_boundary.get('hallucinations', 0)} 处。",
        f"- 数学、物理与化学证明：{new_proof.get('cases', 0)} 道代表题经过 Kimi 复评，"
        f"发现的幻觉从 {old_proof.get('hallucinations', 0)} 处降低到 {new_proof.get('hallucinations', 0)} 处；"
        f"错误前提接受次数从 {old_proof.get('accepted_false_premises', 0)} 降至 "
        f"{new_proof.get('accepted_false_premises', 0)}。",
        f"- 全学科深度题：项目模型完成全部 {complete.get('cases', 0)} 题；其中 "
        f"{sampled.get('cases', 0)} 道此前未复评的跨学科代表题另由 Kimi 独立评审，"
        f"结论 {verdict_text(sample_report)}。",
        f"- 题目前提陷阱：整套共 {complete.get('premise_traps', 0)} 道；"
        f"首轮识别 {complete.get('premise_traps_identified', 0)} 道，修复并复测后识别 {revised_traps} 道。",
        f"- 全套 50 题回答耗时：中位数 {complete.get('median_latency_seconds', 0)} 秒，"
        f"平均 {complete.get('average_latency_seconds', 0)} 秒；去掉最先启动的 3 道后，"
        f"中位数约 {statistics.median(warm_durations):.1f} 秒。",
        *([f"- 完整项目回归测试：{regression_tests} 项全部通过。"] if regression_tests else []),
        "",
        "注意：50 题完整答案只发送给已配置的项目模型并执行本地检查；"
        "不会把整套评测数据发送给独立裁判。Kimi 结论仅覆盖明确选定的代表题，"
        "不能据此宣称全部 50 题都已完成外部事实核验。",
        *([
            "最后一次改进后的 5 道题已由项目模型重新回答并通过本地检查；"
            "但腾讯云返回 HTTP 402：免费试用额度耗尽，且没有开启后付费。"
            "因此最新 5 个答案尚未取得新的 Kimi 评分，表格只保留其上一轮独立评审结论。",
        ] if latest_local else []),
        "",
        "## 三组测试与两轮以上迭代",
        "",
        "| 组别 | 基线表现 | 迭代后表现 | 逐题完整报告 |",
        "| --- | --- | --- | --- |",
        f"| 第一组：书院生活、身体感受、哲学和价值边界 | "
        f"幻觉 {old_boundary.get('hallucinations', 0)}；{verdict_text(boundary_old)} | "
        f"幻觉 {new_boundary.get('hallucinations', 0)}；{verdict_text(boundary_new)} | "
        f"{report_link(boundary_new_path)} |",
        f"| 第二组：数学、物理、化学证明和辨析 | "
        f"幻觉 {old_proof.get('hallucinations', 0)}；接受错误前提 {old_proof.get('accepted_false_premises', 0)} | "
        f"幻觉 {new_proof.get('hallucinations', 0)}；接受错误前提 "
        f"{new_proof.get('accepted_false_premises', 0)}；{verdict_text(proof_new)} | "
        f"{report_link(proof_new_path)} |",
        f"| 第三组：生物、信息、交叉科学及剩余证明题 | "
        f"全部 {complete.get('cases', 0)} 题由项目模型回答并本地检查 | "
        f"{sampled.get('cases', 0)} 道新代表题 Kimi 复核；{verdict_text(sample_report)} | "
        f"{report_link(sample_path)}；{report_link(full_path)} |",
        *([
            f"| 第三轮修复后的最新答案 | 免费 Kimi 额度耗尽，未重新评分 | "
            f"本地检查保留完整推导、证据边界和前提说明 | {report_link(latest_path)} |",
        ] if latest_local else []),
        "",
        "## 回答速度到底卡在哪里",
        "",
        *[
            f"- `{node}`：平均 {seconds} 秒。"
            for node, seconds in complete.get("slowest_nodes_seconds", {}).items()
        ],
        "",
        "已加入本地向量模型和精排模型的 CPU 线程上限，并在应用启动后后台预热。"
        "它们减少多任务并发时的 CPU 争抢与首次冷启动，但项目模型本身的生成时间仍然是主要瓶颈之一；"
        "不能把进程内并行评测的耗时直接当成用户逐条提问的固定耗时。",
        "",
        "## 教材覆盖与诚实边界",
        "",
        *[
            f"- {discipline}：{details.get('cases', 0)} 题，题库静态判定已有完整教材覆盖 "
            f"{details.get('grounded', 0)} 题。"
            for discipline, details in complete.get("by_discipline", {}).items()
        ],
        "",
        "静态教材覆盖判定并不等同于运行时实际引用：系统有时能从已有跨学科教材找出相关段落，"
        "也可能发现教材正文不足而诚实说明证据边界；下表同时保留两种信息，避免夸大。",
        "",
        "## 为什么暂时看不到‘个性化证据’",
        "",
        f"本轮 {new_boundary.get('cases', 0)} 条边界评测的实际个性化状态："
        + "、".join(f"{name}={count}" for name, count in personalization.items())
        + "。",
        "",
        "前端已经有‘为什么这次这样讲’展开区域，后端也保留观察记录、证据编号和置信度；"
        "但当前没有已配置的教学偏好，也没有足够高置信度、与当前学科相关的历史学习证据，"
        "因此系统正确回退到‘标准讲解’，没有编造一个不存在的个性化理由。"
        "同一段对话能够承接上一问，并不等于已经积累了可审计的长期个性化证据。",
        "",
        "## 第三组 Kimi 复核发现",
        "",
    ]
    if warning_rows:
        for row in warning_rows:
            judge = row.get("judge", {})
            output.extend([
                f"- **{row.get('id')} · {judge.get('verdict')}**：{judge.get('issues', '无详细说明')}",
                f"  改进建议：{judge.get('suggestion', '无')}",
            ])
    else:
        output.append("- 本组独立复核没有需要优先处理的警告。")
    output.extend([
        "",
        "## 全部 50 题逐题索引",
        "",
        "| 编号 | 学科 | 问题 | 静态教材覆盖 | 实际证据 | 前提是否识别 | 耗时 | Kimi 抽样 |",
        "| --- | --- | --- | --- | --- | --- | ---: | --- |",
    ])
    for row in full_rows:
        case_id = str(row.get("id", ""))
        latest = latest_by_id.get(case_id, sample_by_id.get(case_id, row))
        trap = latest.get("local_checks", {}).get("premise_identified")
        trap_text = "不适用" if trap is None else yes_no(trap)
        judge = sample_by_id.get(case_id, {}).get("judge", {})
        verdict = str(judge.get("verdict", "")) if isinstance(judge, dict) else ""
        question = str(row.get("question", "")).replace("|", "\\|").replace("\n", " ")
        output.append(
            f"| {case_id} | {row.get('discipline', '')} | {question} | "
            f"{row.get('coverage_status', 'unknown')} | {latest.get('evidence_layer', 'none')} | "
            f"{trap_text} | {float(latest.get('latency_ms', 0))/1000:.1f}s | "
            f"{verdict or '未送独立裁判'} |",
        )
    output.extend([
        "",
        "各题的完整原始回答、实际教材页码、逐节点耗时与 Kimi 具体评分请打开上面的逐题报告。",
        "",
        f"基线文件：{report_link(boundary_old_path)}；{report_link(proof_old_path)}。",
        "",
    ])
    return "\n".join(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary-before", required=True)
    parser.add_argument("--boundary-after", required=True)
    parser.add_argument("--reasoning-before", required=True)
    parser.add_argument("--reasoning-after", required=True)
    parser.add_argument("--full", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--latest-local", default="")
    parser.add_argument("--regression-tests", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    content = render(
        boundary_before=load_report(args.boundary_before),
        boundary_after=load_report(args.boundary_after),
        reasoning_before=load_report(args.reasoning_before),
        reasoning_after=load_report(args.reasoning_after),
        full=load_report(args.full),
        sample=load_report(args.sample),
        latest_local=load_report(args.latest_local) if args.latest_local else None,
        regression_tests=max(0, args.regression_tests),
    )
    destination = Path(args.output) if args.output else (
        REPORTS / f"overnight-summary-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    )
    if not destination.is_absolute():
        destination = ROOT / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    print(json.dumps({"report_markdown": str(destination)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
