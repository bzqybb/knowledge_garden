from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from core.config import DB_PATH
from core.storage import GardenStore
from evals.adapter import load_cases
from evals.boundary_eval import run_case
from evals.judge_config import (
    judge_api_key, judge_base_url, judge_label, judge_model, judge_request_options, judge_slug,
)
from evals.zhili_three_layer import classify_case


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = ROOT / "evals" / "datasets" / "zhili_deep_reasoning_50_v1.jsonl"
REPORT_DIR = ROOT / "evals" / "reports"
DIMENSIONS = (
    "mathematical_correctness",
    "proof_completeness",
    "derivation_validity",
    "calculation_accuracy",
    "premise_critique",
    "evidence_faithfulness",
    "citation_integrity",
    "reasoning_clarity",
    "uncertainty_calibration",
)
LABELS = {
    "mathematical_correctness": "结论与科学正确性",
    "proof_completeness": "证明链条完整性",
    "derivation_validity": "推导步骤有效性",
    "calculation_accuracy": "计算与结果准确性",
    "premise_critique": "错误前提识别与纠正",
    "evidence_faithfulness": "答案与证据一致性",
    "citation_integrity": "引用真实与对应程度",
    "reasoning_clarity": "解释清晰与条件交代",
    "uncertainty_calibration": "不确定性与能力边界",
}
PREMISE_LABELS = {
    "valid": "命题成立",
    "missing_condition": "题目遗漏关键条件",
    "false": "题目结论错误",
    "false_dichotomy": "题目包含错误对立",
    "unit_convention_required": "需要说明单位制或约定",
}


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def premise_identified(case: dict[str, Any], answer: str) -> bool | None:
    status = str(case.get("premise_status") or "valid")
    if status == "valid":
        return None
    text = _compact(answer)
    if not text:
        return False
    case_id = str(case.get("id") or "")
    if case_id == "P-28":
        has_helmholtz = bool(re.search(r"亥姆霍兹|helmholtz|(?:δ|△|Δ|∆)\s*[fa]", text, re.IGNORECASE))
        has_gibbs_conditions = "恒温恒压" in text or "等温等压" in text
        return has_helmholtz and has_gibbs_conditions
    if case_id == "P-22":
        return bool(re.search(r"显(?:含|式).*时间|∂a/∂t|partial.*时间", text, re.IGNORECASE))
    if case_id == "P-06":
        return any(marker in text for marker in ("分裂", "代数闭", "复数域", "复数域上"))
    if case_id == "P-42":
        return ("上界" in text or "θ(n)" in text or "下界" in text) and "log" in text
    if case_id == "P-44":
        return "链式法则" in text and any(marker in text for marker in ("本质", "并不是", "不是", "高效"))
    if case_id == "P-45":
        return any(marker in text for marker in ("满列秩", "可逆", "伪逆", "moore"))
    return any(_compact(marker) in text for marker in case.get("premise_keywords", []) if _compact(marker))


def deep_local_checks(case: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    answer = str(row.get("answer") or "")
    existing = dict(row.get("local_checks") or {})
    source_ids = {
        str(item.get("source_id") or "")
        for item in row.get("citations", []) if isinstance(item, dict)
    }
    mentioned_ids = set(re.findall(r"\[((?:M|L|W|A|T|P)\d+)\]", answer))
    formula_count = len(re.findall(r"(?:=|≤|≥|∂|∇|∫|√|\\(?:frac|sum|int|sqrt))", answer))
    reasoning_steps = len(re.findall(r"因此|所以|从而|故|代入|由此|根据|令|假设|当且仅当|\bif\b", answer))
    coverage = str(row.get("coverage_status") or "")
    calculation_correct = None
    if str(case.get("id")) == "P-10":
        calculation_correct = bool(re.search(r"√\s*33|sqrt\s*\{?33|\\sqrt\s*\{?33", answer, re.IGNORECASE))
    elif str(case.get("id")) == "P-12":
        calculation_correct = bool(re.search(r"奇异值.{0,35}2.{0,20}0", answer, re.DOTALL))
    return {
        **existing,
        "formula_count": formula_count,
        "reasoning_step_markers": reasoning_steps,
        "premise_identified": premise_identified(case, answer),
        "calculation_reference_matched": calculation_correct,
        "unknown_citation_ids": sorted(item for item in mentioned_ids if item not in source_ids),
        "textbook_coverage_available": coverage in {"textbook_grounded", "local_note_grounded"},
        "honest_uncovered_abstention": bool(existing.get("refused")) and coverage == "no_local_evidence",
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    judged = [row for row in rows if isinstance(row.get("judge"), dict) and not row["judge"].get("error")]
    durations = [float(row.get("latency_ms") or 0) for row in rows if row.get("latency_ms")]
    traps = [row for row in rows if str(row.get("premise_status") or "valid") != "valid"]
    covered = [row for row in rows if row.get("coverage_status") in {"textbook_grounded", "local_note_grounded"}]
    dimensions = {}
    for dimension in DIMENSIONS:
        values = [
            float(row["judge"][dimension])
            for row in judged if isinstance(row["judge"].get(dimension), (int, float))
        ]
        if values:
            dimensions[dimension] = round(statistics.fmean(values), 2)
    by_discipline: dict[str, dict[str, Any]] = {}
    for discipline in sorted({str(row.get("discipline") or "未分类") for row in rows}):
        items = [row for row in rows if str(row.get("discipline") or "未分类") == discipline]
        by_discipline[discipline] = {
            "cases": len(items),
            "grounded": sum(item.get("coverage_status") in {"textbook_grounded", "local_note_grounded"} for item in items),
            "verdicts": dict(Counter(str(item.get("judge", {}).get("verdict") or "unscored") for item in items)),
        }
    nodes: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for node, elapsed in row.get("node_timings_ms", {}).items():
            nodes[str(node)].append(float(elapsed))
    return {
        "cases": len(rows),
        "judged": len(judged),
        "judge_model": judge_model(),
        "textbook_grounded_cases": len(covered),
        "coverage": dict(Counter(str(row.get("coverage_status") or "unknown") for row in rows)),
        "average_latency_seconds": round(statistics.fmean(durations) / 1000, 2) if durations else None,
        "median_latency_seconds": round(statistics.median(durations) / 1000, 2) if durations else None,
        "p90_latency_seconds": round(sorted(durations)[max(0, int(len(durations) * 0.9 + 0.999) - 1)] / 1000, 2) if durations else None,
        "premise_traps": len(traps),
        "premise_traps_identified": sum(bool(row.get("local_checks", {}).get("premise_identified")) for row in traps),
        "accepted_false_premises": sum(bool(row.get("judge", {}).get("accepted_false_premise")) for row in judged),
        "hallucinations": sum(bool(row.get("judge", {}).get("hallucination")) for row in judged),
        "honest_uncovered_abstentions": sum(bool(row.get("local_checks", {}).get("honest_uncovered_abstention")) for row in rows),
        "verdicts": dict(Counter(str(row.get("judge", {}).get("verdict") or "unscored") for row in rows)),
        "mean_scores_out_of_5": dimensions,
        "by_discipline": by_discipline,
        "slowest_nodes_seconds": {
            node: round(statistics.fmean(values) / 1000, 2)
            for node, values in sorted(nodes.items(), key=lambda item: statistics.fmean(item[1]), reverse=True)[:7]
        },
    }


def improvement_findings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings = []
    for row in rows:
        judge = row.get("judge") if isinstance(row.get("judge"), dict) else {}
        if judge.get("error"):
            continue
        verdict = str(judge.get("verdict") or "")
        if verdict not in {"warn", "fail"} and not judge.get("hallucination") and not judge.get("accepted_false_premise"):
            continue
        findings.append({
            "id": row.get("id"),
            "discipline": row.get("discipline"),
            "verdict": verdict,
            "coverage_status": row.get("coverage_status"),
            "accepted_false_premise": bool(judge.get("accepted_false_premise")),
            "hallucination": bool(judge.get("hallucination")),
            "issues": str(judge.get("issues") or ""),
            "suggestion": str(judge.get("suggestion") or ""),
        })
    return sorted(findings, key=lambda item: (item["verdict"] != "fail", not item["accepted_false_premise"], item["id"]))


def write_reports(rows: list[dict[str, Any]], stamp: str) -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    slug = judge_slug()
    json_path = REPORT_DIR / f"zhili-deep-reasoning-{slug}-{stamp}.json"
    markdown_path = REPORT_DIR / f"zhili-deep-reasoning-{slug}-{stamp}.md"
    summary = summarize(rows)
    findings = improvement_findings(rows)
    payload = {
        "summary": summary, "actionable_judge_findings": findings,
        "actionable_kimi_findings": findings, "rows": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 知识花园：基础学科证明、推导、计算与错误前提专项评测", "",
        f"- 测评时间：{stamp}",
        f"- 项目回答模型：当前项目配置的 GLM；独立裁判：{judge_label(summary['judge_model'])}",
        f"- 已完成：{summary['cases']} 题；独立裁判已评分：{summary['judged']} 题；已有完整教材依据：{summary['textbook_grounded_cases']} 题",
        f"- 平均耗时：{summary.get('average_latency_seconds')} 秒；中位数：{summary.get('median_latency_seconds')} 秒；P90：{summary.get('p90_latency_seconds')} 秒",
        f"- 问题前提有陷阱：{summary['premise_traps']} 题；已主动识别：{summary['premise_traps_identified']} 题；误接受错误前提：{summary['accepted_false_premises']} 题",
        f"- 独立裁判发现幻觉：{summary['hallucinations']}；无教材时诚实说明证据不足：{summary['honest_uncovered_abstentions']} 题",
        f"- 裁判结论：{summary['verdicts']}", "", "## 各维度平均分（满分 5）", "",
        *[f"- {LABELS.get(name, name)}：{score}" for name, score in summary["mean_scores_out_of_5"].items()],
        "", "## 学科覆盖与表现", "",
        *[
            f"- {discipline}：{item['cases']} 题；完整教材依据 {item['grounded']} 题；评审 {item['verdicts']}"
            for discipline, item in summary["by_discipline"].items()
        ],
        "", "## 独立裁判发现的优先改进事项", "",
        *[
            f"- {item['id']} [{item['verdict']}] {item['issues']} 建议：{item['suggestion']}"
            for item in findings[:25]
        ],
        *( ["- 暂无需要优先修复的问题。"] if not findings else [] ),
        "", "## 最耗时工作流节点", "",
        *[f"- {node}：平均 {seconds} 秒" for node, seconds in summary["slowest_nodes_seconds"].items()],
        "", "## 逐题问题、完整回答、教材引用与独立裁判建议", "",
    ]
    for row in sorted(rows, key=lambda item: str(item.get("id") or "")):
        judge = row.get("judge") if isinstance(row.get("judge"), dict) else {}
        checks = row.get("local_checks", {})
        lines.extend([
            f"### {row['id']} · {row.get('discipline', '')} · {row.get('category', '')}", "",
            f"**问题：** {row['question']}", "",
            f"- 教材覆盖：{row.get('coverage_status', 'unknown')}；证据层：{row.get('evidence_layer', 'none')}；耗时：{round(float(row.get('latency_ms') or 0)/1000, 2)} 秒",
            f"- 题目前提：{PREMISE_LABELS.get(str(row.get('premise_status', 'valid')), str(row.get('premise_status')))}",
        ])
        if row.get("premise_note"):
            lines.extend([
                f"- 应当识别的问题：{row['premise_note']}",
                f"- 是否主动识别：{'是' if checks.get('premise_identified') else '否'}",
            ])
        lines.extend([
            f"- 推理连接词：{checks.get('reasoning_step_markers', 0)}；公式标记：{checks.get('formula_count', 0)}",
            "", "**园丁实际回答：**", "", str(row.get("answer") or "（未生成回答）"), "",
        ])
        if row.get("citations"):
            lines.extend(["**实际引用的教材页：**", "", *[
                f"- [{item.get('source_id', '')}] {item.get('title', '')}"
                for item in row.get("citations", []) if isinstance(item, dict)
            ], ""])
        if judge.get("error"):
            lines.extend([f"**独立裁判评分异常：** {judge['error']}", ""])
        elif judge:
            scores = "；".join(
                f"{LABELS.get(name, name)} {judge.get(name)}/5"
                for name in DIMENSIONS if isinstance(judge.get(name), (int, float))
            )
            lines.extend([
                f"**独立裁判：** {judge.get('verdict', 'unknown')}；{scores}", "",
                f"- 存在幻觉：{'是' if judge.get('hallucination') else '否'}；误接受错误前提：{'是' if judge.get('accepted_false_premise') else '否'}",
                f"- 做得好的地方：{judge.get('strengths', '')}",
                f"- 裁判指出的问题：{judge.get('issues', '')}",
                f"- 裁判改进建议：{judge.get('suggestion', '')}", "",
            ])
        else:
            lines.extend(["**独立裁判：** 等待评分。", ""])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, markdown_path


async def score_rows(rows: list[dict[str, Any]], stamp: str, *, workers: int = 3) -> None:
    key = judge_api_key()
    if not key:
        raise RuntimeError("未找到已保存的腾讯云 TokenHub 独立裁判密钥")
    model = judge_model()
    default_timeout = "150" if model == "kimi-k3" else "75"
    client = AsyncOpenAI(
        api_key=key,
        base_url=judge_base_url(),
        timeout=float(os.getenv("DEEP_JUDGE_TIMEOUT_SECONDS", default_timeout)),
        max_retries=0 if model == "kimi-k3" else 1,
    )
    semaphore = asyncio.Semaphore(max(1, workers))
    quota_exhausted = asyncio.Event()
    judge_prompt = (
        "你是严格、独立的基础科学与数学推理裁判，不是回答者。逐题核对教材依据、标准答案和实际回答。"
        "证明题必须检查命题假设、关键引理、每个逻辑步骤和结论；推导题检查公式、边界条件、"
        "物理量意义、近似与单位；计算题检查中间运算、最终结果和验证；辨析题检查定义、反例和适用条件。"
        "最重要：如果题目本身错了、缺少假设、混淆单位或形成错误对立，答案应先明确纠正；"
        "绝不能顺从用户替错误结论构造证明。请参考 premise_status、premise_note 和 reference。"
        "已有教材的可核验事实应有真实相关来源，正文来源编号必须与 citations 和 evidence_passages 对应；"
        "不可将仅解释前置知识的页面说成证明整条定理或算法。"
        "如果教材确实没有覆盖而系统诚实说明证据不足，不属于幻觉或错误前提接受；"
        "可给 uncertainty_calibration 高分，但若题目本应作答而回答没有给出所求结论、证明或计算过程，"
        "verdict 至多为 warn，不能因为拒答安全就判 pass。pass 表示任务已被实质、正确、完整地完成；"
        "warn 表示边界诚实但未完成，或答案有局部缺漏；fail 表示实质错误、幻觉或接受错误前提。"
        "来源只有部分覆盖时，要区分可由教材严密演绎的数学步骤与凭空虚构教材内容。"
        "只有明确编造事实、公式、来源、实验或无效推导时 hallucination 才为 true。"
        "只有答案实质认可或证明了错误前提时 accepted_false_premise 才为 true；诚实拒答不算接受。"
        "每个适用维度评分 1-5，不适用填 null。只输出 JSON，字段为 "
        "mathematical_correctness,proof_completeness,derivation_validity,calculation_accuracy,"
        "premise_critique,evidence_faithfulness,citation_integrity,reasoning_clarity,"
        "uncertainty_calibration,hallucination,accepted_false_premise,verdict(pass/warn/fail),"
        "strengths,issues,suggestion。最后三个字段是具体、简洁的中文字符串。"
    )
    if model == "kimi-k3":
        judge_prompt = (
            "你是独立的基础科学推理裁判。只依据题目、reference、实际回答和 evidence_passages 评分，"
            "不要代替系统补证据。证明题检查假设、两个方向、关键等式和结论；引用必须与来源编号对应。"
            "安全拒答若未完成本应回答的任务，至多 warn。pass=正确且完整，warn=局部缺漏，fail=实质错误或幻觉。"
            "只输出 JSON：mathematical_correctness,proof_completeness,derivation_validity,"
            "calculation_accuracy,premise_critique,evidence_faithfulness,citation_integrity,"
            "reasoning_clarity,uncertainty_calibration 均为 1-5 或 null；"
            "hallucination,accepted_false_premise 为布尔值；verdict 为 pass/warn/fail；"
            "strengths,issues,suggestion 为简洁中文字符串。"
        )

    async def one(row: dict[str, Any]) -> None:
        if isinstance(row.get("judge"), dict) and not row["judge"].get("error"):
            return
        citation_ids = {
            str(item.get("title") or ""): str(item.get("source_id") or "")
            for item in row.get("citations", []) if isinstance(item, dict)
        }
        evidence_passages = [
            {"source_id": citation_ids.get(str(title), ""), "title": str(title), "text": str(context)}
            for title, context in zip(row.get("retrieved_titles", []), row.get("retrieved_contexts", []))
        ]
        data = {
            key: row.get(key)
            for key in (
                "id", "discipline", "category", "question", "reference", "premise_status",
                "premise_note", "coverage_status", "answer", "citations", "evidence_review", "local_checks",
            )
        }
        data["evidence_passages"] = evidence_passages
        async with semaphore:
            if quota_exhausted.is_set():
                row["judge"] = {
                    "error": "独立评分已暂停：腾讯云免费试用额度耗尽，且未开启后付费。",
                }
                write_reports(rows, stamp)
                return
            started = time.perf_counter()
            try:
                for attempt in range(2):
                    response = await client.chat.completions.create(
                        model=model,
                        max_tokens=900 if model == "kimi-k3" else 1800,
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": judge_prompt},
                            {"role": "user", "content": json.dumps(data, ensure_ascii=False)},
                        ],
                        **judge_request_options(model),
                    )
                    content = str(response.choices[0].message.content or "").strip()
                    if content:
                        row["judge"] = json.loads(content)
                        break
                    if attempt:
                        raise ValueError("独立裁判连续两次返回空评分内容")
                row["judge"]["judge_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            except Exception as exc:
                if getattr(exc, "status_code", None) == 402:
                    quota_exhausted.set()
                    row["judge"] = {
                        "error": "独立评分已暂停：腾讯云免费试用额度耗尽，且未开启后付费（HTTP 402）。",
                    }
                else:
                    row["judge"] = {"error": f"{type(exc).__name__}: {str(exc)[:260]}"}
            write_reports(rows, stamp)
            judge = row["judge"]
            print(
                f"[{model} {row['id']}] {judge.get('verdict', 'ERROR')} "
                f"正确={judge.get('mathematical_correctness', '-')} "
                f"前提={judge.get('premise_critique', '-')} "
                f"幻觉={judge.get('hallucination', '-')}",
                flush=True,
            )

    try:
        await asyncio.gather(*(one(row) for row in rows))
    finally:
        await client.close()


def prepare_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    notes = GardenStore(DB_PATH).list_notes(limit=50_000)
    return [classify_case(case, notes) for case in cases]


def main() -> None:
    # Scientific questions contain characters that legacy Windows code pages
    # cannot represent. Keep reports printable without changing global locale.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Evaluate proofs, derivations and incorrect premises with an independent judge")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--judge-workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--coverage-only", action="store_true")
    parser.add_argument("--input-report", type=Path)
    parser.add_argument("--overlay-report", type=Path)
    parser.add_argument("--reset-judge", action="store_true")
    parser.add_argument("--merge-report", type=Path)
    args = parser.parse_args()
    os.environ["GARDEN_DISABLE_NETWORK"] = "1"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    selected_ids = {item.strip().upper() for item in str(args.ids or "").split(",") if item.strip()}
    if args.input_report:
        payload = json.loads(args.input_report.read_text(encoding="utf-8"))
        rows = list(payload.get("rows", []))
        if selected_ids:
            rows = [row for row in rows if str(row.get("id") or "").upper() in selected_ids]
    else:
        cases = load_cases(args.dataset)
        if selected_ids:
            cases = [case for case in cases if str(case.get("id") or "").upper() in selected_ids]
        if args.limit:
            cases = cases[:args.limit]
        print(f"检查 {len(cases)} 道题与本地教材的实际对应关系……", flush=True)
        cases = prepare_cases(cases)
        counts = dict(Counter(str(case.get("coverage_status")) for case in cases))
        traps = [str(case["id"]) for case in cases if str(case.get("premise_status") or "valid") != "valid"]
        print(f"教材覆盖：{counts}；需要识别的前提陷阱：{', '.join(traps) or '无'}", flush=True)
        if args.coverage_only:
            for case in cases:
                print(f"[{case['id']}] {case['coverage_status']} {case['question']}", flush=True)
            return
        rows = []
        if args.merge_report:
            existing = json.loads(args.merge_report.read_text(encoding="utf-8"))
            excluded = selected_ids or {str(case.get("id") or "").upper() for case in cases}
            rows = [row for row in existing.get("rows", []) if str(row.get("id") or "").upper() not in excluded]
        completed = 0
        print(f"开始项目模型回答：{len(cases)} 题，最多并发 {max(1, args.workers)} 条。", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {executor.submit(run_case, case): case for case in cases}
            for future in as_completed(futures):
                case = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        **case, "answer": "", "latency_ms": 0, "node_timings_ms": {},
                        "local_checks": {"review_passed": False},
                        "generation_error": f"{type(exc).__name__}: {str(exc)[:300]}",
                    }
                row["local_checks"] = deep_local_checks(case, row)
                rows.append(row)
                completed += 1
                write_reports(rows, stamp)
                print(
                    f"[回答 {completed}/{len(cases)}] {row['id']} "
                    f"{float(row.get('latency_ms') or 0)/1000:.1f}s "
                    f"证据={row.get('evidence_layer', 'none')} "
                    f"前提识别={row['local_checks'].get('premise_identified', '-')}",
                    flush=True,
                )
    if args.overlay_report:
        overlay = json.loads(args.overlay_report.read_text(encoding="utf-8"))
        replacements = {str(row.get("id") or ""): row for row in overlay.get("rows", [])}
        rows = [replacements.get(str(row.get("id") or ""), row) for row in rows]
    if args.reset_judge:
        for row in rows:
            row.pop("judge", None)
    if args.limit and args.input_report:
        rows = rows[:args.limit]
    if not args.skip_judge:
        asyncio.run(score_rows(rows, stamp, workers=args.judge_workers))
    json_path, markdown_path = write_reports(rows, stamp)
    print(json.dumps({
        "summary": summarize(rows),
        "actionable_judge_findings": improvement_findings(rows),
        "report_json": str(json_path),
        "report_markdown": str(markdown_path),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
