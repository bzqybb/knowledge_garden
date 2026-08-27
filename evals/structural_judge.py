from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from evals.advanced_learning_eval import (
    FRONTIER_SUITE,
    enforce_hard_assertion_verdict,
    frontier_judge_payload,
    frontier_judge_prompt,
    run_hard_assertions,
)
from evals.judge_config import (
    judge_api_key,
    judge_base_url,
    judge_label,
    judge_model,
    judge_request_options,
    judge_slug,
)


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "evals" / "reports"
SYSTEM_PROMPT = (
    "你是与被测知识花园隔离的结构推理裁判，只在回答完成后评分。"
    "只依据题目、参考要点、常见失败和实际回答；参考要点不是唯一措辞。"
    "分别给 structure_identification、premise_check、method_correctness、"
    "derivation_completeness、boundary_calibration 五项 0-2 分。"
    "2=实质正确完整，1=方向正确但有局部缺口，0=缺失或实质错误。"
    "安全拒答若题目本可由给定条件作答，不能判 pass；篇幅长、格式漂亮或与参考措辞相同均不自动加分。"
    "指出 first_material_error（首个实质错误）、failure_stage、issues、suggestion。"
    "verdict 只能是 pass/warn/fail；只输出 JSON。"
)


def load_report(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        raise ValueError("input report rows must be a list")
    return [row for row in rows if isinstance(row, dict)]


def select_rows(
    rows: list[dict[str, Any]], *, only_local_failures: bool, ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    # New reports explicitly mark endpoint/runtime/empty-fallback rows as
    # unscorable. They must not be converted into artificial semantic fails.
    # Legacy reports without this field retain their historical behavior.
    selected = [
        row for row in rows
        if row.get("execution", {}).get("scorable", True)
    ]
    if ids:
        selected = [row for row in selected if str(row.get("id")) in ids]
    if only_local_failures:
        selected = [row for row in selected if not row.get("observable_checks", {}).get("passed")]
    return selected


def judge_payload(row: dict[str, Any]) -> dict[str, Any]:
    if str(row.get("suite") or "") == FRONTIER_SUITE:
        return frontier_judge_payload(row)
    return {
        "id": row.get("id"),
        "question": row.get("question"),
        "reference": row.get("reference"),
        "common_failures": [item for item in row.get("common_failures", []) if item],
        "answer": row.get("answer"),
    }


def judge_system_prompt(row: dict[str, Any]) -> str:
    if str(row.get("suite") or "") != FRONTIER_SUITE:
        return SYSTEM_PROMPT
    return (
        frontier_judge_prompt(row)
        + " 分别给 structure_identification、premise_check、method_correctness、"
        "derivation_completeness、boundary_calibration 五项 0-2 分。"
        "2=实质正确完整，1=方向正确但有局部缺口，0=缺失或实质错误。"
        "指出 first_material_error、failure_stage、issues、suggestion；"
        "verdict 只能是 pass/warn/fail。参考措辞不是唯一答案，只输出 JSON。"
    )


def valid_judge(result: dict[str, Any]) -> bool:
    dimensions = (
        "structure_identification", "premise_check", "method_correctness",
        "derivation_completeness", "boundary_calibration",
    )
    verdict = result.get("verdict")
    material_error_valid = verdict == "pass" or bool(str(result.get("first_material_error") or "").strip())
    return (
        all(isinstance(result.get(key), (int, float)) and 0 <= result[key] <= 2 for key in dimensions)
        and verdict in {"pass", "warn", "fail"}
        and material_error_valid
    )


def summarize(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    verdicts = {key: 0 for key in ("pass", "warn", "fail", "error")}
    totals = []
    for row in rows:
        result = row.get("judge", {})
        verdict = str(result.get("verdict") or "error")
        verdicts[verdict if verdict in verdicts else "error"] += 1
        scores = [
            result.get(key) for key in (
                "structure_identification", "premise_check", "method_correctness",
                "derivation_completeness", "boundary_calibration",
            ) if isinstance(result.get(key), (int, float))
        ]
        if len(scores) == 5:
            totals.append(sum(scores))
    return {
        "judge": judge_label(model),
        "judge_model": model,
        "cases": len(rows),
        "verdicts": verdicts,
        "mean_total_out_of_10": round(sum(totals) / len(totals), 3) if totals else None,
        "models_used": sorted({
            str(row.get("judge", {}).get("judge_model_used"))
            for row in rows if row.get("judge", {}).get("judge_model_used")
        }),
        "scope": "只复核回答完成后的候选；裁判结果不自动修改规则。",
    }


def write_report(rows: list[dict[str, Any]], *, model: str, stamp: str, report_dir: Path) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = f"structural-independent-judge-{judge_slug(model)}-{stamp}"
    json_path = report_dir / f"{stem}.json"
    md_path = report_dir / f"{stem}.md"
    summary = summarize(rows, model)
    json_path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 结构推理独立裁判", "", f"- 裁判：{summary['judge']}", f"- 样本：{len(rows)}",
        f"- verdict：{summary['verdicts']}", f"- 平均总分：{summary['mean_total_out_of_10']}/10",
        f"- 边界：{summary['scope']}", "",
    ]
    for row in rows:
        result = row.get("judge", {})
        lines.extend([
            f"## {row.get('id')} · {result.get('verdict', 'error')}", "",
            f"- 首个实质错误：{result.get('first_material_error') or result.get('error', '')}",
            f"- 失败阶段：{result.get('failure_stage', '')}",
            f"- 问题：{result.get('issues', '')}",
            f"- 建议：{result.get('suggestion', '')}", "",
        ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


async def score(
    rows: list[dict[str, Any]], *, model: str, fallback_model: str,
    workers: int, report_dir: Path,
) -> tuple[Path, Path]:
    key = judge_api_key()
    if not key:
        raise RuntimeError("未找到已保存的腾讯云 TokenHub 独立裁判密钥")
    client = AsyncOpenAI(api_key=key, base_url=judge_base_url(), timeout=90, max_retries=1)
    semaphore = asyncio.Semaphore(max(1, workers))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    async def one(row: dict[str, Any]) -> None:
        async with semaphore:
            started = time.perf_counter()
            raw_result: dict[str, Any] | None = None
            try:
                response = None
                selected_model = model
                candidates = [model] + ([fallback_model] if fallback_model and fallback_model != model else [])
                for candidate_index, candidate in enumerate(candidates):
                    selected_model = candidate
                    try:
                        response = await client.chat.completions.create(
                            model=candidate,
                            max_tokens=1200,
                            response_format={"type": "json_object"},
                            messages=[
                                {"role": "system", "content": judge_system_prompt(row)},
                                {"role": "user", "content": json.dumps(judge_payload(row), ensure_ascii=False)},
                            ],
                            **judge_request_options(candidate),
                        )
                        break
                    except Exception as candidate_exc:
                        if getattr(candidate_exc, "status_code", None) == 402 and candidate_index + 1 < len(candidates):
                            continue
                        raise
                if response is None:
                    raise RuntimeError("独立裁判没有返回响应")
                raw_result = json.loads(str(response.choices[0].message.content or "{}"))
                if not valid_judge(raw_result):
                    raise ValueError("裁判返回的 JSON 缺少有效评分字段")
                hard_result = run_hard_assertions(row, str(row.get("answer") or ""))
                if hard_result["applicable"]:
                    row["hard_assertions"] = hard_result
                    raw_result = enforce_hard_assertion_verdict(raw_result, hard_result)
                raw_result["judge_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                raw_result["judge_model_used"] = selected_model
                row["judge"] = raw_result
            except Exception as exc:
                row["judge"] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
                if raw_result is not None:
                    row["judge"]["raw"] = raw_result
            write_report(rows, model=model, stamp=stamp, report_dir=report_dir)
            print(
                f"[{row['judge'].get('judge_model_used', model)} {row.get('id')}] "
                f"{row['judge'].get('verdict', 'ERROR')}", flush=True,
            )

    try:
        await asyncio.gather(*(one(row) for row in rows))
    finally:
        await client.close()
    return write_report(rows, model=model, stamp=stamp, report_dir=report_dir)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Independently judge structural reasoning answers")
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--fallback-model", default="deepseek-v4-flash")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--ids", default="")
    parser.add_argument("--all", action="store_true", help="judge all rows instead of local failure candidates")
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    os.environ["JUDGE_MODEL"] = args.model
    ids = {item.strip() for item in args.ids.split(",") if item.strip()}
    rows = select_rows(load_report(args.input_report), only_local_failures=not args.all, ids=ids or None)
    if not rows:
        raise SystemExit("没有符合条件的待裁判回答")
    json_path, md_path = asyncio.run(score(
        rows, model=args.model, fallback_model=args.fallback_model,
        workers=args.workers, report_dir=args.report_dir,
    ))
    print(json.dumps({"summary": summarize(rows, args.model), "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
