from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from core.config import DB_PATH, LLMConfig, llm_config
from evals.adapter import load_cases, run_graph_case, temporary_store
from evals.advanced_learning_eval import (
    FRONTIER_SUITE,
    frontier_agent_payload,
    hard_assertion_instruction,
    run_hard_assertions,
)


ROOT = Path(__file__).resolve().parent.parent
PACK = ROOT / "evals" / "datasets" / "zhili_structural_debug_v2"
REPORT_DIR = ROOT / "evals" / "reports"
RULES_PATH = PACK / "rules" / "rules_v6.md"
PHASE_SPLITS = {
    "develop": {"development"},
    "validate": {"transfer_validation"},
    "challenge": {"author_visible_challenge"},
}
VARIANTS = set("ABCDEF")
INFRASTRUCTURE_ERROR_RE = re.compile(
    r"(?:error code|status(?:_code)?|http)\s*[:=]?\s*(?:401|402|403|408|409|429|500|502|503|504)\b|"
    r"authentication(?:error)?|unauthorized|forbidden|quota|billing|rate.?limit|"
    r"timed?\s*out|timeout|connection(?:error|reset|refused)|dns|name resolution|"
    r"令牌.*(?:过期|不正确)|额度.*(?:耗尽|不足)|欠费|计费",
    re.IGNORECASE,
)
GENERATION_FALLBACK_MARKERS = (
    "没有返回可解析的实质答案",
    "未返回可解析的实质答案",
    "自足推理重试失败",
)


def normalized(text: str) -> str:
    return re.sub(r"\W+", "", text, flags=re.UNICODE).casefold()


def content_fingerprint(case: dict[str, Any]) -> str:
    return hashlib.sha256(normalized(str(case.get("question") or "")).encode("utf-8")).hexdigest()


def load_pack(pack: Path = PACK) -> dict[str, list[dict[str, Any]]]:
    return {
        "development": load_cases(pack / "development_96.jsonl"),
        "transfer_validation": load_cases(pack / "transfer_validation_32.jsonl"),
        "author_visible_challenge": load_cases(pack / "author_visible_challenge_12.jsonl"),
    }


def audit_pack(pack: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = [row for values in pack.values() for row in values]
    ids = [str(row.get("id") or "") for row in rows]
    fingerprints = [content_fingerprint(row) for row in rows]
    development = pack["development"]
    by_structure: dict[str, set[str]] = defaultdict(set)
    for row in development:
        by_structure[str(row.get("structure_id"))].add(str(row.get("variant")))
    dev_structures = set(by_structure)
    validation_structures = {str(row.get("structure_id")) for row in pack["transfer_validation"]}
    challenge_structures = {str(row.get("structure_id")) for row in pack["author_visible_challenge"]}
    required = {
        "id", "suite", "split", "structure_id", "structure_group", "variant",
        "question", "reference", "common_failures", "rule_target", "frozen",
    }
    schema_errors = {
        str(row.get("id") or f"row-{index}"): sorted(required - set(row))
        for index, row in enumerate(rows, 1) if required - set(row)
    }
    null_failures = [
        str(row["id"]) for row in rows
        if not row.get("common_failures") or any(not item for item in row["common_failures"])
    ]
    result = {
        "counts": {key: len(value) for key, value in pack.items()},
        "unique_ids": len(ids) == len(set(ids)),
        "unique_questions": len(fingerprints) == len(set(fingerprints)),
        "schema_errors": schema_errors,
        "null_common_failures": null_failures,
        "development_structures": len(dev_structures),
        "complete_contrast_groups": all(variants == VARIANTS for variants in by_structure.values()),
        "incomplete_groups": {
            key: sorted(VARIANTS - variants) for key, variants in by_structure.items()
            if variants != VARIANTS
        },
        "validation_matches_development_structures": validation_structures == dev_structures,
        "challenge_structures_disjoint": challenge_structures.isdisjoint(dev_structures | validation_structures),
        "frozen_policy_valid": (
            all(not row.get("frozen") for row in development)
            and all(row.get("frozen") for row in pack["transfer_validation"])
            and all(row.get("frozen") for row in pack["author_visible_challenge"])
        ),
        "strict_blind": False,
        "strict_blind_note": "挑战题由实现者可见，只能报告作者可见迁移快照；严格盲测需外部保管题。",
    }
    result["passed"] = all([
        result["counts"] == {"development": 96, "transfer_validation": 32, "author_visible_challenge": 12},
        result["unique_ids"], result["unique_questions"], not schema_errors, not null_failures,
        result["development_structures"] == 16, result["complete_contrast_groups"],
        result["validation_matches_development_structures"], result["challenge_structures_disjoint"],
        result["frozen_policy_valid"],
    ])
    return result


def assert_phase_allowed(cases: list[dict[str, Any]], phase: str) -> None:
    allowed = PHASE_SPLITS[phase]
    actual = {str(case.get("split") or "") for case in cases}
    if not actual.issubset(allowed):
        raise ValueError(f"phase={phase} only accepts {sorted(allowed)}, got {sorted(actual)}")


def public_case(case: dict[str, Any]) -> dict[str, Any]:
    """Return only fields allowed to enter the tested Agent context."""
    if str(case.get("suite") or "") == FRONTIER_SUITE:
        return frontier_agent_payload(case)
    return {
        "id": case["id"],
        "question": case["question"],
        "structure_group": case.get("structure_group", ""),
        "variant": case.get("variant", ""),
    }


def merge_case_result(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Keep post-answer runtime data without letting empty adapter metadata erase the rubric."""
    return {
        **result,
        **case,
        "question": case["question"],
        "answer": str(result.get("answer") or ""),
    }


def build_agent_prompt(case: dict[str, Any], *, phase: str, rules: str = "") -> str:
    if phase == "develop":
        contract = (
            "先识别本题真正需要的推理操作，再给出可核验的关键步骤与结论；"
            "检查前提、系统边界和证据强度。信息不足时明确指出缺项并给条件性结论；"
            "若任务本身不成立，应先修正任务再继续分析。"
        )
    else:
        contract = (
            "请独立作答。先判断条件是否足够及前提是否成立，再给可核验的关键步骤和条件性结论。"
            "不要猜测你可能见过的训练样例，也不要复述模板。"
        )
    rules_block = f"\n\n当前冻结的通用规则：\n{rules.strip()}" if rules.strip() else ""
    assertion_block = hard_assertion_instruction(case)
    return (
        f"【致理结构调试·{phase}】\n{contract}{rules_block}\n\n题目：\n{case['question']}"
        f"{assertion_block}"
    )


def prompt_is_leak_free(prompt: str, case: dict[str, Any]) -> bool:
    forbidden = [str(case.get("reference") or ""), str(case.get("rule_target") or "")]
    forbidden.extend(str(item or "") for item in case.get("common_failures", []))
    return all(not value.strip() or value.strip() not in prompt for value in forbidden)


def observable_checks(case: dict[str, Any], answer: str) -> dict[str, Any]:
    compact = re.sub(r"\s+", "", answer)
    variant = str(case.get("variant") or "")
    has_inference = bool(re.search(r"因此|所以|由此|若|则|不能|取决于|需要|=|→|⇒", answer))
    has_boundary = bool(re.search(r"条件|前提|假设|不足|不确定|边界|仅当|不能仅凭|还需", answer))
    variant_ok = {
        "A": bool(re.search(r"推导|验证|构造|步骤|首先|由.*得|=|∇|Δ|反例|相关|因果|混杂|系统边界", answer)),
        "B": bool(re.search(r"不同|区别|相比|条件|不一定|仍", answer)),
        "C": bool(re.search(
            r"不足|缺少|缺失|尚缺|还需|无法(?:确定|判断|唯一(?:判定|确定))|"
            r"不能(?:判断|仅凭|推出)|无从(?:判断|确定)|需补充|最小缺失信息",
            answer,
        )),
        "D": bool(re.search(
            r"错误|不正确|不成立|不能(?:证明|推出)|命题为假|为假|反例|并非|"
            r"不是.*恒等式|非恒等式|只在.*成立|条件.*相反|结论.*过强|前提.*(?:错|过强)",
            answer,
        )),
        "E": bool(re.search(
            r"不足|不充分|不可靠|不确定|还需|样本|证据|无法证明|不能(?:证明|推出)|缺少.*条件",
            answer,
        )),
        "F": bool(re.search(
            r"错误|错在|错误在于|最小错误|问题在于|混淆|原论证.*(?:无效|不成立)|不能推出",
            answer,
        )) and bool(re.search(
            r"修正|正确|应当|应该|应改为|应改成|改写为|有效改写|修补|"
            r"补充.*条件|条件性结论|反例|严谨.*表述|准确.*表述|合理.*改写",
            answer,
        )),
        "V": has_inference,
        "H": has_inference,
    }.get(variant, has_inference)
    passed = len(compact) >= 80 and has_inference and variant_ok
    return {
        "passed": passed,
        "substantive": len(compact) >= 80,
        "has_inference": has_inference,
        "has_boundary": has_boundary,
        "variant_requirement": variant_ok,
        "semantic_score": None,
        "note": "本地检查只用于调试可观察结构，不代表答案语义正确。",
    }


def generation_error_text(row: dict[str, Any]) -> str:
    parts = [str(row.get("run_error") or "")]
    for item in row.get("agent_trace", []):
        if not isinstance(item, dict) or item.get("node") != "generate_answer":
            continue
        data = item.get("data", {})
        if isinstance(data, dict):
            parts.append(str(data.get("generation_error") or ""))
    return "\n".join(part for part in parts if part).strip()


def classify_execution(row: dict[str, Any]) -> dict[str, Any]:
    """Separate answer quality from endpoint/runtime availability.

    A recovered retry with a substantive answer remains scorable even when its
    trace records the first failed attempt. Empty and explicit fallback answers
    are never assigned a semantic score.
    """
    answer = str(row.get("answer") or "")
    compact = re.sub(r"\s+", "", answer)
    error = generation_error_text(row)
    explicit_fallback = any(marker in answer or marker in error for marker in GENERATION_FALLBACK_MARKERS)
    substantive = len(compact) >= 40 and not explicit_fallback
    if substantive:
        status = "completed"
    elif INFRASTRUCTURE_ERROR_RE.search(error):
        status = "infrastructure_failure"
    elif row.get("run_error"):
        status = "runtime_failure"
    else:
        status = "generation_failure"
    return {
        "status": status,
        "scorable": status == "completed",
        "substantive_answer": substantive,
        "error_class": (
            "external_endpoint_or_quota" if status == "infrastructure_failure"
            else "runtime" if status == "runtime_failure"
            else "generation_or_parser" if status == "generation_failure"
            else None
        ),
        "error_excerpt": error[:500] if status != "completed" else "",
        "note": (
            "仅 completed 样本进入语义与可观察分母；基础设施、运行时和空回退单独报告。"
        ),
    }


def qualification_audit(
    report_path: Path,
    *,
    benchmark: str,
    rules_sha256: str,
    tested_config: LLMConfig,
) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"passed": False, "reasons": [f"资格报告不可读：{type(exc).__name__}"]}
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    contract = summary.get("run_contract", {}) if isinstance(summary, dict) else {}
    if summary.get("phase") != "develop":
        reasons.append("资格报告不是 develop 阶段")
    if summary.get("benchmark") != benchmark:
        reasons.append("资格报告与当前题库 suite 不一致")
    if not summary.get("score_valid"):
        reasons.append("开发运行含非计分项或为空")
    if summary.get("scorable_cases") != summary.get("cases"):
        reasons.append("开发运行没有全部得到实质答案")
    if not contract.get("complete_dataset_run"):
        reasons.append("资格报告不是完整开发集运行")
    if contract.get("rules_sha256") != rules_sha256:
        reasons.append("规则版本与当前冻结规则不一致")
    if contract.get("tested_model") != tested_config.model:
        reasons.append("被测模型与当前配置不一致")
    if str(contract.get("tested_base_url") or "").rstrip("/") != tested_config.base_url.rstrip("/"):
        reasons.append("被测模型端点与当前配置不一致")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "qualification_report": str(report_path),
    }


def summarize(rows: list[dict[str, Any]], *, phase: str) -> dict[str, Any]:
    completed = [row for row in rows if row.get("execution", {}).get("scorable", True)]
    unscorable = [row for row in rows if not row.get("execution", {}).get("scorable", True)]
    failures = [row for row in completed if not row.get("observable_checks", {}).get("passed")]
    status_counts = Counter(
        str(row.get("execution", {}).get("status") or "completed") for row in rows
    )
    by_structure: dict[str, dict[str, int]] = defaultdict(lambda: {"cases": 0, "observable_pass": 0})
    for row in completed:
        item = by_structure[str(row.get("structure_id"))]
        item["cases"] += 1
        item["observable_pass"] += int(bool(row.get("observable_checks", {}).get("passed")))
    suites = Counter(str(row.get("suite") or "structural_debug") for row in rows)
    benchmark = suites.most_common(1)[0][0] if suites else "structural_debug"
    contracts = [
        row.get("run_contract")
        for row in rows
        if isinstance(row.get("run_contract"), dict)
    ]
    homogeneous_contract = not contracts or all(contract == contracts[0] for contract in contracts)
    run_contract = contracts[0] if contracts and homogeneous_contract else {}
    return {
        "benchmark": benchmark,
        "phase": phase,
        "cases": len(rows),
        "scorable_cases": len(completed),
        "unscorable_cases": len(unscorable),
        "execution_statuses": dict(sorted(status_counts.items())),
        "end_to_end_completion_rate": round(len(completed) / max(1, len(rows)), 4),
        "observable_pass": len(completed) - len(failures),
        "observable_pass_rate": (
            round((len(completed) - len(failures)) / len(completed), 4)
            if completed else None
        ),
        "failure_ids": [row.get("id") for row in failures],
        "unscorable_ids": [row.get("id") for row in unscorable],
        "score_valid": bool(rows) and not unscorable and homogeneous_contract,
        "semantic_score": None,
        "run_contract": run_contract,
        "homogeneous_run_contract": homogeneous_contract,
        "by_structure": dict(sorted(by_structure.items())),
        "generalization_claim": (
            "存在未完成运行，不能形成迁移结论" if unscorable and phase != "develop" else
            "同结构冻结迁移验证" if phase == "validate" else
            "作者可见新结构快照，不是严格盲测" if phase == "challenge" else
            "开发/调试结果，不代表泛化"
        ),
    }


def write_report(rows: list[dict[str, Any]], *, phase: str, report_dir: Path) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suite = str(rows[0].get("suite") or "structural-debug") if rows else "structural-debug"
    suite_slug = re.sub(r"[^A-Za-z0-9_-]+", "-", suite).strip("-") or "structural-debug"
    stem = f"{suite_slug}-{phase}-{stamp}"
    json_path = report_dir / f"{stem}.json"
    md_path = report_dir / f"{stem}.md"
    summary = summarize(rows, phase=phase)
    json_path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 致理结构调试 v2", "", f"- 阶段：{phase}", f"- 题数：{len(rows)}",
        f"- 可计分完成：{summary['scorable_cases']}/{len(rows)}",
        f"- 运行状态：{json.dumps(summary['execution_statuses'], ensure_ascii=False)}",
        f"- 可观察通过：{summary['observable_pass']}/{summary['scorable_cases']}",
        f"- 结论边界：{summary['generalization_claim']}",
        "- 语义正确率：未由本地规则评分。", "", "## 失败项", "",
        *(f"- {item}" for item in summary["failure_ids"]),
        "", "## 非计分运行项", "",
        *(f"- {item}" for item in summary["unscorable_ids"]), "", "## 逐题", "",
    ]
    for row in rows:
        lines.extend([
            f"### {row['id']} · {row.get('structure_group')}", "",
            f"- variant：{row.get('variant')}",
            f"- 执行状态：{row.get('execution', {}).get('status', 'completed')}",
            f"- 可观察检查：{'通过' if row.get('observable_checks', {}).get('passed') else '需复核'}", "",
            "**问题**", "", str(row.get("question") or ""), "", "**回答**", "",
            str(row.get("answer") or "（空）"), "", "<details><summary>调试参考（回答后才加载）</summary>", "",
            str(row.get("reference") or ""), "", *[f"- 常见失败：{item}" for item in row.get("common_failures", [])],
            f"- 原子评分：{json.dumps(row.get('scoring_rubric', {}), ensure_ascii=False)}" if row.get("scoring_rubric") else "",
            "", "</details>", "",
        ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Run the Zhili structural failure-driven debug pack")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--phase", choices=sorted(PHASE_SPLITS), default="develop")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="")
    parser.add_argument("--rules", type=Path, default=RULES_PATH)
    parser.add_argument(
        "--qualification-report",
        type=Path,
        help="完整且可计分的 develop 报告；validate/challenge 阶段必填",
    )
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    default_file = {
        "develop": PACK / "development_96.jsonl",
        "validate": PACK / "transfer_validation_32.jsonl",
        "challenge": PACK / "author_visible_challenge_12.jsonl",
    }[args.phase]
    dataset_path = (args.dataset or default_file).resolve()
    cases = load_cases(dataset_path)
    assert_phase_allowed(cases, args.phase)
    dataset_cases = len(cases)
    wanted = {item.strip() for item in args.ids.split(",") if item.strip()}
    if wanted:
        cases = [case for case in cases if str(case.get("id")) in wanted]
    if args.limit:
        cases = cases[:args.limit]
    if not args.run:
        audit = audit_pack(load_pack())
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        raise SystemExit(0 if audit["passed"] else 1)
    os.environ["GARDEN_DISABLE_NETWORK"] = "1"
    rules = args.rules.read_text(encoding="utf-8") if args.rules.exists() else ""
    rules_sha256 = hashlib.sha256(rules.encode("utf-8")).hexdigest()
    tested_config = llm_config()
    benchmark = str(cases[0].get("suite") or "structural_debug") if cases else "structural_debug"
    if args.phase != "develop":
        if not args.qualification_report:
            raise SystemExit("冻结阶段拒绝启动：必须提供 --qualification-report")
        gate = qualification_audit(
            args.qualification_report,
            benchmark=benchmark,
            rules_sha256=rules_sha256,
            tested_config=tested_config,
        )
        if not gate["passed"]:
            raise SystemExit(
                "冻结阶段资格检查失败：" + "；".join(str(item) for item in gate["reasons"])
            )
    run_contract = {
        "dataset": str(dataset_path),
        "dataset_cases": dataset_cases,
        "selected_cases": len(cases),
        "complete_dataset_run": not wanted and not args.limit and len(cases) == dataset_cases,
        "rules_path": str(args.rules.resolve()),
        "rules_sha256": rules_sha256,
        "tested_model": tested_config.model,
        "tested_base_url": tested_config.base_url,
        "qualification_report": (
            str(args.qualification_report.resolve()) if args.qualification_report else None
        ),
    }
    rows: list[dict[str, Any]] = []
    with temporary_store(DB_PATH) as store:
        for index, case in enumerate(cases, 1):
            prompt = build_agent_prompt(case, phase=args.phase, rules=rules)
            if not prompt_is_leak_free(prompt, case):
                raise RuntimeError(f"reference leakage detected for {case['id']}")
            runnable = {**public_case(case), "question": prompt}
            try:
                result = run_graph_case(store, runnable)
                answer = str(result.get("answer") or "")
                row = merge_case_result(case, result)
            except Exception as exc:
                answer = ""
                row = {**case, "answer": "", "run_error": f"{type(exc).__name__}: {str(exc)[:500]}"}
            row["run_contract"] = run_contract
            row["execution"] = classify_execution(row)
            row["observable_checks"] = observable_checks(case, answer)
            hard_result = run_hard_assertions(case, answer)
            if hard_result["applicable"]:
                row["hard_assertions"] = hard_result
            rows.append(row)
            write_report(rows, phase=args.phase, report_dir=args.report_dir)
            print(f"[{index}/{len(cases)}] {case['id']} {'pass' if row['observable_checks']['passed'] else 'review'}", flush=True)
    json_path, md_path = write_report(rows, phase=args.phase, report_dir=args.report_dir)
    print(json.dumps({"summary": summarize(rows, phase=args.phase), "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
