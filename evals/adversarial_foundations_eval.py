from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)
from openai import AsyncOpenAI

from core.config import llm_config
from core.reasoning_capability import classify_reasoning_task, evidence_route
from evals.adapter import load_cases
from evals.judge_config import judge_api_key, judge_base_url, judge_model, judge_request_options


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "evals" / "datasets" / "adversarial_foundations_v1.jsonl"
DEFAULT_REPORT_DIR = ROOT / "evals" / "reports"
PRIVATE_FIELDS = {"routing_target", "reference", "atomic_rubric", "common_failures", "symbolic_checks"}


def route_task(question: str) -> dict[str, Any]:
    profile = classify_reasoning_task(question)
    return {**evidence_route(question, profile=profile), "reasoning_profile": profile.get("key", "general")}


_MATH_BLOCKS = re.compile(
    r"\$\$(.+?)\$\$|\\\[(.+?)\\\]|\$(.+?)\$|\\\((.+?)\\\)|"
    r"\\begin\{(?:align\*?|aligned|equation\*?|gather\*?)\}(.+?)"
    r"\\end\{(?:align\*?|aligned|equation\*?|gather\*?)\}",
    re.S,
)

_RELATION_PATTERN = re.compile(
    r"\\(?:geq?|leq?|approx)(?![A-Za-z])|>=|<=|≥|≤|≈|(?<![<>!])=(?!=)"
)


def _canonical_relation(value: str) -> str:
    relation = value.strip()
    if relation in {r"\ge", r"\geq", "≥", ">="}:
        return ">="
    if relation in {r"\le", r"\leq", "≤", "<="}:
        return "<="
    if relation in {r"\approx", "≈", "~="}:
        return "~="
    return "="


def extract_math_expressions(markdown: str) -> list[str]:
    """Extract LaTeX/Markdown math plus equation-like plaintext lines."""
    return [item["raw"] for item in _extract_math_records(markdown)]


def _extract_math_records(markdown: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for match in _MATH_BLOCKS.finditer(markdown):
        block = next(group for group in match.groups() if group is not None).strip()
        rows = re.split(r"\\\\(?:\[[^\]]*\])?", block)
        context = markdown[max(0, match.start() - 36):min(len(markdown), match.end() + 36)]
        for row_index, row in enumerate(rows):
            if row.strip():
                found.append({
                    "raw": row.strip(), "position": match.start() + row_index,
                    "context": context,
                    "negated": bool(re.search(r"(?:错误|不正确|并非|不应|不能写成|不等于|\\ne|≠)", context)),
                })
    offset = 0
    for line in markdown.splitlines(keepends=True):
        clean = re.sub(r"^[#>*\-\s]+", "", line).strip()
        clean = re.sub(r"(?<!\\)[*_]{1,3}$", "", clean).strip()
        if _RELATION_PATTERN.search(clean) and len(clean) <= 500 and not _MATH_BLOCKS.search(line):
            for clause_match in re.finditer(r"[^，。；;：:]+", clean):
                clause = clause_match.group(0).strip()
                relations = list(_RELATION_PATTERN.finditer(clause))
                if not relations:
                    continue
                first_left = clause[:relations[0].start()]
                root_lhs = re.search(
                    r"([A-Za-zΔδρλξκμεωΘβα\\][A-Za-z0-9_₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹⁽⁾ΔδρλξκμεωΘβα\\]*(?:'|\^\*)*(?:\([^=，。；;]{0,40}\))?)\s*$",
                    first_left,
                )
                for relation_index, relation_match in enumerate(relations):
                    left = clause[:relation_match.start()]
                    right_end = relations[relation_index + 1].start() if relation_index + 1 < len(relations) else len(clause)
                    right = clause[relation_match.end():right_end]
                lhs = re.search(
                    r"([A-Za-zΔδρλξκμεωΘβα\\][A-Za-z0-9_₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹⁽⁾ΔδρλξκμεωΘβα\\]*(?:'|\^\*)*(?:\([^=，。；;]{0,40}\))?)\s*$",
                    left,
                )
                rhs = re.match(r"\s*([A-Za-z0-9_₀₁₂₃₄₅₆₇₈₉ΔδρλξκμεωΘβαħℏ\\{}^*+\-/().\s²³⁴⁵⁶⁷⁸⁹⁰ⁿ⁽⁾−]+)", right)
                effective_lhs = root_lhs or lhs
                if not effective_lhs or not rhs or not rhs.group(1).strip():
                    continue
                raw = (
                    f"{effective_lhs.group(1).strip()}"
                    f"{relation_match.group(0)}{rhs.group(1).strip()}"
                )
                found.append({
                    "raw": raw, "position": offset + clause_match.start() + relation_match.start(),
                    "context": line.strip(),
                    "negated": bool(re.search(r"(?:错误|不正确|并非|不应|不能写成|不等于|\\ne|≠)", line)),
                })
        offset += len(line)
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for item in sorted(found, key=lambda row: int(row["position"])):
        key = (str(item["raw"]), int(item["position"]))
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _replace_latex_command(text: str, command: str, arity: int, formatter: Any) -> str:
    """Replace a LaTeX command with balanced braced arguments, including nesting."""
    cursor = 0
    while True:
        start = text.find(command, cursor)
        if start < 0:
            return text
        index = start + len(command)
        args: list[str] = []
        valid = True
        for _ in range(arity):
            while index < len(text) and text[index].isspace():
                index += 1
            if index >= len(text) or text[index] != "{":
                valid = False
                break
            depth, end = 0, index
            while end < len(text):
                if text[end] == "{":
                    depth += 1
                elif text[end] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                end += 1
            if depth != 0:
                valid = False
                break
            args.append(text[index + 1:end])
            index = end + 1
        if not valid:
            cursor = start + len(command)
            continue
        rendered = formatter(*args)
        text = text[:start] + rendered + text[index:]
        cursor = max(0, start - 1)


def _latex_to_sympy_text(value: str) -> str:
    text = value.strip().replace("−", "-").replace("×", "*").replace("ℏ", "hbar").replace("ħ", "hbar")
    # Chemistry convention: [S], [A], ... denote concentrations, not array access.
    text = re.sub(r"\[([A-Za-z][A-Za-z0-9_]*)\]", r"\1", text)
    # Pure presentation/quantifier suffixes do not change the equation value.
    text = re.sub(r"\(\s*\\forall\s+[A-Za-z][A-Za-z0-9_]*\s*\)\s*$", "", text)
    text = re.sub(r"\\(?:quad|qquad)(?![A-Za-z])", "", text)
    text = text.replace(r"\Delta\lambda", "Delta_lambda").replace("Δλ", "Delta_lambda")
    text = re.sub(
        r"E₀⁽([⁰¹²³⁴⁵⁶⁷⁸⁹]+)⁾",
        lambda match: "E_0_" + match.group(1).translate(str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")),
        text,
    )
    text = text.translate(str.maketrans({"₀": "_0", "₁": "_1", "₂": "_2", "₃": "_3", "₄": "_4", "₅": "_5", "₆": "_6", "₇": "_7", "₈": "_8", "₉": "_9"}))
    text = re.sub(r"(?<=[A-Za-z0-9])__+(?=[A-Za-z0-9])", "_", text)
    text = text.translate(str.maketrans({"²": "**2", "³": "**3", "⁴": "**4", "⁵": "**5", "⁶": "**6", "⁷": "**7", "⁸": "**8", "⁹": "**9", "⁰": "**0"}))
    text = text.replace("ⁿ", "**n")
    text = re.sub(r"\\geq?(?![A-Za-z])", ">=", text)
    text = re.sub(r"\\leq?(?![A-Za-z])", "<=", text)
    text = re.sub(r"\\approx(?![A-Za-z])", "~=", text)
    text = text.replace("≥", ">=").replace("≤", "<=").replace("≈", "~=")
    for unicode_name, plain_name in {
        "Δ": "Delta", "δ": "delta", "ρ": "rho", "λ": "lambda_", "ξ": "xi",
        "κ": "kappa", "μ": "mu", "ε": "epsilon", "ω": "omega", "Θ": "Theta",
        "β": "beta", "α": "alpha", "θ": "theta",
    }.items():
        text = text.replace(unicode_name, plain_name)
    text = re.sub(r"\\(?:begin|end)\{[^{}]+\}", "", text)
    text = re.sub(r"\\(?:label|tag)\s*\{[^{}]*\}", "", text)
    text = re.sub(r"\\(?:big|Big|bigg|Bigg)[lrm]?", "", text)
    text = re.sub(
        r"\\(?:mathrm|text)\{(?:s|ms|kg|g|m|cm|mm|J|kJ|K|Pa|bar|V|A|C|F|H|W|Hz|mol)\}",
        "", text,
    )
    text = text.replace("\\left", "").replace("\\right", "").replace("&", "")
    text = _replace_latex_command(text, r"\boxed", 1, lambda body: body)
    text = _replace_latex_command(text, r"\operatorname", 1, lambda body: body)
    text = _replace_latex_command(text, r"\mathrm", 1, lambda body: body)
    text = _replace_latex_command(text, r"\text", 1, lambda body: body)
    text = _replace_latex_command(text, r"\frac", 2, lambda top, bottom: f"(({top})/({bottom}))")
    text = _replace_latex_command(text, r"\dfrac", 2, lambda top, bottom: f"(({top})/({bottom}))")
    text = _replace_latex_command(text, r"\tfrac", 2, lambda top, bottom: f"(({top})/({bottom}))")
    text = _replace_latex_command(text, r"\sqrt", 1, lambda body: f"sqrt({body})")
    greek = {
        "alpha": "alpha", "beta": "beta", "gamma": "gamma", "Delta": "Delta",
        "epsilon": "epsilon", "varepsilon": "epsilon", "lambda": "lambda_",
        "mu": "mu", "omega": "omega", "rho": "rho", "kappa": "kappa",
        "Theta": "Theta", "theta": "theta", "xi": "xi", "phi": "phi", "psi": "psi",
    }
    for latex_name, plain_name in greek.items():
        base_name = plain_name.rstrip("_")
        text = re.sub(
            rf"\\{latex_name}_\{{?([A-Za-z0-9]+)\}}?",
            lambda match: f"{base_name}_{match.group(1)}",
            text,
        )
        text = re.sub(rf"\\{latex_name}(?![A-Za-z])", f"({plain_name})", text)
    text = (
        text.replace("\\ln", "log").replace("\\log", "log")
        .replace("\\sin", "sin").replace("\\cos", "cos").replace("\\tan", "tan")
        .replace("\\exp", "exp").replace("\\sqrt", "sqrt")
        .replace("\\cdot", "*").replace("\\times", "*")
    )
    text = re.sub(r"\bln(?=\s*\()", "log", text)
    text = re.sub(
        r"\b(sqrt|log|sin|cos|tan|exp)(?!\s*\()(\d+(?:\.\d+)?|[A-Za-z_][A-Za-z0-9_]*)",
        r"\1(\2)", text,
    )
    text = re.sub(r"(?<=[A-Za-z0-9_)])(sqrt|log|sin|cos|tan|exp)\(", r"*\1(", text)
    text = re.sub(r"\\lvert\s*([A-Za-z][A-Za-z0-9_]*)\s*\\rvert", r"Abs(\1)", text)
    text = re.sub(r"\|\s*([A-Za-z][A-Za-z0-9_]*)\s*\|", r"Abs(\1)", text)
    text = re.sub(r"\\[,;!:\s]", "", text)
    text = re.sub(r"([A-Za-z][A-Za-z0-9_]*)\^\*", r"\1_star", text)
    text = re.sub(r"([A-Za-z][A-Za-z0-9_]*(?:_[A-Za-z0-9_]+)?)\^\{\(([A-Za-z0-9]+)\)\}", r"\1_\2", text)
    text = re.sub(r"([A-Za-z][A-Za-z0-9_]*)'", r"\1prime", text)
    text = re.sub(r"([A-Za-z]+)_\{([^{}]+)\}", r"\1_\2", text)
    text = re.sub(r"([A-Za-z]+)_([A-Za-z0-9]+)", r"\1_\2", text)
    text = re.sub(r"\^\{([^{}]+)\}", r"**(\1)", text)
    text = re.sub(r"\^([A-Za-z0-9_.+-]+)", r"**\1", text)
    text = text.replace("{", "(").replace("}", ")").replace("\\", "")
    text = re.sub(r"(?<=[A-Za-z0-9_)])\s+(?=[A-Za-z_(])", "*", text)
    text = re.sub(r"\s+", "", text)
    text = text.strip("，。；;:：")
    if not re.fullmatch(r"[A-Za-z0-9_+*/().,=<>~\-]+", text):
        raise ValueError("unsupported_or_unsafe_math_syntax")
    return text


def _split_relation_chain(normalized: str) -> tuple[str, list[dict[str, str]]]:
    matches = list(re.finditer(r">=|<=|~=|(?<![<>!])=(?!=)", normalized))
    if not matches:
        return "", []
    lhs = normalized[:matches[0].start()].strip()
    sides: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        rhs = normalized[match.end():end].strip()
        if rhs:
            sides.append({"relation": _canonical_relation(match.group(0)), "rhs": rhs})
    return lhs, sides


def _parse_sympy_expression(
    value: str,
    declared_symbols: list[str] | None = None,
    assumptions: dict[str, dict[str, bool]] | None = None,
) -> sp.Expr:
    normalized = _latex_to_sympy_text(value)
    if "=" in normalized:
        raise ValueError("expected_expression_not_equation")
    declared = [str(item) for item in (declared_symbols or []) if str(item)]
    for declared_name in declared:
        if declared_name.endswith("_abs"):
            normalized = normalized.replace(f"Abs({declared_name[:-4]})", declared_name)

    def split_declared(match: re.Match[str]) -> str:
        token = match.group(0)
        if token in declared or token in {"sqrt", "log", "exp", "sin", "cos", "tan", "Abs", "integrate", "tr"}:
            return token
        parts: list[str] = []
        cursor = 0
        ordered = sorted(declared, key=len, reverse=True)
        while cursor < len(token):
            part = next((name for name in ordered if token.startswith(name, cursor)), None)
            if part is None:
                return token
            parts.append(part)
            cursor += len(part)
        return "*".join(parts) if len(parts) >= 2 else token

    normalized = re.sub(r"[A-Za-z_][A-Za-z0-9_]*", split_declared, normalized)
    symbol_names = set(declared) | set(re.findall(r"[A-Za-z_]\w*", normalized))
    reserved = {"sqrt", "log", "exp", "sin", "cos", "tan", "Abs", "integrate", "tr"}
    names = {
        name: sp.Symbol(name, **dict((assumptions or {}).get(name, {})))
        for name in symbol_names
        if name not in reserved
    }
    locals_map: dict[str, Any] = {
        **names,
        "sqrt": sp.sqrt, "log": sp.log, "exp": sp.exp,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "Abs": sp.Abs,
        "integrate": sp.integrate, "tr": sp.Function("tr"),
    }
    return parse_expr(
        normalized,
        local_dict=locals_map,
        transformations=standard_transformations + (convert_xor, implicit_multiplication_application),
        evaluate=True,
    )


def answer_symbolic_grounding(case: dict[str, Any], answer: str) -> dict[str, Any]:
    records = _extract_math_records(answer)
    extracted = [item["raw"] for item in records]
    equations: list[dict[str, Any]] = []
    parse_errors: list[dict[str, str]] = []
    for record in records:
        raw = str(record["raw"])
        cleaned = re.sub(r"\\(?:tag|label)\s*\{[^{}]*\}", "", raw)
        try:
            normalized_equation = _latex_to_sympy_text(cleaned)
        except Exception as exc:
            parse_errors.append({"raw": raw[:240], "side": "", "error": str(exc)[:160]})
            continue
        lhs, relation_sides = _split_relation_chain(normalized_equation)
        if lhs and relation_sides:
            equations.append({
                **record, "normalized": normalized_equation, "lhs": lhs,
                "relation_sides": relation_sides,
                "rhs_sides": [item["rhs"] for item in relation_sides],
            })
    checks = []
    for spec in case.get("symbolic_checks", []):
        target_values = [str(spec.get("target_lhs") or ""), *[str(item) for item in spec.get("target_aliases", [])]]
        target_values = [item for item in target_values if item.strip()]
        if not target_values:
            checks.append({
                "id": spec.get("id"), "status": "INVALID_SPEC", "expected": str(spec.get("rhs")),
                "reason": "symbolic check 缺少 target_lhs，禁止在全篇候选池中碰撞参考值",
            })
            continue
        normalized_targets = set()
        normalized_primary = ""
        for target_index, target in enumerate(target_values):
            try:
                normalized_target = _latex_to_sympy_text(target).strip("()")
                normalized_targets.add(normalized_target)
                if target_index == 0:
                    normalized_primary = normalized_target
            except Exception as exc:
                parse_errors.append({"raw": target[:240], "side": "target_lhs", "error": str(exc)[:160]})
        alias_context_pattern = str(spec.get("alias_context_pattern") or "")
        target_equations = [
            equation for equation in equations
            if str(equation["lhs"]).strip("()") in normalized_targets
            and (
                str(equation["lhs"]).strip("()") == normalized_primary
                or not alias_context_pattern
                or re.search(alias_context_pattern, str(equation.get("context") or ""), re.I)
            )
        ]
        accepted_relations = {
            _canonical_relation(str(item))
            for item in (spec.get("accepted_relations") or [spec.get("relation", "=")])
        }
        nonnegated_target_equations = [
            equation for equation in target_equations if not equation.get("negated")
        ]
        usable_equations = [
            equation for equation in nonnegated_target_equations
            if any(
                side.get("relation") in accepted_relations
                for side in equation.get("relation_sides", [])
            )
        ]
        declared_symbols = [str(item) for item in spec.get("symbols", [])]

        def selection_key(indexed: tuple[int, dict[str, Any]]) -> tuple[int, int]:
            index, equation = indexed
            rhs_text = " ".join(str(item.get("rhs") or "") for item in equation.get("relation_sides", []))
            tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", rhs_text))
            # Prefer the general formula carrying the declared variables over a
            # later conditional specialization (for example n=1).  For equal
            # coverage, the last occurrence remains the correction winner.
            return sum(symbol in tokens for symbol in declared_symbols), index

        selected = max(enumerate(usable_equations), key=selection_key)[1] if usable_equations else None
        assumptions = dict(spec.get("assumptions") or {})
        expected = _parse_sympy_expression(
            str(spec["rhs"]), list(spec.get("symbols", [])), assumptions,
        )
        candidates: list[dict[str, Any]] = []
        matches: list[dict[str, Any]] = []
        if selected:
            for relation_side in selected["relation_sides"]:
                side = relation_side["rhs"]
                relation = relation_side["relation"]
                if relation not in accepted_relations:
                    candidates.append({
                        "raw": str(selected["raw"])[:240], "side": side[:160],
                        "relation": relation,
                        "residual": f"RELATION_MISMATCH: expected {sorted(accepted_relations)}",
                    })
                    continue
                try:
                    candidate = _parse_sympy_expression(side, list(spec.get("symbols", [])), assumptions)
                    residual = sp.simplify(candidate - expected)
                except Exception as exc:
                    candidates.append({"raw": str(selected["raw"])[:240], "side": side[:160], "residual": f"ERROR: {exc}"})
                    continue
                candidate_record = {
                    "raw": str(selected["raw"])[:240], "side": side[:160],
                    "relation": relation,
                    "normalized": str(candidate), "residual": str(residual),
                    "position": selected["position"],
                }
                candidates.append(candidate_record)
                if residual == 0:
                    matches.append(candidate_record)
        if matches:
            status = "PASS"
        elif selected and candidates:
            status = "MISMATCH"
        elif nonnegated_target_equations:
            status = "MISMATCH"
        elif target_equations:
            status = "NEGATED_OR_EXTRACTION_FAILED"
        elif equations:
            status = "TARGET_NOT_FOUND"
        else:
            status = "EXTRACTION_FAILED"
        checks.append({
            "id": spec.get("id"), "status": status,
            "target_lhs": target_values[0], "target_aliases": target_values[1:],
            "alias_context_pattern": alias_context_pattern,
            "accepted_relations": sorted(accepted_relations),
            "matched_expressions": matches[:3], "expected": str(expected),
            "candidate_residuals": candidates[:12],
            "target_equations_found": len(target_equations),
            "selection_policy": "declared_symbol_coverage_then_last",
            "selected_equation": str(selected["raw"])[:240] if selected else "",
        })
    return {
        "extracted": extracted,
        "parse_errors": parse_errors[:12],
        "checks": checks,
        "applicable": bool(checks),
        "passed": all(c["status"] == "PASS" for c in checks) if checks else None,
    }


class EventLog:
    def __init__(self, run_id: str):
        self.run_id, self.events = run_id, []

    def add(self, case_id: str, event: str, **data: Any) -> None:
        self.events.append({"run_id": self.run_id, "case_id": case_id, "seq": len(self.events)+1,
                            "timestamp": datetime.now().astimezone().isoformat(), "event": event, "data": data})


def public_payload(case: dict[str, Any]) -> dict[str, str]:
    payload = {"question": str(case["question"])}
    if case.get("reading_brief"):
        payload["reading_brief"] = str(case["reading_brief"])
    assert not (set(payload) & PRIVATE_FIELDS)
    return payload


def normalize_rubric(case: dict[str, Any]) -> list[dict[str, Any]]:
    rubric = case.get("atomic_rubric") or {}
    points: list[dict[str, Any]] = []
    labels = {
        "required_claims": "必要结论", "experimental_controls": "实验对照",
        "readouts": "直接读出", "falsification_criteria": "证伪标准",
        "excluded_explanations": "替代解释排除",
    }
    for field, label in labels.items():
        for item in rubric.get(field, []) if isinstance(rubric, dict) else []:
            if str(item).strip():
                points.append({"id": f"R{len(points)+1}", "dimension": label, "criterion": str(item)})
    if not 3 <= len(points) <= 5:
        raise ValueError(f"{case.get('id')}: atomic_rubric 必须前置拆成 3-5 个非空得分点，实际 {len(points)}")
    weight = round(1 / len(points), 6)
    for point in points:
        point["weight"] = weight
    points[-1]["weight"] = round(1 - sum(p["weight"] for p in points[:-1]), 6)
    return points


def symbolic_grounding(case: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for check in case.get("symbolic_checks", []):
        symbols = {name: sp.Symbol(name, positive=True) for name in check.get("symbols", [])}
        lhs = sp.sympify(check["lhs"], locals=symbols)
        rhs = sp.sympify(check["rhs"], locals=symbols)
        residual = sp.simplify(lhs - rhs)
        results.append({
            "id": check.get("id", f"S{len(results)+1}"),
            "lhs": check["lhs"], "rhs": check["rhs"],
            "residual": str(residual), "passed": residual == 0,
        })
    return results


async def _answer_glm(case: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    config = llm_config()
    if not config.enabled or not config.model.lower().startswith("glm"):
        raise RuntimeError("被测模型必须配置为 GLM，当前主模型未启用或不是 glm-*。")
    client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url, timeout=120, max_retries=1)
    payload = public_payload(case)
    system = (
        "你是基础学科被测模型。只依据收到的 question 与可选 reading_brief 回答。"
        "闭环证明和计算题直接推导，不以缺少外部资料为由拒答；只有题目明确要求当前事实或文献时才说明检索需求。"
        "给出可复核的公式、计算、条件与明确结论。"
    )
    started = time.perf_counter()
    response = await client.chat.completions.create(
        model=config.model, messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        temperature=0.2, max_tokens=2200,
    )
    answer = str(response.choices[0].message.content or "").strip()
    return answer, {"model": config.model, "latency_ms": round((time.perf_counter()-started)*1000, 2), "visible_fields": sorted(payload)}


async def _deepseek_judge(case: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    model = judge_model()
    if not model.startswith("deepseek-v4"):
        raise RuntimeError(f"辅助裁判必须是 deepseek-v4*，当前为 {model}")
    rubric = row["scoring_rubric"]
    data = {"question": case["question"], "reference": case["reference"], "answer": row["answer"], "rubric": rubric,
            "router": row["router"], "symbolic_grounding": row["symbolic_grounding"], "common_failures": case.get("common_failures", [])}
    system = (
        "你是异构辅助裁判，职责是红队找错，不是替答案补步骤。逐条 rubric 输出 met(true/false)、evidence、reason，"
        "并检查错误检索/拒答、符号校验冲突和极性偏斜。只输出 JSON：rubric_results 数组、score(0-100)、"
        "verdict(pass/warn/fail)、fatal_errors 数组、router_assessment、suggestion。"
    )
    client = AsyncOpenAI(api_key=judge_api_key(model), base_url=judge_base_url(model), timeout=120, max_retries=1)
    started = time.perf_counter()
    response = await client.chat.completions.create(
        model=model, response_format={"type": "json_object"}, max_tokens=1800,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(data, ensure_ascii=False)}],
        **judge_request_options(model),
    )
    result = json.loads(str(response.choices[0].message.content or "{}"))
    result.update({"model": model, "role": "auxiliary_adversarial_judge", "latency_ms": round((time.perf_counter()-started)*1000, 2)})
    return result


async def run(cases: list[dict[str, Any]], *, skip_models: bool = False, event_log: EventLog | None = None) -> list[dict[str, Any]]:
    rows = []
    for index, case in enumerate(cases, 1):
        router = route_task(str(case["question"]))
        if event_log: event_log.add(case["id"], "router_decided", router=router)
        rubric = normalize_rubric(case)
        rubric_hash = hashlib.sha256(json.dumps(rubric, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if event_log: event_log.add(case["id"], "rubric_frozen", rubric_hash=rubric_hash, rubric=rubric)
        symbolic = symbolic_grounding(case)
        row: dict[str, Any] = {
            "id": case["id"], "category": case["category"], "question": case["question"],
            "expected_routing_target": case["routing_target"], "router": router,
            "routing_correct": router["routing_target"] == case["routing_target"],
            "scoring_rubric": rubric, "symbolic_grounding": symbolic,
            "blind_isolation": {"passed": True, "model_visible_fields": sorted(public_payload(case))},
        }
        if not skip_models:
            try:
                if event_log: event_log.add(case["id"], "glm_requested", visible_fields=sorted(public_payload(case)))
                row["answer"], row["tested_model"] = await _answer_glm(case)
                row["answer_symbolic_grounding"] = answer_symbolic_grounding(case, row["answer"])
                if event_log: event_log.add(case["id"], "symbolic_checked", result=row["answer_symbolic_grounding"])
                row["auxiliary_judge"] = await _deepseek_judge(case, row)
                if event_log: event_log.add(case["id"], "judge_completed", judge=row["auxiliary_judge"])
            except Exception as exc:
                row["runtime_error"] = f"{exc.__class__.__name__}: {exc}"
                if event_log: event_log.add(case["id"], "component_failed", error=row["runtime_error"])
        rows.append(row)
        print(f"[{index}/{len(cases)}] {case['id']} route={'PASS' if row['routing_correct'] else 'FAIL'} models={'SKIP' if skip_models else 'DONE'}", flush=True)
    return rows


def write_report(rows: list[dict[str, Any]], report_dir: Path, events: list[dict[str, Any]] | None = None) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = report_dir / f"adversarial-foundations-{stamp}.json"
    md_path = report_dir / f"adversarial-foundations-{stamp}.md"
    summary = {"cases": len(rows), "routing_correct": sum(r["routing_correct"] for r in rows),
               "rubric_non_null": sum(bool(r["scoring_rubric"]) for r in rows),
               "symbolic_pass": sum(all(c["passed"] for c in r["symbolic_grounding"]) for r in rows),
               "model_errors": sum(bool(r.get("runtime_error")) for r in rows)}
    json_path.write_text(json.dumps({"summary": summary, "events": events or [], "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 对抗性基础学科评测调试日志", "", f"- 样本：{len(rows)}", f"- 路由正确：{summary['routing_correct']}/{len(rows)}",
             f"- Rubric 非空：{summary['rubric_non_null']}/{len(rows)}", f"- 符号校验通过：{summary['symbolic_pass']}/{len(rows)}", ""]
    for row in rows:
        lines += [f"## {row['id']} · {row['category']}", "", f"- 路由：{row['router']['routing_target']}（{row['router']['confidence']:.1%}）",
                  f"- 题面隔离：{row['blind_isolation']}", f"- Symbolic：{json.dumps(row['symbolic_grounding'], ensure_ascii=False)}",
                  f"- 运行错误：{row.get('runtime_error', '无')}", "", "### GLM 回答", "", str(row.get("answer") or "（未运行）"), "",
                  "### DeepSeek-V4 辅助裁判", "", "```json", json.dumps(row.get("auxiliary_judge", {}), ensure_ascii=False, indent=2), "```", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    cases = load_cases(args.dataset)
    if args.limit:
        cases = cases[:args.limit]
    event_log = EventLog(datetime.now().strftime("run-%Y%m%d-%H%M%S"))
    rows = asyncio.run(run(cases, skip_models=args.validate_only, event_log=event_log))
    paths = write_report(rows, args.report_dir, event_log.events)
    print(json.dumps({"reports": [str(p) for p in paths]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
