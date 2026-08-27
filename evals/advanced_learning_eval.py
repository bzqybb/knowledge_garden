from __future__ import annotations

import json
import math
import re
from typing import Any


FRONTIER_SUITE = "zhili_frontier_guided_reading_v1"

FRONTIER_JUDGE_PROMPTS = {
    "A": (
        "你是前沿领读概念解析裁判。只依据题目、reading_brief、source_claims、rubric 与实际回答评分。"
        "重点检查概念定位、材料主张与回答推断的区分、推理链和适用边界；"
        "不得要求实验设计题才需要的对照组清单。"
    ),
    "E": (
        "你是前沿领读证据与边界裁判。只依据题目、reading_brief、source_claims、rubric 与实际回答评分。"
        "重点检查比较对象、证据成熟度、数据切分或覆盖范围、校准/标度/部署接口、"
        "失败边界及结论强度；不得强行要求与任务无关的实验操作细节。"
    ),
    "B": (
        "你是前沿领读实验设计裁判。只依据题目、reading_brief、source_claims、rubric 与实际回答评分。"
        "重点检查可检验假设、对照或基线、干预、主要读出、预先失败判据和可区分的替代解释；"
        "不得因没有写成概念综述而扣分。"
    ),
}


def frontier_agent_payload(case: dict[str, Any]) -> dict[str, Any]:
    """Fields visible to the tested Agent for frontier guided-reading cases."""
    return {
        "question": str(case.get("question") or ""),
        "reading_brief": str(case.get("reading_brief") or ""),
    }


def frontier_judge_task_type(case: dict[str, Any]) -> str:
    """Route a frontier task to one of three scoring contracts."""
    task = str(case.get("question") or "").splitlines()[-1]
    structure = str(case.get("reasoning_structure_id") or "")
    experiment_markers = (
        "设计一个", "设计一种", "实验", "干预", "阴性对照", "正交验证",
        "压力测试", "消融", "新增波段", "盲化基线", "验证链",
    )
    benchmark_markers = (
        "基准", "部署", "外推", "泛化", "校准", "业务价值", "扩展区间",
        "不矛盾", "为什么", "失败的机制链", "联合失败判据",
    )
    experiment_structures = {
        "FR01_BLINDED_BASELINE_AUDIT",
        "FR02_INTERVENTION_PROXY_TEST",
        "FR03_ORTHOGONAL_RECOVERY",
        "FR05_ABLATION_OOD_STRESS",
        "FR10_RARE_FAILURE_STRESS",
    }
    benchmark_structures = {
        "FR04_SCALING_EXTRAPOLATION",
        "FR05_BENCHMARK_DEPLOYMENT_GAP",
        "FR06_CALIBRATION_DECISION_UTILITY",
        "FR06_SOURCE_MATURITY_LEAKAGE",
        "FR10_PHYSICAL_LIMITS_OOD",
    }
    if structure in experiment_structures or any(marker in task for marker in experiment_markers):
        return "B"
    if structure in benchmark_structures or any(marker in task for marker in benchmark_markers):
        return "E"
    return "A"


def frontier_judge_payload(case: dict[str, Any], answer: str | None = None) -> dict[str, Any]:
    """Post-answer whitelist; curator-written inference fields are intentionally absent."""
    return {
        "case_id": str(case.get("id") or ""),
        "question": str(case.get("question") or ""),
        "reading_brief": str(case.get("reading_brief") or ""),
        "source_claims": [
            str(item) for item in case.get("source_claims", []) if str(item).strip()
        ],
        "rubric": dict(case.get("scoring_rubric") or {}),
        "answer": str(case.get("answer") if answer is None else answer or ""),
    }


def frontier_judge_prompt(case: dict[str, Any]) -> str:
    return FRONTIER_JUDGE_PROMPTS[frontier_judge_task_type(case)]


def hard_assertion_instruction(case: dict[str, Any]) -> str:
    spec = case.get("hard_assertion")
    if not isinstance(spec, dict):
        return ""
    schema = json.dumps(spec.get("schema", {}), ensure_ascii=False, separators=(",", ":"))
    return (
        "\n本题含确定性校验。请在自然语言推导后追加且只追加一个机器可读块："
        f"\n<hard_assertion>{schema}</hard_assertion>\n"
        "把 schema 中的占位值替换为你的实际计算或构造；该块将由代码校验，不能用文字解释替代。"
    )


def _extract_hard_payload(answer: str) -> dict[str, Any] | None:
    match = re.search(r"<hard_assertion>\s*(\{.*?\})\s*</hard_assertion>", answer, re.S)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _complex_cell(value: Any) -> complex:
    if isinstance(value, (int, float)):
        return complex(float(value), 0.0)
    if isinstance(value, list) and len(value) == 2:
        return complex(float(value[0]), float(value[1]))
    raise ValueError("matrix cell must be a number or [real, imag]")


def _check_hc10(payload: dict[str, Any]) -> dict[str, bool]:
    raw = payload.get("rho")
    if not (
        isinstance(raw, list) and len(raw) == 2
        and all(isinstance(row, list) and len(row) == 2 for row in raw)
    ):
        raise ValueError("rho must be a 2x2 matrix")
    rho = [[_complex_cell(cell) for cell in row] for row in raw]
    tol = 1e-8
    hermitian = (
        abs(rho[0][0].imag) <= tol
        and abs(rho[1][1].imag) <= tol
        and abs(rho[0][1] - rho[1][0].conjugate()) <= tol
    )
    trace = rho[0][0] + rho[1][1]
    trace_one = abs(trace - 1) <= tol
    if hermitian:
        a, d, b = rho[0][0].real, rho[1][1].real, rho[0][1]
        discriminant = max(0.0, (a - d) ** 2 + 4 * abs(b) ** 2)
        eig_min = (a + d - math.sqrt(discriminant)) / 2
        positive_semidefinite = eig_min >= -tol
    else:
        positive_semidefinite = False
    expected = [
        [complex(0.7, 0), complex(0.3, 0.1)],
        [complex(0.3, -0.1), complex(0.3, 0)],
    ]
    tomography_matches = all(
        abs(rho[i][j] - expected[i][j]) <= tol for i in range(2) for j in range(2)
    )
    return {
        "hermitian": hermitian,
        "trace_one": trace_one,
        "positive_semidefinite": positive_semidefinite,
        "tomography_matches_measurements": tomography_matches,
    }


def _check_hc02(payload: dict[str, Any]) -> dict[str, bool]:
    curves = payload.get("curves")
    claimed_lengths = payload.get("claimed_lengths")
    claimed_winding = payload.get("claimed_winding_numbers")
    if not isinstance(curves, list) or len(curves) != 2:
        raise ValueError("curves must contain two circles")
    if not isinstance(claimed_lengths, list) or len(claimed_lengths) != 2:
        raise ValueError("claimed_lengths must contain two values")
    if not isinstance(claimed_winding, list) or len(claimed_winding) != 2:
        raise ValueError("claimed_winding_numbers must contain two values")
    computed_lengths: list[float] = []
    computed_winding: list[int] = []
    avoids_origin = True
    valid_circles = True
    tol = 1e-8
    for curve in curves:
        if not isinstance(curve, dict):
            raise ValueError("each curve must be an object")
        center = curve.get("center")
        radius = float(curve.get("radius"))
        orientation = str(curve.get("orientation") or "ccw").lower()
        if not isinstance(center, list) or len(center) != 2 or radius <= 0:
            valid_circles = False
            continue
        cx, cy = float(center[0]), float(center[1])
        distance = math.hypot(cx, cy)
        avoids_origin = avoids_origin and abs(distance - radius) > tol
        computed_lengths.append(2 * math.pi * radius)
        if distance < radius:
            computed_winding.append(-1 if orientation == "cw" else 1)
        else:
            computed_winding.append(0)
    equal_required_length = (
        len(computed_lengths) == 2
        and all(abs(value - 2 * math.pi) <= tol for value in computed_lengths)
        and all(abs(float(claimed_lengths[i]) - computed_lengths[i]) <= tol for i in range(2))
    )
    winding_claims_match = (
        len(computed_winding) == 2
        and all(int(claimed_winding[i]) == computed_winding[i] for i in range(2))
    )
    different_homotopy_classes = len(set(computed_winding)) == 2
    return {
        "valid_simple_circles": valid_circles,
        "both_avoid_origin": avoids_origin,
        "both_lengths_equal_2pi": equal_required_length,
        "winding_claims_match": winding_claims_match,
        "different_winding_numbers": different_homotopy_classes,
    }


HARD_CHECKERS = {
    "circle_winding": _check_hc02,
    "qubit_density_matrix": _check_hc10,
}


def run_hard_assertions(case: dict[str, Any], answer: str) -> dict[str, Any]:
    spec = case.get("hard_assertion")
    if not isinstance(spec, dict):
        return {"applicable": False, "passed": True, "checks": {}}
    checker_id = str(spec.get("checker") or "")
    checker = HARD_CHECKERS.get(checker_id)
    if checker is None:
        return {
            "applicable": True,
            "passed": False,
            "checks": {},
            "error": f"unknown hard assertion checker: {checker_id}",
        }
    payload = _extract_hard_payload(answer)
    if payload is None:
        return {
            "applicable": True,
            "passed": False,
            "checks": {},
            "error": "machine-readable hard_assertion payload missing or invalid",
        }
    try:
        checks = checker(payload)
    except (TypeError, ValueError, OverflowError) as exc:
        return {
            "applicable": True,
            "passed": False,
            "checks": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "applicable": True,
        "passed": all(checks.values()),
        "checks": checks,
        "checker": checker_id,
    }


def enforce_hard_assertion_verdict(
    semantic_judgment: dict[str, Any], hard_result: dict[str, Any],
) -> dict[str, Any]:
    result = dict(semantic_judgment)
    if hard_result.get("applicable") and not hard_result.get("passed"):
        result["semantic_verdict_before_hard_assertion"] = result.get("verdict")
        result["verdict"] = "fail"
        result["first_material_error"] = (
            "确定性硬断言失败：" + str(
                hard_result.get("error")
                or ", ".join(key for key, value in hard_result.get("checks", {}).items() if not value)
            )
        )
        result["hard_assertion_override"] = True
    return result
