from evals.adversarial_foundations_eval import answer_symbolic_grounding, extract_math_expressions, normalize_rubric, public_payload, route_task, symbolic_grounding


def test_closed_loop_router_disables_search():
    result = route_task("已知 H，证明并推导最低能量")
    assert result["routing_target"] == "MUST_NOT_SEARCH"
    assert result["confidence"] > 0.85
    assert result["search_enabled"] is False


def test_search_augmented_router_emits_two_stage_action():
    result = route_task("检索 2026 年权威实验数据，再推导参数")
    assert result["routing_target"] == "SEARCH_FIRST_THEN_PROVE"
    assert set(result["two_stage_action"]) == {"step1_search_query", "step2_deduction_rule"}


def test_blind_payload_only_exposes_question_and_reading_brief():
    case = {"question": "q", "reading_brief": "b", "reference": "secret", "atomic_rubric": {"required_claims": ["x"]}}
    assert public_payload(case) == {"question": "q", "reading_brief": "b"}


def test_rubric_is_non_null_and_atomic():
    case = {"id": "x", "atomic_rubric": {"required_claims": ["a", "b", "c"]}}
    rubric = normalize_rubric(case)
    assert len(rubric) == 3
    assert sum(point["weight"] for point in rubric) == 1


def test_sympy_grounding_is_deterministic():
    case = {"symbolic_checks": [{"id": "v", "symbols": ["mu", "epsilon"], "lhs": "1/sqrt(mu*epsilon)", "rhs": "(mu*epsilon)**(-1/2)"}]}
    assert symbolic_grounding(case)[0]["passed"] is True


def test_extract_and_grade_latex_answer():
    answer = r"因此 $$E_0 = \\frac{hbar*omega}{2}$$，且不存在更低能级。"
    case = {"symbolic_checks": [{"id": "e0", "target_lhs": "E_0", "symbols": ["hbar", "omega"], "lhs": "hbar*omega/2", "rhs": "hbar*omega/2"}]}
    assert extract_math_expressions(answer)
    assert answer_symbolic_grounding(case, answer)["passed"] is True


def test_negated_search_phrase_stays_closed_loop():
    result = route_task("无需检索最新资料，只证明给定恒等式")
    assert result["routing_target"] == "MUST_NOT_SEARCH"
