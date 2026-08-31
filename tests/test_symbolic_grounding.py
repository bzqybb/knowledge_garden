import unittest

from evals.adversarial_foundations_eval import (
    answer_symbolic_grounding,
    extract_math_expressions,
)


class SymbolicGroundingTests(unittest.TestCase):
    def test_chemical_concentration_brackets_are_safe_identifiers(self):
        case = {
            "symbolic_checks": [{
                "id": "substrate", "target_lhs": "[S]_{0.8}",
                "symbols": ["K_m"], "rhs": "4*K_m",
            }],
        }
        result = answer_symbolic_grounding(case, r"$[S]_{0.8}=4K_m$")
        self.assertEqual(result["checks"][0]["status"], "PASS", result)

    def test_cosmetic_quantifier_suffix_does_not_change_formula(self):
        case = {"symbolic_checks": [{
            "id": "constant_error", "target_lhs": "sup_error", "symbols": [], "rhs": "1",
        }]}
        result = answer_symbolic_grounding(case, r"$sup_error=1 \quad (\forall n)$")
        self.assertEqual(result["checks"][0]["status"], "PASS", result)

    def test_common_latex_functions_units_and_absolute_charge(self):
        cases = [
            ({"target_lhs": "t_*", "symbols": [], "rhs": "log(6)"}, r"$t_* = \ln 6$"),
            ({"target_lhs": "a_{cm}", "symbols": ["g", "theta"], "rhs": "5*g*sin(theta)/7"}, r"$a_{cm}=\dfrac{5g\sin\theta}{7}$"),
            ({"target_lhs": "t_{hit}", "symbols": [], "rhs": "log(4)/3"}, r"$t_{hit}=\frac{\ln 4}{3}\,\mathrm{s}$"),
            ({"target_lhs": "E_{cell}", "symbols": ["R", "T", "F", "a_high", "a_low"], "rhs": "R*T/F*log(a_high/a_low)"}, r"$E_{cell}=(RT/F) ln(a_high/a_low)$"),
            ({"target_lhs": "r_L", "symbols": ["m", "v_perp", "q_abs", "B"], "rhs": "m*v_perp/(q_abs*B)"}, r"$r_L=mv_perp/(|q|B)$"),
        ]
        for index, (check, answer) in enumerate(cases):
            case = {"symbolic_checks": [{"id": f"common-{index}", **check}]}
            result = answer_symbolic_grounding(case, answer)
            self.assertEqual(result["checks"][0]["status"], "PASS", result)

    def test_nested_fraction_sqrt_and_implicit_greek_product(self):
        answer = r"在线性均匀介质中，最终有 $\boxed{v=\frac{1}{\sqrt{\mu\varepsilon}}}$。"
        case = {"symbolic_checks": [{
            "id": "wave_speed", "target_lhs": "v", "symbols": ["mu", "epsilon"],
            "rhs": "1/sqrt(mu*epsilon)",
        }]}
        result = answer_symbolic_grounding(case, answer)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["checks"][0]["status"], "PASS")
        self.assertEqual(result["checks"][0]["matched_expressions"][0]["residual"], "0")

    def test_align_environment_is_split_into_independent_equations(self):
        answer = r"""
        \begin{align}
        x^* &= \frac{\alpha}{1+(x^*)^n} \\
        f'(x^*) &= \frac{n(x^*)^n}{1+(x^*)^n}
        \end{align}
        """
        expressions = extract_math_expressions(answer)
        self.assertGreaterEqual(len(expressions), 2)
        case = {"symbolic_checks": [{
            "id": "hill_slope", "target_lhs": "f'(x^*)", "symbols": ["n", "x_star"],
            "rhs": "n*x_star**n/(1+x_star**n)",
        }]}
        result = answer_symbolic_grounding(case, answer)
        self.assertEqual(result["checks"][0]["status"], "PASS", result)

    def test_parsed_but_wrong_formula_is_mismatch(self):
        answer = r"最终写成 $v=\frac{1}{\mu\varepsilon}$。"
        case = {"symbolic_checks": [{
            "id": "wave_speed", "target_lhs": "v", "symbols": ["mu", "epsilon"],
            "rhs": "1/sqrt(mu*epsilon)",
        }]}
        result = answer_symbolic_grounding(case, answer)
        self.assertFalse(result["passed"])
        self.assertEqual(result["checks"][0]["status"], "MISMATCH")
        self.assertTrue(result["checks"][0]["candidate_residuals"])

    def test_no_parseable_formula_is_extraction_failed(self):
        case = {"symbolic_checks": [{"id": "energy", "target_lhs": "E_0", "symbols": ["hbar", "omega"], "rhs": "hbar*omega/2"}]}
        result = answer_symbolic_grounding(case, "由此得到正确的基态能量，但这里略去公式。")
        self.assertEqual(result["checks"][0]["status"], "EXTRACTION_FAILED")

    def test_expected_rhs_under_wrong_target_does_not_pass(self):
        case = {"symbolic_checks": [{
            "id": "purity", "target_lhs": "purity", "symbols": [], "rhs": "0.78",
        }]}
        result = answer_symbolic_grounding(case, r"$wrong=0.78$，但纯度尚未计算。")
        self.assertEqual(result["checks"][0]["status"], "TARGET_NOT_FOUND")

    def test_negated_correct_equation_does_not_pass(self):
        case = {"symbolic_checks": [{
            "id": "wave_speed", "target_lhs": "v", "symbols": ["mu", "epsilon"],
            "rhs": "1/sqrt(mu*epsilon)",
        }]}
        result = answer_symbolic_grounding(case, r"$v=\frac{1}{\sqrt{\mu\varepsilon}}$ 是错误的写法。")
        self.assertEqual(result["checks"][0]["status"], "NEGATED_OR_EXTRACTION_FAILED")

    def test_domain_assumptions_are_explicit_not_implicitly_positive(self):
        generic = {"symbolic_checks": [{
            "id": "root", "target_lhs": "y", "symbols": ["x"], "rhs": "x",
        }]}
        positive = {"symbolic_checks": [{
            "id": "root", "target_lhs": "y", "symbols": ["x"], "rhs": "x",
            "assumptions": {"x": {"positive": True}},
        }]}
        answer = r"$y=\sqrt{x^2}$"
        self.assertEqual(answer_symbolic_grounding(generic, answer)["checks"][0]["status"], "MISMATCH")
        self.assertEqual(answer_symbolic_grounding(positive, answer)["checks"][0]["status"], "PASS")

    def test_target_bound_inequality_requires_matching_relation(self):
        case = {"symbolic_checks": [{
            "id": "cheeger", "target_lhs": "h(G)", "relation": ">=",
            "symbols": ["lambda_2"], "rhs": "lambda_2/2",
        }]}
        good = answer_symbolic_grounding(case, r"$\boxed{h(G)\ge\lambda_2/2}$")
        wrong_direction = answer_symbolic_grounding(case, r"$h(G)\le\lambda_2/2$")
        wrong_target = answer_symbolic_grounding(case, r"$q(G)\ge\lambda_2/2$")
        self.assertEqual(good["checks"][0]["status"], "PASS", good)
        self.assertEqual(wrong_direction["checks"][0]["status"], "MISMATCH", wrong_direction)
        self.assertEqual(wrong_target["checks"][0]["status"], "TARGET_NOT_FOUND", wrong_target)

    def test_later_opposite_cheeger_bound_does_not_hide_target_relation(self):
        case = {"symbolic_checks": [{
            "id": "cheeger", "target_lhs": "h(G)", "relation": ">=",
            "symbols": ["lambda_2"], "rhs": "lambda_2/2",
        }]}
        answer = r"先证 $h(G)\ge\lambda_2/2$。另一侧还有 $h(G)\le\sqrt{2\lambda_2}$。"
        result = answer_symbolic_grounding(case, answer)
        self.assertEqual(result["checks"][0]["status"], "PASS", result)

        unicode_result = answer_symbolic_grounding(case, "结论 h(G)≥λ₂/2。")
        self.assertEqual(unicode_result["checks"][0]["status"], "PASS", unicode_result)

    def test_plaintext_approximation_chain_keeps_root_target(self):
        case = {"symbolic_checks": [{
            "id": "heat", "target_lhs": "C_V", "accepted_relations": ["=", "≈"],
            "symbols": ["R", "Theta_r", "T", "e"],
            "rhs": "180*R*(Theta_r/T)**2*e**(-6*Theta_r/T)",
        }]}
        answer = "热容：C_V = ∂U/∂T ≈ 5R(6Θ_r/T)² e^{−6Θ_r/T} = 180R(Θ_r/T)² e^{−6Θ_r/T}。"
        result = answer_symbolic_grounding(case, answer)
        self.assertEqual(result["checks"][0]["status"], "PASS", result)

    def test_unicode_subscripts_superscripts_and_bold_plaintext(self):
        case = {"symbolic_checks": [{
            "id": "energy", "target_lhs": "E_0^{(2)}", "target_aliases": ["E₀⁽²⁾"],
            "symbols": ["lambda_", "hbar", "m", "omega"],
            "rhs": "-21*lambda_**2*hbar**3/(8*m**4*omega**5)",
        }]}
        answer = "**E₀⁽²⁾ = −(21/8) λ² ħ³/(m⁴ω⁵)。**"
        self.assertEqual(answer_symbolic_grounding(case, answer)["checks"][0]["status"], "PASS")

    def test_unicode_beta_alias_and_symbolic_power(self):
        case = {"symbolic_checks": [{
            "id": "hill", "target_lhs": "f'(x^*)", "target_aliases": ["β"],
            "symbols": ["n", "s"], "rhs": "n*s**n/(1+s**n)",
        }]}
        answer = "由此得到 β = nsⁿ/(1+sⁿ)。"
        self.assertEqual(answer_symbolic_grounding(case, answer)["checks"][0]["status"], "PASS")

    def test_general_target_formula_beats_later_conditional_specialization(self):
        case = {"symbolic_checks": [{
            "id": "hill", "target_lhs": "f'(x^*)", "target_aliases": ["b"],
            "symbols": ["n", "s"], "rhs": "n*s**n/(1+s**n)",
        }]}
        answer = r"一般地 $b=\frac{ns^n}{1+s^n}$。当 $n=1$ 时，$b=\frac{s}{1+s}$。"
        result = answer_symbolic_grounding(case, answer)
        self.assertEqual(result["checks"][0]["status"], "PASS", result)
        self.assertIn("ns^n", result["checks"][0]["selected_equation"])

    def test_broad_alias_requires_semantic_context(self):
        case = {"symbolic_checks": [{
            "id": "hill", "target_lhs": "f'(x^*)", "target_aliases": ["b"],
            "alias_context_pattern": "Jacobian|偏导|斜率|失稳|化简得",
            "symbols": ["n", "s"], "rhs": "n*s**n/(1+s**n)",
        }]}
        unrelated = answer_symbolic_grounding(case, r"统计量 $b=\frac{ns^n}{1+s^n}$。")
        grounded = answer_symbolic_grounding(case, r"Jacobian 的交叉斜率化简得 $b=\frac{ns^n}{1+s^n}$。")
        self.assertNotEqual(unrelated["checks"][0]["status"], "PASS", unrelated)
        self.assertEqual(grounded["checks"][0]["status"], "PASS", grounded)


if __name__ == "__main__":
    unittest.main()
