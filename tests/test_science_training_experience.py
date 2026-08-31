import unittest

from evals.build_science_training_experience import build_experience


class ScienceTrainingExperienceTests(unittest.TestCase):
    def test_report_is_split_into_auditable_surface_records(self):
        report = {
            "rows": [{
                "id": "SCI-TEST-01",
                "discipline": "数学",
                "topic": "测试题",
                "question": "请推导并运行代码验证。",
                "reference": "参考答案",
                "scoring_rubric": [{"id": "R1", "criterion": "正确性"}],
                "rubric_hash": "abc",
                "requires_tool_execution": True,
                "generator_model": "glm-5.2",
                "generator_base_host": "open.bigmodel.cn",
                "gardener": {
                    "answer": "fallback",
                    "generation_failed": True,
                    "generation_diagnostics": {
                        "errors": ["Error code: 401, token expired"],
                    },
                    "local_checks": {},
                },
                "inspiration": {
                    "answer": "```python\nprint(1)\n```",
                    "generation_failed": False,
                    "local_checks": {
                        "tool_execution_verified": False,
                        "tool_execution": {
                            "status": "no_python_block", "reason": "no_python_block",
                        },
                        "deterministic_oracle": {"passed": True, "issues": []},
                    },
                },
                "auxiliary_judge": {
                    "model": "deepseek-v4-pro",
                    "base_host": "api.deepseek.com",
                    "gardener_verdict": "unscorable",
                    "gardener_score": None,
                    "inspiration_verdict": "fail",
                    "inspiration_score": 60.0,
                    "rubric_results": [{
                        "rubric_id": "R1",
                        "criterion": "正确性",
                        "gardener_score": 0,
                        "inspiration_score": 1,
                    }],
                    "failures": {"gardener": [], "inspiration": []},
                },
            }],
        }

        records, summary = build_experience(report)

        self.assertEqual(len(records), 2)
        by_surface = {record["surface"]: record for record in records}
        self.assertEqual(by_surface["gardener"]["disposition"], "exclude_infrastructure")
        self.assertEqual(by_surface["inspiration"]["disposition"], "execution_missing_code")
        self.assertEqual(by_surface["gardener"]["answer"], "fallback")
        self.assertEqual(by_surface["inspiration"]["generator"]["model"], "glm-5.2")
        self.assertEqual(summary["cases"], 1)
        self.assertEqual(summary["surface_records"], 2)
        self.assertEqual(summary["trainable_now"], 0)

    def test_verified_oracle_failure_takes_precedence_over_missing_execution(self):
        report = {
            "rows": [{
                "id": "SCI-MATH-10",
                "requires_tool_execution": True,
                "gardener": {
                    "answer": "bad",
                    "generation_failed": False,
                    "local_checks": {
                        "tool_execution_verified": False,
                        "deterministic_oracle": {
                            "passed": False,
                            "issues": ["SELF_REFERENTIAL_TOUR_APPEND"],
                        },
                    },
                },
                "inspiration": {
                    "answer": "bad",
                    "generation_failed": False,
                    "local_checks": {
                        "tool_execution_verified": False,
                        "deterministic_oracle": {
                            "passed": False,
                            "issues": ["SELF_REFERENTIAL_TOUR_APPEND"],
                        },
                    },
                },
                "auxiliary_judge": {"failures": {}},
            }],
        }

        records, summary = build_experience(report)

        self.assertEqual({record["disposition"] for record in records}, {"repair_required_verified"})
        self.assertTrue(all(not record["trainable"] for record in records))
        self.assertEqual(summary["failure_labels"]["tool_execution_not_verified"], 2)
        self.assertEqual(summary["failure_labels"]["oracle:SELF_REFERENTIAL_TOUR_APPEND"], 2)

    def test_runtime_trace_is_preserved_and_failure_is_classified(self):
        trace = {
            "status": "failed", "reason": "nonzero_exit", "exit_code": 1,
            "stderr": "Traceback\nValueError: bad shape",
        }
        report = {"rows": [{
            "id": "SCI-X", "requires_tool_execution": True,
            "gardener": {"answer": "bad", "local_checks": {
                "tool_execution_verified": False, "tool_execution": trace,
                "deterministic_oracle": {},
            }},
            "inspiration": {"answer": "none", "local_checks": {
                "tool_execution_verified": False,
                "tool_execution": {"status": "no_python_block", "reason": "no_python_block"},
                "deterministic_oracle": {},
            }},
            "auxiliary_judge": {"failures": {}},
        }]}
        records, summary = build_experience(report)
        gardener = next(item for item in records if item["surface"] == "gardener")
        self.assertEqual(gardener["disposition"], "execution_failed")
        self.assertEqual(gardener["tool_execution"]["failure_class"], "generated_code_error")
        self.assertEqual(gardener["tool_execution"]["trace"]["exit_code"], 1)
        self.assertEqual(summary["failure_labels"]["execution:generated_code_error"], 1)


if __name__ == "__main__":
    unittest.main()
