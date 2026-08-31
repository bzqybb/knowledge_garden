import unittest

from unittest.mock import patch

from evals.science_code_execution import (
    audit_answer_code,
    build_docker_python_command,
    build_execution_plan,
    execute_python_in_docker,
    execute_answer_in_docker,
    execute_report_code,
    execution_preflight,
)
from evals.science_runtime_repair import (
    repair_answer_once,
    repair_answer_with_retries,
    runtime_failure_is_repairable,
)


class ScienceCodeExecutionTests(unittest.TestCase):
    def test_runtime_repair_only_accepts_concrete_nonzero_exit(self):
        self.assertTrue(runtime_failure_is_repairable({
            "executed": True, "status": "failed", "reason": "nonzero_exit",
        }))
        for trace in (
            {"executed": False, "status": "failed", "reason": "nonzero_exit"},
            {"executed": True, "status": "failed", "reason": "timeout", "timed_out": True},
            {"executed": True, "status": "blocked", "reason": "static_risk"},
        ):
            with self.subTest(trace=trace):
                self.assertFalse(runtime_failure_is_repairable(trace))

    def test_runtime_repair_candidate_requires_second_execution_pass(self):
        initial = {
            "executed": True, "status": "failed", "reason": "nonzero_exit",
            "stderr": "AttributeError: bad api",
        }

        def fake_chat(*args, **kwargs):
            return "完整回答\n```python\nprint('fixed')\n```"

        result = repair_answer_once(
            question="修复代码", answer="旧回答", execution=initial,
            chat_fn=fake_chat,
            execute_fn=lambda *args, **kwargs: {"status": "passed", "stdout": "fixed\n"},
        )
        self.assertTrue(result["attempted"])
        self.assertTrue(result["accepted"])
        self.assertEqual(result["candidate_execution"]["status"], "passed")

        failed = repair_answer_once(
            question="修复代码", answer="旧回答", execution=initial,
            chat_fn=fake_chat,
            execute_fn=lambda *args, **kwargs: {"status": "failed", "reason": "nonzero_exit"},
        )
        self.assertFalse(failed["accepted"])
        self.assertEqual(failed["reason"], "candidate_execution_failed")

    def test_runtime_repair_loop_stops_after_verified_second_candidate(self):
        initial = {
            "executed": True, "status": "failed", "reason": "nonzero_exit",
            "stderr": "first api error",
        }
        candidates = iter((
            "回答一\n```python\nraise AttributeError('second api error')\n```",
            "回答二\n```python\nprint('verified')\n```",
        ))
        executions = iter((
            {"executed": True, "status": "failed", "reason": "nonzero_exit", "stderr": "second api error"},
            {"executed": True, "status": "passed", "reason": "completed", "stdout": "verified\n"},
        ))
        result = repair_answer_with_retries(
            question="修复串联错误", answer="首稿", execution=initial, max_attempts=2,
            chat_fn=lambda *args, **kwargs: next(candidates),
            execute_fn=lambda *args, **kwargs: next(executions),
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(result["candidate_execution"]["status"], "passed")

    def test_benign_code_still_requires_isolated_backend(self):
        audit = audit_answer_code("""```python
import math
print(math.sqrt(4))
```""")
        self.assertEqual(audit["decision"], "isolated_backend_required")
        self.assertFalse(audit["host_execution_allowed"])
        self.assertEqual(audit["imports"], ["math"])

    def test_file_network_process_and_dynamic_access_are_rejected(self):
        answer = """```python
import os
import requests
print(open("secret.txt").read())
os.system("whoami")
getattr(object(), "x")
```"""
        audit = audit_answer_code(answer)
        self.assertEqual(audit["decision"], "rejected_static_risk")
        self.assertIn("blocked_import:os", audit["risks"])
        self.assertIn("blocked_import:requests", audit["risks"])
        self.assertIn("blocked_call:open", audit["risks"])
        self.assertIn("blocked_attribute_call:system", audit["risks"])
        self.assertIn("blocked_call:getattr", audit["risks"])

    def test_syntax_error_and_absent_code_are_distinct(self):
        broken = audit_answer_code("```python\nfor x in:\n    pass\n```")
        absent = audit_answer_code("这里只有推导，没有代码块。")
        self.assertEqual(broken["decision"], "rejected_syntax")
        self.assertEqual(absent["decision"], "no_python_block")

    def test_common_mapping_string_and_main_guard_are_not_false_positives(self):
        answer = """```python
data = {"x": 1}
print(data.get("x"))
print("a-b".replace("-", "+"))
if __name__ == "__main__":
    print("ok")
```"""
        audit = audit_answer_code(answer)
        self.assertEqual(audit["decision"], "isolated_backend_required")
        self.assertEqual(audit["risks"], [])

    def test_report_plan_never_claims_execution(self):
        report = {"rows": [{
            "id": "SCI-X-01", "discipline": "测试",
            "gardener": {"answer": "```python\nprint(1)\n```", "generation_failed": False},
            "inspiration": {"answer": "no code", "generation_failed": False},
        }]}
        records, summary = build_execution_plan(report)
        self.assertEqual(len(records), 2)
        self.assertEqual(summary["executed"], 0)
        self.assertTrue(all(record["execution"]["status"] == "not_executed" for record in records))

    def test_docker_command_has_strict_isolation_and_no_mount_or_pull(self):
        command = build_docker_python_command(image="zhili-science-runner:test")
        joined = " ".join(command)
        self.assertIn("--network=none", command)
        self.assertIn("--read-only", command)
        self.assertIn("--cap-drop=ALL", command)
        self.assertIn("--pull=never", command)
        self.assertIn("--interactive", command)
        self.assertIn("--user=65534:65534", command)
        self.assertIn("--env=HOME=/tmp", command)
        self.assertIn("--env=OMP_NUM_THREADS=1", command)
        self.assertIn("--env=OPENBLAS_NUM_THREADS=1", command)
        self.assertIn("--env=MKL_NUM_THREADS=1", command)
        self.assertIn("--env=NUMEXPR_NUM_THREADS=1", command)
        self.assertIn("--ulimit=nofile=64:64", command)
        self.assertNotIn("--volume", joined)
        self.assertNotIn("-v ", joined)
        self.assertEqual(command[-3:], ["python", "-I", "-"])

    def test_preflight_never_falls_back_to_host_python(self):
        audit = audit_answer_code("```python\nprint(1)\n```")
        with patch("evals.science_code_execution.shutil.which", return_value=None):
            preflight = execution_preflight(audit)
        self.assertFalse(preflight["ready"])
        self.assertEqual(preflight["reason"], "isolated_backend_unavailable")
        self.assertFalse(preflight["backend"]["host_execution_fallback"])

    def test_executor_refuses_static_risk_before_starting_process(self):
        with patch("evals.science_code_execution.subprocess.Popen") as popen:
            result = execute_python_in_docker("import os\nprint(os.getcwd())")
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["executed"])
        popen.assert_not_called()

    def test_runtime_report_does_not_claim_scientific_correctness(self):
        report = {"rows": [{
            "id": "SCI-X-01", "discipline": "测试",
            "gardener": {"answer": "```python\nprint(4)\n```"},
            "inspiration": {"answer": "没有代码"},
        }]}
        fake = {
            "status": "passed", "reason": "completed", "executed": True,
            "backend": "docker", "exit_code": 0, "stdout": "4\n", "stderr": "",
        }
        with patch(
            "evals.science_code_execution.execute_python_in_docker",
            return_value=fake,
        ):
            records, summary = execute_report_code(report, limit=1)
        self.assertEqual(summary["executed"], 1)
        self.assertEqual(records[0]["runtime_status"], "runtime_passed")
        self.assertFalse(records[0]["scientific_correctness_verified"])
        self.assertEqual(summary["scientific_correctness_verified"], 0)

    def test_answer_blocks_share_one_execution_namespace(self):
        answer = """```python
x = 3
```
```python
print(x + 1)
```"""
        fake = {
            "status": "passed", "reason": "completed", "executed": True,
            "stdout": "4\n", "stderr": "", "exit_code": 0,
        }
        with patch(
            "evals.science_code_execution.execute_python_in_docker",
            return_value=fake,
        ) as execute:
            result = execute_answer_in_docker(answer)
        source = execute.call_args.args[0]
        self.assertLess(source.index("x = 3"), source.index("print(x + 1)"))
        self.assertEqual(result["blocks_combined"], 2)

    def test_answer_selects_domain_image_from_imports(self):
        with patch(
            "evals.science_code_execution.execute_python_in_docker",
            return_value={"status": "passed", "reason": "completed", "executed": True},
        ) as execute:
            execute_answer_in_docker("```python\nfrom rdkit import Chem\n```")
        self.assertEqual(execute.call_args.kwargs["image"], "zhili-science-bio:latest")

    def test_answer_rejects_incompatible_multiple_profiles(self):
        result = execute_answer_in_docker(
            "```python\nimport qiskit\nfrom rdkit import Chem\n```",
        )
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["reason"], "multiple_dependency_profiles_required")


if __name__ == "__main__":
    unittest.main()
