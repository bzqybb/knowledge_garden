import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.inspiration import (
    _META_REFUSAL_LANGUAGE,
    _normalize_inspiration_answer,
    explore_inspiration,
)
from core.llm import LLMError
from core.storage import GardenStore


class InspirationRoutingTests(unittest.TestCase):
    def test_general_model_failure_returns_fallback_but_is_not_counted_as_success(self):
        question = "为什么人在压力下更容易沿用熟悉的做法？"
        with tempfile.TemporaryDirectory() as folder:
            store = GardenStore(Path(folder) / "garden.db")
            with patch("core.inspiration.chat_json", side_effect=LLMError("timeout")), patch(
                "core.inspiration.search_notes", return_value=[]
            ):
                result = explore_inspiration(store, question)
        self.assertTrue(result["generation_failed"])
        self.assertTrue(result["answer"].strip())

    def test_closed_loop_model_failure_never_falls_into_psychology_template(self):
        question = "推导四次非谐振子基态能量的二阶修正，并证明微扰级数收敛半径为零。"
        with tempfile.TemporaryDirectory() as folder:
            store = GardenStore(Path(folder) / "garden.db")
            with patch("core.inspiration.chat", side_effect=LLMError("timeout")) as plain, patch(
                "core.inspiration.chat_json",
            ) as structured, patch("core.inspiration.search_notes") as search:
                result = explore_inspiration(store, question)
        self.assertEqual(plain.call_count, 2)
        structured.assert_not_called()
        search.assert_not_called()
        self.assertTrue(result["generation_failed"])
        self.assertEqual(result["primary_type"], "rigorous_exploration")
        self.assertIn("题面自足", result["answer"])
        self.assertIn("可验证路径", result["answer"])
        self.assertNotIn("请重试", result["answer"])
        self.assertNotIn("人物动机", result["answer"])
        self.assertNotIn("群体氛围", result["answer"])

    def test_closed_loop_transient_failure_retries_once_and_recovers(self):
        question = "将 k 个键均匀散列到 m 个槽，用 union bound 推导碰撞概率上界。"
        recovered = "共有 C(k,2) 对，每对碰撞概率为 1/m，故 P(collision)≤k(k-1)/(2m)。"
        with tempfile.TemporaryDirectory() as folder:
            store = GardenStore(Path(folder) / "garden.db")
            with patch(
                "core.inspiration.chat",
                side_effect=[LLMError("429 model overloaded"), recovered],
            ) as plain, patch("core.inspiration.search_notes") as search:
                result = explore_inspiration(store, question)
        self.assertEqual(plain.call_count, 2)
        search.assert_not_called()
        self.assertFalse(result["generation_failed"])
        self.assertIn("k(k-1)", result["answer"])

    def test_code_task_requires_auditable_python_fence(self):
        question = "推导 Ramsey 数 R(3,3)，并编写 Python 穷举算法验证。"
        generated = "推导如下。\n```python\nprint(6)\n```"
        with tempfile.TemporaryDirectory() as folder:
            store = GardenStore(Path(folder) / "garden.db")
            with patch("core.inspiration.chat", return_value=generated) as model, patch(
                "core.inspiration.search_notes",
            ) as search:
                result = explore_inspiration(store, question)
        search.assert_not_called()
        system_prompt = model.call_args.args[0]
        self.assertIn("```python", system_prompt)
        self.assertTrue(result["generation_diagnostics"]["auditable_python_required"])
        self.assertTrue(result["generation_diagnostics"]["auditable_python_present"])

    def test_escaped_document_newlines_do_not_corrupt_latex_nabla(self):
        raw = r"第一段\n\n第二段：$\nabla f=0$"
        normalized = _normalize_inspiration_answer(raw)
        self.assertIn("第一段\n\n第二段", normalized)
        self.assertIn(r"\nabla f", normalized)

    def test_meta_refusal_guard_covers_paraphrases(self):
        for answer in (
            "这次先不回答。",
            "我暂时不直接作答。",
            "当前无法给出答案，请稍后再试。",
            "请换个问题。",
            "抱歉，我不能回答这个问题。",
            "我无法作答。",
            "这个问题我不方便回答。",
            "现阶段无法回答。",
            "无法给出答案。",
            "我不能直接回答。",
        ):
            with self.subTest(answer=answer):
                self.assertRegex(answer, _META_REFUSAL_LANGUAGE)
        self.assertNotRegex(
            "这个实验不回答因果机制，只测相关性，因此结论不能外推。",
            _META_REFUSAL_LANGUAGE,
        )


if __name__ == "__main__":
    unittest.main()
