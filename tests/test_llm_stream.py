from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.llm import chat_stream


class _FakeModel:
    def __init__(self, **_kwargs) -> None:
        pass

    def stream(self, messages):
        assert messages[0][0] == "system"
        yield SimpleNamespace(content="速度")
        yield SimpleNamespace(content="已经改善")


class LLMStreamTests(unittest.TestCase):
    def test_stream_forwards_each_delta_and_returns_complete_text(self) -> None:
        config = SimpleNamespace(
            enabled=True, api_key="test", base_url="https://example.test", model="model",
        )
        deltas: list[str] = []
        with patch("core.llm.llm_config", return_value=config), patch(
            "core.llm._langchain_components",
            return_value=(object, _FakeModel, object, object),
        ):
            answer = chat_stream("system", "question", on_delta=deltas.append)
        self.assertEqual(deltas, ["速度", "已经改善"])
        self.assertEqual(answer, "速度已经改善")


if __name__ == "__main__":
    unittest.main()
