from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from core import config
from core.beta_access import BetaAccessError, _validate_cloud_url
from core.model_proxy import ModelProxyError, _prepared_payload


class PublicBetaTests(unittest.TestCase):
    def test_beta_model_uses_encrypted_session_instead_of_provider_key(self) -> None:
        with patch.dict(os.environ, {
            "GARDEN_BETA_MODE": "true",
            "GARDEN_BETA_CLOUD_URL": "https://beta.example.test",
            "GARDEN_API_KEY": "must-not-be-used",
        }, clear=True), patch("core.credentials.load_secret", return_value="desktop-session") as load:
            selected = config.llm_config()

        self.assertEqual(selected.api_key, "desktop-session")
        self.assertEqual(selected.base_url, "https://beta.example.test/api/model/v1")
        self.assertEqual(selected.model, "glm-5.2")
        load.assert_called_once_with(config.SAVED_BETA_SESSION_PATH)

    def test_beta_cloud_requires_https_except_for_local_smoke_tests(self) -> None:
        self.assertEqual(_validate_cloud_url("https://beta.example.test"), "https://beta.example.test")
        self.assertEqual(_validate_cloud_url("http://127.0.0.1:8876"), "http://127.0.0.1:8876")
        with self.assertRaises(BetaAccessError):
            _validate_cloud_url("http://untrusted.example.test")

    def test_model_proxy_forces_server_model_and_drops_unknown_fields(self) -> None:
        with patch("core.model_proxy.llm_config", return_value=config.LLMConfig(
            "owner-secret", "https://open.bigmodel.cn/api/paas/v4", "glm-5.2",
        )):
            prepared = _prepared_payload({
                "model": "expensive-unapproved-model",
                "messages": [{"role": "user", "content": "你好"}],
                "stream": True,
                "user": "should-not-pass-through",
            })
        self.assertEqual(prepared["model"], "glm-5.2")
        self.assertNotIn("user", prepared)
        self.assertEqual(prepared["thinking"], {"type": "disabled"})
        self.assertEqual(prepared["reasoning_effort"], "none")

    def test_model_proxy_rejects_missing_or_oversized_messages(self) -> None:
        with self.assertRaises(ModelProxyError):
            _prepared_payload({})
        with self.assertRaises(ModelProxyError):
            _prepared_payload({"messages": [{"role": "user", "content": "x" * 250_000}]})


if __name__ == "__main__":
    unittest.main()
