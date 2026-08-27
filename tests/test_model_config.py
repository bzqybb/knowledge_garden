from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import config
from core.llm import _primary_provider_options


class ModelConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.garden_key = self.root / "garden.dpapi"
        self.glm_generator_key = self.root / "glm-generator.dpapi"
        self.glm_key = self.root / "understanding.dpapi"
        self.garden_key.touch()
        self.glm_generator_key.touch()
        self.glm_key.touch()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_saved_glm_generator_key_is_preferred_for_primary_teaching_model(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch.object(
            config, "SAVED_API_KEY_PATH", self.garden_key
        ), patch.object(
            config, "SAVED_GLM_GENERATOR_API_KEY_PATH", self.glm_generator_key
        ), patch.object(
            config, "SAVED_UNDERSTANDING_API_KEY_PATH", self.glm_key
        ), patch("core.credentials.load_secret", return_value="glm-generator-secret") as load:
            configured = config.llm_config()

        self.assertEqual(configured.api_key, "glm-generator-secret")
        self.assertEqual(configured.model, "glm-5.2")
        self.assertEqual(configured.base_url, "https://open.bigmodel.cn/api/coding/paas/v4")
        load.assert_called_once_with(self.glm_generator_key)

    def test_explicit_provider_still_uses_its_own_saved_credential(self) -> None:
        with patch.dict(
            os.environ, {"GARDEN_BASE_URL": "https://api.deepseek.com"}, clear=True
        ), patch.object(
            config, "SAVED_API_KEY_PATH", self.garden_key
        ), patch.object(
            config, "SAVED_GLM_GENERATOR_API_KEY_PATH", self.root / "missing-glm-generator.dpapi"
        ), patch.object(
            config, "SAVED_UNDERSTANDING_API_KEY_PATH", self.glm_key
        ), patch("core.credentials.load_secret", return_value="garden-secret") as load:
            configured = config.llm_config()

        self.assertEqual(configured.api_key, "garden-secret")
        self.assertEqual(configured.model, "deepseek-v4-pro")
        load.assert_called_once_with(self.garden_key)

    def test_saved_credentials_can_still_be_disabled(self) -> None:
        with patch.dict(
            os.environ, {"GARDEN_DISABLE_SAVED_API_KEY": "1"}, clear=True
        ), patch.object(
            config, "SAVED_API_KEY_PATH", self.garden_key
        ), patch.object(config, "SAVED_UNDERSTANDING_API_KEY_PATH", self.glm_key):
            configured = config.llm_config()

        self.assertFalse(configured.enabled)

    def test_glm_primary_disables_extra_thinking_for_structured_answers(self) -> None:
        configured = config.LLMConfig(
            "not-a-real-key", "https://open.bigmodel.cn/api/paas/v4", "glm-4.5-airx"
        )
        self.assertEqual(
            _primary_provider_options(configured),
            {"extra_body": {"thinking": {"type": "disabled"}}},
        )

    def test_project_temp_and_model_caches_default_to_workspace_drive(self) -> None:
        self.assertEqual(config.TEMP_DIR.drive.casefold(), config.ROOT.drive.casefold())
        self.assertEqual(config.CACHE_DIR.drive.casefold(), config.ROOT.drive.casefold())
        self.assertEqual(config.MODEL_CACHE_DIR.drive.casefold(), config.ROOT.drive.casefold())
        self.assertEqual(Path(tempfile.gettempdir()).resolve(), config.TEMP_DIR.resolve())
        self.assertEqual(
            Path(os.environ["HF_HOME"]).resolve(),
            (config.MODEL_CACHE_DIR / "huggingface").resolve(),
        )


if __name__ == "__main__":
    unittest.main()
