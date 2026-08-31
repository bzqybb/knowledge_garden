"""Shared configuration for the independent OpenAI-compatible evaluator."""

from __future__ import annotations

import os
import re

from core.config import RUNTIME_DIR
from core.credentials import load_secret


LEGACY_KEY_PATH = RUNTIME_DIR / "kimi-eval-api-key.dpapi"
GLM_KEY_PATH = RUNTIME_DIR / "glm-eval-api-key.dpapi"
DEEPSEEK_KEY_PATH = RUNTIME_DIR / "deepseek-eval-api-key.dpapi"
GLM_GENERATOR_KEY_PATH = RUNTIME_DIR / "glm-generator-api-key.dpapi"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_BASE_URL = "https://api.deepseek.com"
GLM_BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4"
TOKENHUB_BASE_URL = "https://tokenhub.tencentmaas.com/v1"


def judge_api_key(model: str | None = None) -> str:
    """Load only the credential belonging to the selected evaluator provider."""
    selected = model or judge_model()
    generic = os.getenv("JUDGE_API_KEY", "").strip()
    if generic:
        return generic
    if selected.startswith("glm-"):
        if os.getenv("JUDGE_USE_GENERATOR_CREDENTIAL", "").strip().lower() in {"1", "true", "yes", "on"}:
            return load_secret(GLM_GENERATOR_KEY_PATH).strip()
        for name in ("GLM_API_KEY", "ZHIPU_API_KEY", "BIGMODEL_API_KEY"):
            value = os.getenv(name, "").strip()
            if value:
                return value
        return load_secret(GLM_KEY_PATH).strip()
    if selected in {"deepseek-v4-pro", "deepseek-v4-flash"}:
        for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_EVAL_API_KEY"):
            value = os.getenv(name, "").strip()
            if value:
                return value
        return load_secret(DEEPSEEK_KEY_PATH).strip()
    for name in ("TOKENHUB_API_KEY", "KIMI_API_KEY", "MOONSHOT_API_KEY"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return load_secret(LEGACY_KEY_PATH).strip()


def judge_model() -> str:
    return (
        os.getenv("JUDGE_MODEL", "").strip()
        or os.getenv("KIMI_EVAL_MODEL", "").strip()
        or DEFAULT_MODEL
    )


def judge_base_url(model: str | None = None) -> str:
    explicit = os.getenv("JUDGE_BASE_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    selected = model or judge_model()
    if selected.startswith("glm-"):
        return (
            os.getenv("GLM_BASE_URL", "").strip()
            or os.getenv("ZHIPU_BASE_URL", "").strip()
            or GLM_BASE_URL
        ).rstrip("/")
    if selected in {"deepseek-v4-pro", "deepseek-v4-flash"}:
        return (
            os.getenv("DEEPSEEK_BASE_URL", "").strip()
            or DEFAULT_BASE_URL
        ).rstrip("/")
    return (
        os.getenv("KIMI_BASE_URL", "").strip()
        or TOKENHUB_BASE_URL
    ).rstrip("/")


def judge_label(model: str | None = None) -> str:
    selected = model or judge_model()
    if selected.startswith("glm-"):
        return f"智谱 GLM（{selected}）"
    if selected in {"deepseek-v4-pro", "deepseek-v4-flash"}:
        return f"DeepSeek 官方 API（{selected}）"
    if selected.startswith("deepseek-v4-flash"):
        return f"腾讯云 DeepSeek-V4-Flash（{selected}）"
    if selected.startswith("kimi"):
        return f"腾讯云 Kimi（{selected}）"
    return f"腾讯云 TokenHub（{selected}）"


def judge_independence(model: str | None = None) -> str:
    selected = model or judge_model()
    if selected.startswith("glm-") and os.getenv("JUDGE_USE_GENERATOR_CREDENTIAL", "").strip().lower() in {"1", "true", "yes", "on"}:
        return "same_provider_and_credential_lane_as_generator"
    if selected.startswith("glm-"):
        return "same_provider_separate_credential_lane"
    return "heterogeneous_provider"


def judge_request_options(model: str | None = None) -> dict[str, object]:
    """Use concise non-thinking JSON judging when the provider supports it."""
    selected = model or judge_model()
    default_temperature = "0.2"
    options: dict[str, object] = {
        "temperature": float(os.getenv("JUDGE_TEMPERATURE", default_temperature)),
    }
    # TokenHub currently validates Kimi K3 with a fixed temperature.  Sending
    # the generic evaluator default (0.2) yields HTTP 400 before any scoring.
    if selected == "kimi-k3" and "JUDGE_TEMPERATURE" not in os.environ:
        options["temperature"] = 1.0
    if selected in {"kimi-k2.6", "kimi-k2.5"} or selected.startswith("deepseek-v4"):
        thinking = os.getenv("JUDGE_THINKING", "disabled").strip().lower()
        if thinking not in {"enabled", "disabled"}:
            thinking = "disabled"
        options["extra_body"] = {"thinking": {"type": thinking}}
        if selected in {"kimi-k2.6", "kimi-k2.5"} and thinking == "disabled" and "JUDGE_TEMPERATURE" not in os.environ:
            options["temperature"] = 0.6
    return options


def judge_slug(model: str | None = None) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", (model or judge_model()).casefold()).strip("-") or "judge"
