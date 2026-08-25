from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RUNTIME_DIR = DATA_DIR / "runtime"
DB_PATH = RUNTIME_DIR / "garden.db"
SAVED_API_KEY_PATH = RUNTIME_DIR / "garden-api-key.dpapi"
SAVED_UNDERSTANDING_API_KEY_PATH = RUNTIME_DIR / "understanding-api-key.dpapi"
WEB_DIR = ROOT / "web"


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


def llm_config() -> LLMConfig:
    """Prefer a session key, then decrypt the current-user DPAPI credential in memory."""
    api_key = os.getenv("GARDEN_API_KEY", "").strip()
    saved_key_allowed = os.getenv("GARDEN_DISABLE_SAVED_API_KEY", "").strip() != "1"
    if not api_key and saved_key_allowed and SAVED_API_KEY_PATH.is_file():
        from core.credentials import load_secret

        api_key = load_secret(SAVED_API_KEY_PATH).strip()
    return LLMConfig(
        api_key=api_key,
        base_url=os.getenv("GARDEN_BASE_URL", "https://api.deepseek.com").rstrip("/"),
        model=os.getenv("GARDEN_MODEL", "deepseek-v4-flash").strip(),
    )


def understanding_llm_config() -> LLMConfig:
    """Return the dedicated question-understanding model configuration.

    A separate key keeps the GLM understanding agent independent from the
    DeepSeek teaching/generation model.  When GLM is not configured the caller
    deliberately falls back to ``llm_config`` rather than disabling the whole
    agent pipeline.
    """
    api_key = os.getenv("GARDEN_UNDERSTANDING_API_KEY", "").strip()
    saved_key_allowed = os.getenv("GARDEN_DISABLE_SAVED_API_KEY", "").strip() != "1"
    if not api_key and saved_key_allowed and SAVED_UNDERSTANDING_API_KEY_PATH.is_file():
        from core.credentials import load_secret

        api_key = load_secret(SAVED_UNDERSTANDING_API_KEY_PATH).strip()
    return LLMConfig(
        api_key=api_key,
        base_url=os.getenv(
            "GARDEN_UNDERSTANDING_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"
        ).rstrip("/"),
        # AirX is used as a low-latency semantic router.  The understanding
        # agent should parse intent quickly; it does not need the slower
        # deliberative models used for the final teaching answer.
        model=os.getenv("GARDEN_UNDERSTANDING_MODEL", "glm-4.5-airx").strip(),
    )


def ensure_runtime_dirs() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
