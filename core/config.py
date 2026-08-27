from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RUNTIME_DIR = DATA_DIR / "runtime"
CACHE_DIR = DATA_DIR / "cache"
TEMP_DIR = DATA_DIR / "tmp"
MODEL_CACHE_DIR = DATA_DIR / "models"
DB_PATH = RUNTIME_DIR / "garden.db"
SAVED_API_KEY_PATH = RUNTIME_DIR / "garden-api-key.dpapi"
SAVED_GLM_GENERATOR_API_KEY_PATH = RUNTIME_DIR / "glm-generator-api-key.dpapi"
SAVED_UNDERSTANDING_API_KEY_PATH = RUNTIME_DIR / "understanding-api-key.dpapi"
WEB_DIR = ROOT / "web"


def configure_project_storage() -> None:
    """Keep project-created temporary data and model caches on the D-drive workspace.

    ``GARDEN_*_DIR`` remains an explicit escape hatch, but the default is always
    below this project's ``data`` directory.  Setting ``tempfile.tempdir`` is
    intentional: Python may cache the system temp directory before a later
    environment-variable lookup.
    """
    temp_dir = Path(os.getenv("GARDEN_TEMP_DIR", str(TEMP_DIR))).resolve()
    cache_dir = Path(os.getenv("GARDEN_CACHE_DIR", str(CACHE_DIR))).resolve()
    model_cache_dir = Path(
        os.getenv("GARDEN_MODEL_CACHE_DIR", str(MODEL_CACHE_DIR))
    ).resolve()
    for directory in (RUNTIME_DIR, temp_dir, cache_dir, model_cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # These variables affect this project process and children only; they do
    # not modify the user's Windows-wide TEMP/TMP settings.
    os.environ.update({
        "GARDEN_TEMP_DIR": str(temp_dir),
        "GARDEN_CACHE_DIR": str(cache_dir),
        "GARDEN_MODEL_CACHE_DIR": str(model_cache_dir),
        "TEMP": str(temp_dir),
        "TMP": str(temp_dir),
        "TMPDIR": str(temp_dir),
        "XDG_CACHE_HOME": str(cache_dir),
        "HF_HOME": str(model_cache_dir / "huggingface"),
        "HUGGINGFACE_HUB_CACHE": str(model_cache_dir / "huggingface" / "hub"),
        "TRANSFORMERS_CACHE": str(model_cache_dir / "huggingface" / "transformers"),
        "SENTENCE_TRANSFORMERS_HOME": str(model_cache_dir / "sentence-transformers"),
        "TORCH_HOME": str(model_cache_dir / "torch"),
        "PIP_CACHE_DIR": str(cache_dir / "pip"),
        "MPLCONFIGDIR": str(cache_dir / "matplotlib"),
        "NLTK_DATA": str(model_cache_dir / "nltk"),
        "JOBLIB_TEMP_FOLDER": str(temp_dir / "joblib"),
    })
    tempfile.tempdir = str(temp_dir)


configure_project_storage()


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


def llm_config() -> LLMConfig:
    """Return the primary teaching model configuration."""
    api_key = os.getenv("GARDEN_API_KEY", "").strip()
    saved_key_allowed = os.getenv("GARDEN_DISABLE_SAVED_API_KEY", "").strip() != "1"
    configured_base_url = os.getenv("GARDEN_BASE_URL", "").strip()
    configured_model = os.getenv("GARDEN_MODEL", "").strip()
    understanding_key = os.getenv("GARDEN_UNDERSTANDING_API_KEY", "").strip()
    saved_glm_generator_available = (
        saved_key_allowed and SAVED_GLM_GENERATOR_API_KEY_PATH.is_file()
    )
    saved_glm_available = saved_key_allowed and SAVED_UNDERSTANDING_API_KEY_PATH.is_file()
    use_glm = (
        "bigmodel.cn" in configured_base_url.casefold()
        or configured_model.casefold().startswith("glm")
        or (
            not configured_base_url and not configured_model and not api_key
            and (
                saved_glm_generator_available
                or bool(understanding_key)
                or saved_glm_available
            )
        )
    )
    if use_glm:
        api_key = api_key or understanding_key
        if not api_key and saved_glm_generator_available:
            from core.credentials import load_secret

            api_key = load_secret(SAVED_GLM_GENERATOR_API_KEY_PATH).strip()
        elif not api_key and saved_glm_available:
            from core.credentials import load_secret

            api_key = load_secret(SAVED_UNDERSTANDING_API_KEY_PATH).strip()
        return LLMConfig(
            api_key=api_key,
            base_url=(
                configured_base_url
                or os.getenv(
                    "GARDEN_UNDERSTANDING_BASE_URL",
                    "https://open.bigmodel.cn/api/coding/paas/v4",
                )
            ).rstrip("/"),
            model=(
                configured_model
                or os.getenv("GARDEN_UNDERSTANDING_MODEL", "glm-5.2")
            ).strip(),
        )
    if not api_key and saved_key_allowed and SAVED_API_KEY_PATH.is_file():
        from core.credentials import load_secret

        api_key = load_secret(SAVED_API_KEY_PATH).strip()
    selected_base_url = configured_base_url or "https://api.deepseek.com"
    default_model = (
        "deepseek-v4-flash-0731"
        if "tokenhub.tencentmaas.com" in selected_base_url.casefold()
        else "deepseek-v4-pro"
    )
    return LLMConfig(
        api_key=api_key,
        base_url=selected_base_url.rstrip("/"),
        model=(configured_model or default_model).strip(),
    )


def understanding_llm_config() -> LLMConfig:
    """Return the question-understanding configuration.

    DeepSeek on the primary credential is now the default so an
    exhausted auxiliary GLM account cannot add a failed remote call to every
    ambiguous question. Set GARDEN_UNDERSTANDING_PROVIDER=glm only for an
    explicit compatibility or A/B test.
    """
    provider = os.getenv("GARDEN_UNDERSTANDING_PROVIDER", "primary").strip().casefold()
    if provider not in {"glm", "zhipu", "dedicated"}:
        return llm_config()
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
    configure_project_storage()
