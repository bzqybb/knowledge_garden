from __future__ import annotations

from functools import lru_cache
from queue import Queue
from threading import Thread
from typing import Any

from core.config import LLMConfig, llm_config, understanding_llm_config


class LLMError(RuntimeError):
    pass


def _primary_provider_options(config: LLMConfig) -> dict[str, Any]:
    """Keep GLM's structured teaching answers responsive and JSON-compatible."""
    if "bigmodel.cn" in config.base_url.casefold():
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    return {}


def _invoke_with_hard_timeout(function: Any, timeout: float, label: str) -> Any:
    """Enforce a wall-clock deadline even if a provider socket ignores timeout.

    Some Windows TLS/DNS failures remained blocked far beyond the timeout
    forwarded by LangChain/OpenAI. A daemon worker lets the tutoring turn fall
    back on time; the abandoned socket cannot keep the process alive.
    """
    mailbox: Queue[tuple[bool, Any]] = Queue(maxsize=1)

    def run() -> None:
        try:
            mailbox.put((True, function()))
        except BaseException as exc:  # pass provider exceptions to caller
            mailbox.put((False, exc))

    worker = Thread(target=run, name=f"garden-{label}-deadline", daemon=True)
    worker.start()
    worker.join(max(0.5, float(timeout) + 1.0))
    if worker.is_alive():
        raise LLMError(f"{label}超过 {timeout:g} 秒硬截止，已切换安全回退")
    ok, value = mailbox.get_nowait()
    if not ok:
        raise value
    return value


def _langchain_components():
    """Import lazily so the offline rule engine can still start without model extras."""
    try:
        from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise LLMError(
            "已配置大模型，但缺少 LangChain 模型组件。请运行：pip install -r requirements.txt"
        ) from exc
    return ChatPromptTemplate, ChatOpenAI, StrOutputParser, JsonOutputParser


def _model():
    config = llm_config()
    if not config.enabled:
        return None
    _, ChatOpenAI, _, _ = _langchain_components()
    return ChatOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        temperature=0.3,
        timeout=60,
        max_retries=2,
        **_primary_provider_options(config),
    )


def chat(system: str, user: str, *, temperature: float = 0.3, json_mode: bool = False) -> str | None:
    """Run the official LangChain LCEL prompt → model → parser pipeline."""
    config = llm_config()
    if not config.enabled:
        return None
    ChatPromptTemplate, ChatOpenAI, StrOutputParser, _ = _langchain_components()
    prompt = ChatPromptTemplate.from_messages([("system", "{system}"), ("human", "{user}")])
    model = ChatOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        temperature=temperature,
        timeout=60,
        max_retries=2,
        model_kwargs={"response_format": {"type": "json_object"}} if json_mode else {},
        **_primary_provider_options(config),
    )
    chain = prompt | model | StrOutputParser()
    try:
        return chain.invoke({"system": system, "user": user}).strip()
    except Exception as exc:
        raise LLMError(f"LangChain 大模型链执行失败：{exc}") from exc


def chat_json(
    system: str, user: str, *, timeout: float = 60, max_retries: int = 2,
) -> dict[str, Any] | None:
    config = llm_config()
    if not config.enabled:
        return None
    ChatPromptTemplate, ChatOpenAI, _, JsonOutputParser = _langchain_components()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system}\n{format_instructions}"),
        ("human", "{user}"),
    ])
    parser = JsonOutputParser()
    model = ChatOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        temperature=0.25,
        timeout=timeout,
        max_retries=max_retries,
        model_kwargs={"response_format": {"type": "json_object"}},
        **_primary_provider_options(config),
    )
    chain = prompt | model | parser
    try:
        result = _invoke_with_hard_timeout(
            lambda: chain.invoke({
                "system": system,
                "user": user,
                "format_instructions": parser.get_format_instructions(),
            }),
            timeout,
            "结构化模型调用",
        )
    except Exception as exc:
        raise LLMError(f"LangChain 结构化输出链执行失败：{exc}") from exc
    if not isinstance(result, dict):
        raise LLMError("大模型返回格式不正确")
    return result


def _chat_json_with_config(
    config: LLMConfig, system: str, user: str, *, timeout: float, max_retries: int,
) -> dict[str, Any] | None:
    if not config.enabled:
        return None
    ChatPromptTemplate, ChatOpenAI, _, JsonOutputParser = _langchain_components()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system}"),
        ("human", "{user}"),
    ])
    parser = JsonOutputParser()
    model = _configured_json_model(
        config.api_key, config.base_url, config.model, timeout, max_retries,
    )
    try:
        chain = prompt | model | parser
        result = _invoke_with_hard_timeout(
            lambda: chain.invoke({"system": system, "user": user}),
            timeout,
            "问题理解模型调用",
        )
    except Exception as exc:
        raise LLMError(f"LangChain 问题理解链执行失败：{exc}") from exc
    if not isinstance(result, dict):
        raise LLMError("问题理解模型返回格式不正确")
    return result


@lru_cache(maxsize=4)
def _configured_json_model(
    api_key: str, base_url: str, model_name: str, timeout: float, max_retries: int,
):
    """Reuse the provider client and its HTTPS connection pool in-process."""
    _, ChatOpenAI, _, _ = _langchain_components()
    is_glm = "bigmodel.cn" in base_url.casefold()
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=0.1,
        timeout=timeout,
        max_retries=max_retries,
        model_kwargs={"response_format": {"type": "json_object"}},
        # Problem parsing is a bounded routing task, not deep research.
        extra_body={"thinking": {"type": "disabled"}} if is_glm else None,
    )


def understanding_chat_json(
    system: str, user: str, *, timeout: float = 6, max_retries: int = 0,
) -> tuple[dict[str, Any] | None, str]:
    """Use the configured low-latency understanding route.

    A timeout or provider failure has already consumed the interaction budget,
    so the caller uses its deterministic, auditable parser instead of waiting
    for a second remote model.
    """
    config = understanding_llm_config()
    if config.enabled:
        try:
            payload = _chat_json_with_config(
                config, system, user, timeout=timeout, max_retries=max_retries,
            )
            family = "glm" if (
                "bigmodel.cn" in config.base_url.casefold()
                or config.model.casefold().startswith("glm")
            ) else "deepseek"
            return payload, f"{family}:{config.model}"
        except LLMError as model_error:
            _configured_json_model.cache_clear()
            message = str(model_error)
            reason = (
                "rate_limited"
                if any(marker in message for marker in ("429", "1305", "1113"))
                else "unavailable"
            )
            return None, f"deterministic-fallback-after-understanding-{reason}"
    return chat_json(system, user, timeout=timeout, max_retries=max_retries), "primary-model-fallback"


def prewarm_understanding_model() -> tuple[bool, str]:
    """Warm the parser connection without blocking server startup."""
    config = understanding_llm_config()
    if not config.enabled:
        return False, "问题理解模型未配置"
    try:
        payload, provider = understanding_chat_json(
            "只输出 JSON：{\"ready\":true}。不解释。",
            "ready",
            timeout=6,
            max_retries=0,
        )
    except Exception as exc:
        return False, f"问题理解模型预热失败：{exc.__class__.__name__}"
    return bool(payload), f"{provider} 预热{'完成' if payload else '跳过'}"
