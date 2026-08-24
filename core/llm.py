from __future__ import annotations

from typing import Any

from core.config import llm_config


class LLMError(RuntimeError):
    pass


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
    )
    chain = prompt | model | parser
    try:
        result = chain.invoke({
            "system": system,
            "user": user,
            "format_instructions": parser.get_format_instructions(),
        })
    except Exception as exc:
        raise LLMError(f"LangChain 结构化输出链执行失败：{exc}") from exc
    if not isinstance(result, dict):
        raise LLMError("大模型返回格式不正确")
    return result
