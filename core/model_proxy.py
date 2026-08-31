from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Any, Iterator
from urllib.request import Request, urlopen

from core.config import llm_config


class ModelProxyError(RuntimeError):
    pass


_LOCK = threading.Lock()
_RECENT: dict[str, deque[float]] = defaultdict(deque)
_ACTIVE: dict[str, int] = defaultdict(int)
_WINDOW_SECONDS = 600.0
_MAX_REQUESTS_PER_WINDOW = 120
_MAX_CONCURRENT = 2


@contextmanager
def _account_lease(user_id: str) -> Iterator[None]:
    now = time.monotonic()
    with _LOCK:
        recent = _RECENT[user_id]
        while recent and recent[0] <= now - _WINDOW_SECONDS:
            recent.popleft()
        if len(recent) >= _MAX_REQUESTS_PER_WINDOW:
            raise ModelProxyError("该账号十分钟内请求过多，请稍后再试")
        if _ACTIVE[user_id] >= _MAX_CONCURRENT:
            raise ModelProxyError("该账号已有两个回答正在生成，请等待其中一个完成")
        recent.append(now)
        _ACTIVE[user_id] += 1
    try:
        yield
    finally:
        with _LOCK:
            _ACTIVE[user_id] = max(0, _ACTIVE[user_id] - 1)


def _prepared_payload(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ModelProxyError("模型请求缺少 messages")
    serialized_messages = json.dumps(messages, ensure_ascii=False)
    if len(serialized_messages) > 240_000:
        raise ModelProxyError("本次发送给模型的材料过长，请缩小选中范围")
    allowed = {
        "messages", "stream", "temperature", "top_p", "max_tokens",
        "max_completion_tokens", "response_format", "stop", "tools", "tool_choice",
        "frequency_penalty", "presence_penalty", "seed",
    }
    clean = {key: value for key, value in payload.items() if key in allowed}
    clean["model"] = llm_config().model
    clean["thinking"] = {"type": "disabled"}
    clean["reasoning_effort"] = "none"
    return clean


@contextmanager
def open_completion(user_id: str, payload: dict[str, Any], *, timeout: float = 180) -> Iterator[Any]:
    config = llm_config()
    if not config.enabled:
        raise ModelProxyError("公测模型服务尚未配置")
    upstream = f"{config.base_url.rstrip('/')}/chat/completions"
    request = Request(
        upstream,
        data=json.dumps(_prepared_payload(payload), ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if payload.get("stream") else "application/json",
            "User-Agent": "KnowledgeGarden-ModelProxy/0.2",
        },
        method="POST",
    )
    with _account_lease(user_id):
        with urlopen(request, timeout=timeout) as response:
            yield response
