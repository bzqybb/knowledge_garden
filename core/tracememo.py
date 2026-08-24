from __future__ import annotations

import json
import os
import re
import socket
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from core.config import RUNTIME_DIR
from core.credentials import load_secret, save_secret


TRACEMEMO_TOKEN_PATH = RUNTIME_DIR / "tracememo-api-token.dpapi"
DEFAULT_BASE_URL = "http://127.0.0.1:6131"
MAX_MESSAGES = 300
MAX_CONTENT_CHARS = 10_000


class TraceMemoError(RuntimeError):
    pass


@dataclass(frozen=True)
class TraceMemoConfig:
    base_url: str
    token: str
    token_saved: bool

    @property
    def enabled(self) -> bool:
        return bool(self.token)


def _validate_loopback_url(value: str) -> str:
    base = (value or DEFAULT_BASE_URL).strip().rstrip("/")
    parsed = urlparse(base)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("TraceMemo 地址必须是本机 HTTP 地址（127.0.0.1、localhost 或 ::1）")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("TraceMemo 地址中不能包含凭证、查询参数或片段")
    return base


def tracememo_config(base_url: str | None = None) -> TraceMemoConfig:
    # New installations use TRACEMEMO_API_TOKEN.  The legacy variable remains
    # readable so an existing WechatExplorer/early TraceMemo setup does not
    # silently break during migration.
    token = (
        os.getenv("TRACEMEMO_API_TOKEN", "").strip()
        or os.getenv("WECHATEXPLORER_API_TOKEN", "").strip()
    )
    saved = TRACEMEMO_TOKEN_PATH.is_file()
    if not token and saved and os.getenv("GARDEN_DISABLE_SAVED_TRACEMEMO_TOKEN", "").strip() != "1":
        token = load_secret(TRACEMEMO_TOKEN_PATH).strip()
    return TraceMemoConfig(
        base_url=_validate_loopback_url(base_url or os.getenv("TRACEMEMO_BASE_URL", DEFAULT_BASE_URL)),
        token=token,
        token_saved=saved,
    )


def configure_tracememo(*, base_url: str, token: str = "", save_token: bool = True) -> dict[str, Any]:
    validated = _validate_loopback_url(base_url)
    os.environ["TRACEMEMO_BASE_URL"] = validated
    clean_token = token.strip()
    if clean_token:
        os.environ["TRACEMEMO_API_TOKEN"] = clean_token
        if save_token:
            save_secret(TRACEMEMO_TOKEN_PATH, clean_token)
    return connection_summary(validated)


def forget_tracememo_token() -> None:
    os.environ.pop("TRACEMEMO_API_TOKEN", None)
    if TRACEMEMO_TOKEN_PATH.is_file():
        TRACEMEMO_TOKEN_PATH.unlink()


def connection_summary(base_url: str | None = None) -> dict[str, Any]:
    config = tracememo_config(base_url)
    return {
        "base_url": config.base_url,
        "token_configured": config.enabled,
        "token_saved": config.token_saved,
    }


class TraceMemoClient:
    """Narrow, read-only client for TraceMemo's loopback Local HTTP API."""

    def __init__(self, config: TraceMemoConfig | None = None, timeout: float = 12.0):
        self.config = config or tracememo_config()
        self.timeout = timeout

    def _get(self, endpoint: str, params: dict[str, Any] | None = None, *, auth: bool = True) -> dict[str, Any]:
        if not endpoint.startswith("/") or ".." in endpoint:
            raise ValueError("非法 TraceMemo API 路径")
        if auth and not self.config.token:
            raise TraceMemoError("尚未配置 TraceMemo Token")
        query = urlencode({k: v for k, v in (params or {}).items() if v not in (None, "")})
        url = f"{self.config.base_url}/api/v1{endpoint}" + (f"?{query}" if query else "")
        headers = {"Accept": "application/json"}
        if auth:
            headers["Authorization"] = f"Bearer {self.config.token}"
        request = Request(url, headers=headers, method="GET")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            messages = {
                401: "TraceMemo Token 无效或已经轮换，请重新复制",
                403: "TraceMemo 拒绝了当前来源；请确认 API 只监听本机",
                404: "TraceMemo 没找到对应会话或接口",
                503: "TraceMemo 数据库尚未就绪，请先在 API Center 完成连接",
            }
            raise TraceMemoError(messages.get(exc.code, f"TraceMemo 请求失败（HTTP {exc.code}）：{detail}")) from exc
        except (URLError, socket.timeout, TimeoutError) as exc:
            raise TraceMemoError("无法连接 TraceMemo，请确认应用和 API Center 已启动") from exc
        except json.JSONDecodeError as exc:
            raise TraceMemoError("TraceMemo 返回了无法解析的数据") from exc

    def health(self) -> dict[str, Any]:
        return self._get("/health", auth=False)

    def current_time(self) -> dict[str, Any]:
        return self._get("/current_time")

    def recent_chats(self, limit: int = 20) -> dict[str, Any]:
        return self._get("/recent_chat", {"limit": max(1, min(int(limit), 50))})

    def contacts(self, filter_text: str = "", contact_type: str = "") -> dict[str, Any]:
        return self._get("/contact", {"filter": filter_text.strip(), "type": contact_type.strip()})

    def official_accounts(self, filter_text: str = "") -> dict[str, Any]:
        payload = self.contacts(filter_text)
        contacts = payload.get("contacts") if isinstance(payload.get("contacts"), list) else []
        official = [
            item for item in contacts
            if isinstance(item, dict) and (
                bool(item.get("isOfficialAccount"))
                or str(item.get("m_nsUsrName", "")).startswith("gh_")
            )
        ]
        official.sort(key=lambda item: str(item.get("m_nsNickName") or item.get("remark") or ""))
        return {"count": len(official), "items": official}

    def resolve(self, query: str) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("请填写联系人或群聊名称")
        return self._get("/resolve", {"q": query.strip()})

    def chatlog(self, talker: str, *, time_range: str = "", start_time: str = "", end_time: str = "") -> dict[str, Any]:
        if not talker.strip():
            raise ValueError("请填写要读取的会话")
        data = self._get("/chatlog", {
            "talker": talker.strip(), "time": time_range.strip(),
            "startTime": start_time.strip(), "endTime": end_time.strip(),
        })
        raw_messages = data.get("messages") if isinstance(data.get("messages"), list) else []
        # TraceMemo chat logs are ordered oldest-first.  A bounded reader must
        # retain the newest window, otherwise an active official account shows
        # only historical articles and appears not to update.
        selected = raw_messages[-MAX_MESSAGES:]
        offset = max(0, len(raw_messages) - len(selected))
        data["messages"] = [
            normalize_message(item, offset + index) for index, item in enumerate(selected)
        ]
        data["truncated"] = len(raw_messages) > MAX_MESSAGES
        return data

    def official_articles(self, talker: str, *, days: int = 30) -> dict[str, Any]:
        days = max(1, min(int(days), 365))
        # Reader Skill requires checking TraceMemo's own clock before resolving
        # relative windows. TraceMemo and the Garden share the same local host.
        clock = self.current_time()
        now = _clock_datetime(clock)
        result = self.chatlog(
            talker,
            start_time=str(int((now - timedelta(days=days)).timestamp())),
            end_time=str(int(now.timestamp())),
        )
        articles = [
            item for item in result.get("messages", [])
            if isinstance(item.get("article"), dict) and item["article"].get("url")
        ]
        articles.sort(key=lambda item: item.get("sent_at_sort", 0), reverse=True)
        return {
            "talker": talker,
            "days": days,
            "clock": clock,
            "count": len(articles),
            "articles": articles,
            "truncated": result.get("truncated", False),
        }


def _first(mapping: dict[str, Any], names: tuple[str, ...], default: Any = "") -> Any:
    for name in names:
        value = mapping.get(name)
        if value not in (None, ""):
            return value
    return default


def _clock_datetime(clock: dict[str, Any]) -> datetime:
    for key in ("localTime", "local_time", "now", "datetime", "timestamp", "time"):
        value = clock.get(key)
        if value in (None, ""):
            continue
        try:
            if isinstance(value, (int, float)):
                return datetime.fromtimestamp(float(value)).astimezone()
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone()
        except (ValueError, TypeError, OSError):
            continue
    return datetime.now().astimezone()


def _normalize_timestamp(value: Any) -> tuple[str, float]:
    try:
        numeric = float(value)
        if numeric > 10_000_000_000:
            numeric /= 1000
        return datetime.fromtimestamp(numeric).astimezone().isoformat(timespec="minutes"), numeric
    except (ValueError, TypeError, OSError):
        return str(value or ""), 0.0


def _article_data(item: dict[str, Any]) -> dict[str, str]:
    raw = item.get("contentData")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {}
    if not isinstance(raw, dict):
        return {}
    title = str(_first(raw, ("title", "name"))).strip()
    url = str(_first(raw, ("url", "link"))).strip()
    if not title or not url:
        return {}
    return {
        "title": title,
        "description": str(_first(raw, ("des", "description", "digest"))).strip(),
        "url": url,
        "publisher": str(_first(raw, ("appname", "publisher", "source"))).strip(),
        "card_type": str(_first(raw, ("type", "typeVal"))).strip(),
    }


def normalize_message(value: Any, index: int = 0) -> dict[str, Any]:
    item = value if isinstance(value, dict) else {"content": str(value)}
    article = _article_data(item)
    content = _first(item, ("content", "text", "message", "displayContent", "strContent"))
    if isinstance(content, (dict, list)):
        content = json.dumps(content, ensure_ascii=False)
    content = str(content or "").strip()
    if article:
        content = "\n".join(part for part in (article["title"], article["description"]) if part)
    content = content[:MAX_CONTENT_CHARS]
    message_id = str(_first(item, ("id", "messageId", "message_id", "localId", "local_id", "msgSvrId"), f"message-{index}"))
    message_type = str(_first(item, ("type", "messageType", "message_type"), "text"))
    sender = str(_first(item, ("senderName", "sender", "displayName", "nickname", "from")))
    if article and (not sender or sender.lower() == "user"):
        sender = article.get("publisher") or sender
    sent_at, sent_at_sort = _normalize_timestamp(
        _first(item, ("createTime", "timestamp", "time", "sentAt", "sent_at"))
    )
    is_system = bool(
        sender.lower() == "system"
        or "system" in message_type.lower()
        or re.search(r"joined the group chat|加入了群聊|撤回了一条消息|修改群名", content, re.I)
    )
    return {
        "source_message_id": message_id,
        "sender": sender,
        "sent_at": sent_at,
        "sent_at_sort": sent_at_sort,
        "content": content or "[非文本消息]",
        "message_type": message_type,
        "is_system": is_system,
        "article": article,
        "source": item,
    }
