from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from core.config import DATA_DIR
from core.credentials import CredentialError, load_secret, save_secret


BETA_SESSION_PATH = DATA_DIR / "runtime" / "public-beta-session.dpapi"
BETA_ACCOUNT_PATH = DATA_DIR / "runtime" / "public-beta-account.json"


class BetaAccessError(RuntimeError):
    pass


def beta_cloud_url() -> str:
    return os.getenv("GARDEN_BETA_CLOUD_URL", "").strip().rstrip("/")


def beta_mode() -> bool:
    return os.getenv("GARDEN_BETA_MODE", "").strip().casefold() in {
        "1", "true", "yes", "on",
    } and bool(beta_cloud_url())


def _validate_cloud_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme == "https" and parsed.netloc:
        return url.rstrip("/")
    if parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}:
        return url.rstrip("/")
    raise BetaAccessError("公测服务地址必须使用 HTTPS")


def _read_account() -> dict[str, str] | None:
    if not BETA_ACCOUNT_PATH.is_file():
        return None
    try:
        value = json.loads(BETA_ACCOUNT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    user_id = str(value.get("id", "")).strip()
    email = str(value.get("email", "")).strip()
    return {"id": user_id, "email": email} if user_id and email else None


def beta_session_token() -> str:
    if not beta_mode() or not BETA_SESSION_PATH.is_file():
        return ""
    try:
        return load_secret(BETA_SESSION_PATH).strip()
    except (CredentialError, OSError):
        return ""


def beta_user() -> dict[str, str] | None:
    account = _read_account()
    return account if account and beta_session_token() else None


def beta_status() -> dict[str, Any]:
    user = beta_user()
    return {
        "enabled": beta_mode(),
        "authenticated": bool(user),
        "user": user,
        "cloud_url": beta_cloud_url(),
    }


def cloud_json(
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    authenticated: bool = True,
    timeout: float = 30,
) -> dict[str, Any]:
    base = _validate_cloud_url(beta_cloud_url())
    headers = {"Accept": "application/json", "User-Agent": "KnowledgeGarden-PublicBeta/0.2"}
    token = beta_session_token() if authenticated else ""
    if authenticated and not token:
        raise BetaAccessError("请先登录公测账号")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = None
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(f"{base}{path}", data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("error", "")
        except Exception:
            detail = ""
        raise BetaAccessError(detail or f"公测服务返回 HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise BetaAccessError(f"无法连接公测服务：{exc}") from exc
    except (ValueError, json.JSONDecodeError) as exc:
        raise BetaAccessError("公测服务返回了无法识别的数据") from exc
    if not isinstance(result, dict) or result.get("ok") is False:
        raise BetaAccessError(str(result.get("error", "公测请求失败")))
    return result


def beta_authenticate(email: str, password: str, *, register: bool = False) -> dict[str, str]:
    action = "register" if register else "login"
    result = cloud_json(
        f"/api/auth/desktop/{action}",
        method="POST",
        payload={"email": email, "password": password},
        authenticated=False,
    )
    token = str(result.get("token", "")).strip()
    user = result.get("user")
    if not token or not isinstance(user, dict):
        raise BetaAccessError("公测登录响应缺少会话信息")
    account = {"id": str(user.get("id", "")).strip(), "email": str(user.get("email", "")).strip()}
    if not account["id"] or not account["email"]:
        raise BetaAccessError("公测登录响应缺少账号信息")
    save_secret(BETA_SESSION_PATH, token)
    BETA_ACCOUNT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = BETA_ACCOUNT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(account, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, BETA_ACCOUNT_PATH)
    return account


def beta_logout() -> None:
    if beta_session_token():
        try:
            cloud_json("/api/auth/desktop/logout", method="POST", payload={})
        except BetaAccessError:
            pass
    for path in (BETA_SESSION_PATH, BETA_ACCOUNT_PATH):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
