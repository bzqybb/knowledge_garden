from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from core.config import DATA_DIR, DB_PATH
from core.storage import GardenStore, utc_now


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


AUTH_REQUIRED = env_flag("GARDEN_AUTH_REQUIRED", False)
ALLOW_SIGNUP = env_flag("GARDEN_ALLOW_SIGNUP", False)


class TenantGardenStore:
    """Thread-scoped GardenStore router.

    Existing agent code can keep receiving a GardenStore-shaped object while each
    authenticated request is routed to a physically separate SQLite database.
    This avoids relying on application code to remember a user_id filter.
    """

    def __init__(self, default_path: str | Path = DB_PATH, users_dir: str | Path | None = None):
        self.default_store = GardenStore(default_path)
        self.users_dir = Path(users_dir or (DATA_DIR / "users"))
        self.users_dir.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._stores: dict[str, GardenStore] = {}
        self._lock = threading.Lock()

    def bind_user(self, user_id: str | None) -> None:
        self._local.user_id = (user_id or "local").strip() or "local"

    def current_user_id(self) -> str:
        return str(getattr(self._local, "user_id", "local"))

    @contextmanager
    def using_user(self, user_id: str | None) -> Iterator[GardenStore]:
        previous = self.current_user_id()
        self.bind_user(user_id)
        try:
            yield self.current_store()
        finally:
            self.bind_user(previous)

    def current_store(self) -> GardenStore:
        user_id = self.current_user_id()
        if user_id == "local":
            return self.default_store
        if not re.fullmatch(r"[a-f0-9-]{32,36}", user_id):
            raise ValueError("无效的用户空间")
        with self._lock:
            store = self._stores.get(user_id)
            if store is None:
                store = GardenStore(self.users_dir / user_id / "garden.db")
                self._stores[user_id] = store
            return store

    @property
    def path(self) -> Path:
        return self.current_store().path

    def __getattr__(self, name: str) -> Any:
        return getattr(self.current_store(), name)


class AuthRegistry:
    """Small password/session registry kept outside all tenant knowledge stores."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or (DATA_DIR / "auth" / "accounts.db"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS users(
                    user_id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    password_salt BLOB NOT NULL,
                    password_hash BLOB NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS auth_sessions(
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS improvement_preferences(
                    user_id TEXT PRIMARY KEY,
                    consent INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS interaction_candidates(
                    candidate_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    surface TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    explicit_feedback TEXT NOT NULL DEFAULT '',
                    helpful INTEGER,
                    dataset_split TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(user_id,request_id,surface),
                    FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_candidates_queue
                    ON interaction_candidates(dataset_split,status,created_at);
                CREATE TABLE IF NOT EXISTS judge_reviews(
                    review_id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL,
                    judge_name TEXT NOT NULL,
                    judge_version TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    score REAL,
                    findings_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    UNIQUE(candidate_id,judge_name,judge_version),
                    FOREIGN KEY(candidate_id) REFERENCES interaction_candidates(candidate_id) ON DELETE CASCADE
                );
                """
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _normalize_email(email: str) -> str:
        value = email.strip().casefold()
        if len(value) > 254 or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value):
            raise ValueError("请输入有效邮箱")
        return value

    @staticmethod
    def _password_hash(password: str, salt: bytes) -> bytes:
        return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000)

    def user_count(self) -> int:
        with self.connect() as conn:
            return int(conn.execute("SELECT COUNT(*) n FROM users").fetchone()["n"])

    def signup_allowed(self) -> bool:
        return ALLOW_SIGNUP or self.user_count() == 0

    def register(self, email: str, password: str) -> tuple[dict[str, str], str]:
        if not self.signup_allowed():
            raise ValueError("当前站点未开放注册")
        email = self._normalize_email(email)
        if len(password) < 10:
            raise ValueError("密码至少需要 10 个字符")
        salt = secrets.token_bytes(16)
        user_id = str(uuid4())
        try:
            with self.connect() as conn:
                conn.execute(
                    "INSERT INTO users(user_id,email,password_salt,password_hash,created_at) VALUES(?,?,?,?,?)",
                    (user_id, email, salt, self._password_hash(password, salt), utc_now()),
                )
                conn.execute(
                    "INSERT INTO improvement_preferences(user_id,consent,updated_at) VALUES(?,0,?)",
                    (user_id, utc_now()),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("这个邮箱已经注册") from exc
        return {"id": user_id, "email": email}, self.create_session(user_id)

    def login(self, email: str, password: str) -> tuple[dict[str, str], str]:
        email = self._normalize_email(email)
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email=? AND status='active'", (email,)
            ).fetchone()
        if not row or not hmac.compare_digest(
            bytes(row["password_hash"]), self._password_hash(password, bytes(row["password_salt"]))
        ):
            raise ValueError("邮箱或密码不正确")
        return {"id": str(row["user_id"]), "email": str(row["email"])}, self.create_session(str(row["user_id"]))

    def create_session(self, user_id: str, days: int = 14) -> str:
        token = secrets.token_urlsafe(32)
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        now = datetime.now(timezone.utc)
        expires = now + timedelta(days=max(1, min(days, 30)))
        with self.connect() as conn:
            conn.execute("DELETE FROM auth_sessions WHERE expires_at<=?", (now.isoformat(),))
            conn.execute(
                "INSERT INTO auth_sessions(token_hash,user_id,expires_at,created_at,last_seen_at) VALUES(?,?,?,?,?)",
                (digest, user_id, expires.isoformat(), now.isoformat(), now.isoformat()),
            )
        return token

    def user_for_token(self, token: str) -> dict[str, str] | None:
        if not token:
            return None
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            row = conn.execute(
                """SELECT u.user_id,u.email FROM auth_sessions s
                   JOIN users u ON u.user_id=s.user_id
                   WHERE s.token_hash=? AND s.expires_at>? AND u.status='active'""",
                (digest, now),
            ).fetchone()
            if row:
                conn.execute("UPDATE auth_sessions SET last_seen_at=? WHERE token_hash=?", (now, digest))
        return {"id": str(row["user_id"]), "email": str(row["email"])} if row else None

    def logout(self, token: str) -> None:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest() if token else ""
        if digest:
            with self.connect() as conn:
                conn.execute("DELETE FROM auth_sessions WHERE token_hash=?", (digest,))

    def consent(self, user_id: str) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT consent FROM improvement_preferences WHERE user_id=?", (user_id,)
            ).fetchone()
        return bool(row and row["consent"])

    def set_consent(self, user_id: str, consent: bool) -> dict[str, Any]:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO improvement_preferences(user_id,consent,updated_at) VALUES(?,?,?)
                   ON CONFLICT(user_id) DO UPDATE SET consent=excluded.consent,updated_at=excluded.updated_at""",
                (user_id, int(consent), utc_now()),
            )
        return self.improvement_status(user_id)

    @staticmethod
    def _sanitize(text: str) -> str:
        value = str(text)
        value = re.sub(r"(?i)\b(?:sk|ak)-[A-Za-z0-9_-]{12,}\b", "[密钥已移除]", value)
        value = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~-]{12,}", "Bearer [令牌已移除]", value)
        value = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[邮箱已移除]", value)
        value = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[手机号已移除]", value)
        value = re.sub(r"(?i)[A-Z]:\\Users\\[^\\\s]+", "[用户目录]", value)
        return value

    def capture_interaction(
        self,
        *,
        user_id: str,
        request_id: str,
        surface: str,
        question: str,
        answer: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if user_id == "local" or not self.consent(user_id):
            return {"captured": False, "reason": "not_consented"}
        candidate_id = f"candidate-{uuid4().hex}"
        digest = hashlib.sha256(f"{user_id}:{request_id}".encode("utf-8")).hexdigest()
        split = "holdout" if int(digest[:8], 16) % 100 < 15 else "development"
        status = "sealed" if split == "holdout" else "pending"
        safe_metadata = {
            "model": os.getenv("GARDEN_MODEL", ""),
            "release": os.getenv("GARDEN_RELEASE_VERSION", "development"),
            "routing_target": str((metadata or {}).get("routing_target") or ""),
            "evidence_layer": str((metadata or {}).get("evidence_layer") or ""),
            "latency_seconds": (metadata or {}).get("latency_seconds"),
        }
        with self.connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO interaction_candidates(
                       candidate_id,user_id,request_id,surface,question,answer,metadata_json,
                       dataset_split,status,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    candidate_id, user_id, request_id, surface,
                    self._sanitize(question)[:10_000], self._sanitize(answer)[:50_000],
                    json.dumps(safe_metadata, ensure_ascii=False), split, status, utc_now(),
                ),
            )
            inserted = bool(conn.execute("SELECT changes() n").fetchone()["n"])
        return {"captured": inserted, "candidate_id": candidate_id if inserted else None, "split": split}

    def record_candidate_feedback(
        self, *, user_id: str, request_id: str, helpful: bool, note: str = ""
    ) -> None:
        if user_id == "local" or not self.consent(user_id):
            return
        with self.connect() as conn:
            conn.execute(
                """UPDATE interaction_candidates SET helpful=?,explicit_feedback=?
                   WHERE user_id=? AND request_id=?""",
                (int(helpful), self._sanitize(note)[:1000], user_id, request_id),
            )

    def improvement_status(self, user_id: str) -> dict[str, Any]:
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT dataset_split,status,COUNT(*) n FROM interaction_candidates
                   WHERE user_id=? GROUP BY dataset_split,status""",
                (user_id,),
            ).fetchall()
        return {
            "consent": self.consent(user_id),
            "counts": [dict(row) for row in rows],
            "policy": "只有明确授权且完成脱敏的问题才进入候选集；保留集不会用于提示词优化。",
        }
