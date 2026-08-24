from __future__ import annotations

import json
import hashlib
import re
import sqlite3
import unicodedata
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from core.config import DB_PATH, ensure_runtime_dirs


MIGRATIONS_DIR = Path(__file__).with_name("migrations")
MIGRATION_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    checksum TEXT NOT NULL,
    applied_at TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'interest',
    content TEXT NOT NULL,
    tags_json TEXT NOT NULL DEFAULT '[]',
    source TEXT NOT NULL DEFAULT 'obsidian',
    source_url TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS links (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL,
    target_id INTEGER,
    target_title TEXT NOT NULL,
    relation TEXT NOT NULL DEFAULT 'wikilink',
    strength REAL NOT NULL DEFAULT 1.0,
    explanation TEXT NOT NULL DEFAULT '',
    UNIQUE(source_id, target_title, relation)
);
CREATE TABLE IF NOT EXISTS cards (
    id INTEGER PRIMARY KEY,
    concept TEXT NOT NULL,
    frontier_title TEXT NOT NULL,
    frontier_url TEXT NOT NULL DEFAULT '',
    textbook_refs_json TEXT NOT NULL DEFAULT '[]',
    explanation TEXT NOT NULL,
    questions_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    task_type TEXT NOT NULL,
    concept TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL DEFAULT '{}',
    due_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    xp INTEGER NOT NULL DEFAULT 10,
    completed_at TEXT
);
CREATE TABLE IF NOT EXISTS feeds (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    url TEXT NOT NULL UNIQUE,
    enabled INTEGER NOT NULL DEFAULT 1,
    last_checked_at TEXT
);
CREATE TABLE IF NOT EXISTS activities (
    id INTEGER PRIMARY KEY,
    action TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    xp INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_kind ON notes(kind);
CREATE INDEX IF NOT EXISTS idx_tasks_due ON tasks(status, due_at);
"""


class GardenStore:
    def __init__(self, path: str | Path = DB_PATH):
        ensure_runtime_dirs()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(links)")}
            if "status" not in columns:
                conn.execute("ALTER TABLE links ADD COLUMN status TEXT NOT NULL DEFAULT 'accepted'")
            if "evidence_json" not in columns:
                conn.execute("ALTER TABLE links ADD COLUMN evidence_json TEXT NOT NULL DEFAULT '[]'")
            if "reviewed_at" not in columns:
                conn.execute("ALTER TABLE links ADD COLUMN reviewed_at TEXT")
            self._apply_migrations(conn)

    @contextmanager
    def connect(self):
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
    def _sql_literal(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    def _apply_migrations(self, conn: sqlite3.Connection) -> None:
        """Apply immutable numbered SQL migrations exactly once.

        A checksum mismatch is treated as an error instead of silently changing an
        already-created database. Schema changes therefore require a new numbered
        migration, which keeps local installations reproducible.
        """
        conn.executescript(MIGRATION_TABLE_SCHEMA)
        conn.commit()
        if not MIGRATIONS_DIR.is_dir():
            return

        seen_versions: set[int] = set()
        for migration_path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            match = re.fullmatch(r"(\d+)_([A-Za-z0-9_-]+)\.sql", migration_path.name)
            if not match:
                raise RuntimeError(f"Invalid migration filename: {migration_path.name}")
            version = int(match.group(1))
            if version in seen_versions:
                raise RuntimeError(f"Duplicate migration version: {version}")
            seen_versions.add(version)

            sql = migration_path.read_text(encoding="utf-8")
            checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()
            applied = conn.execute(
                "SELECT name,checksum FROM schema_migrations WHERE version=?", (version,)
            ).fetchone()
            if applied:
                if applied["name"] != migration_path.name or applied["checksum"] != checksum:
                    raise RuntimeError(
                        f"Migration {version} changed after it was applied; "
                        "create a new migration instead"
                    )
                continue

            name_literal = self._sql_literal(migration_path.name)
            checksum_literal = self._sql_literal(checksum)
            applied_at_literal = self._sql_literal(utc_now())
            script = (
                "BEGIN IMMEDIATE;\n"
                f"{sql}\n"
                "INSERT INTO schema_migrations(version,name,checksum,applied_at) "
                f"VALUES({version},{name_literal},{checksum_literal},{applied_at_literal});\n"
                "COMMIT;"
            )
            try:
                conn.executescript(script)
            except Exception:
                conn.rollback()
                raise

    def setting(self, key: str, default: Any = None) -> Any:
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return row["value"]

    def set_setting(self, key: str, value: Any) -> None:
        encoded = json.dumps(value, ensure_ascii=False)
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO settings(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, encoded),
            )

    def upsert_note(self, note: dict[str, Any]) -> tuple[int, bool]:
        now = utc_now()
        path = str(note["path"])
        with self.connect() as conn:
            old = conn.execute("SELECT id,content_hash FROM notes WHERE path=?", (path,)).fetchone()
            changed = not old or old["content_hash"] != note.get("content_hash", "")
            conn.execute(
                """INSERT INTO notes(path,title,kind,content,tags_json,source,source_url,content_hash,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(path) DO UPDATE SET title=excluded.title,kind=excluded.kind,
                   content=excluded.content,tags_json=excluded.tags_json,source=excluded.source,
                   source_url=excluded.source_url,content_hash=excluded.content_hash,updated_at=excluded.updated_at""",
                (
                    path,
                    note["title"],
                    note.get("kind", "interest"),
                    note.get("content", ""),
                    json.dumps(note.get("tags", []), ensure_ascii=False),
                    note.get("source", "obsidian"),
                    note.get("source_url", ""),
                    note.get("content_hash", ""),
                    now,
                    now,
                ),
            )
            row = conn.execute("SELECT id FROM notes WHERE path=?", (path,)).fetchone()
        return int(row["id"]), changed

    def replace_wikilinks(self, source_id: int, titles: Iterable[str]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM links WHERE source_id=? AND relation='wikilink'", (source_id,))
            for title in sorted(set(titles)):
                target = conn.execute("SELECT id FROM notes WHERE title=? LIMIT 1", (title,)).fetchone()
                conn.execute(
                    "INSERT OR IGNORE INTO links(source_id,target_id,target_title,relation) VALUES(?,?,?,'wikilink')",
                    (source_id, target["id"] if target else None, title),
                )

    def resolve_links(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """UPDATE links SET target_id=(SELECT id FROM notes WHERE notes.title=links.target_title LIMIT 1)
                   WHERE target_id IS NULL"""
            )

    def prune_obsidian_paths(self, existing_paths: set[str]) -> int:
        """Remove stale index rows only; the user's Vault files are never deleted."""
        with self.connect() as conn:
            rows = conn.execute("SELECT id,path FROM notes WHERE source='obsidian'").fetchall()
            stale_ids = [row["id"] for row in rows if row["path"] not in existing_paths]
            if not stale_ids:
                return 0
            marks = ",".join("?" for _ in stale_ids)
            conn.execute(f"DELETE FROM links WHERE source_id IN ({marks}) OR target_id IN ({marks})", (*stale_ids, *stale_ids))
            conn.execute(f"DELETE FROM notes WHERE id IN ({marks})", stale_ids)
            return len(stale_ids)

    def list_notes(self, kind: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        sql = "SELECT * FROM notes"
        args: list[Any] = []
        if kind:
            sql += " WHERE kind=?"
            args.append(kind)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        args.append(limit)
        with self.connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [self._decode_note(row) for row in rows]

    def get_note(self, note_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM notes WHERE id=?", (note_id,)).fetchone()
        return self._decode_note(row) if row else None

    @staticmethod
    def _decode_note(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["tags"] = json.loads(item.pop("tags_json") or "[]")
        return item

    def add_semantic_link(
        self, source_id: int, target_id: int, target_title: str, explanation: str, strength: float,
        evidence: list[str] | None = None, status: str = "proposed",
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO links(source_id,target_id,target_title,relation,strength,explanation,status,evidence_json)
                   VALUES(?,?,?,'semantic',?,?,?,?)
                   ON CONFLICT(source_id,target_title,relation) DO UPDATE SET
                   target_id=excluded.target_id,strength=excluded.strength,explanation=excluded.explanation,
                   evidence_json=excluded.evidence_json,
                   status=CASE WHEN links.status='rejected' THEN links.status ELSE excluded.status END""",
                (source_id, target_id, target_title, strength, explanation, status, json.dumps(evidence or [], ensure_ascii=False)),
            )

    def add_structural_link(
        self, source_id: int, target_id: int, target_title: str, relation: str = "contains",
        explanation: str = "", strength: float = 1.0,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO links(source_id,target_id,target_title,relation,strength,explanation,status,evidence_json)
                   VALUES(?,?,?,?,?,?,'accepted','[]')
                   ON CONFLICT(source_id,target_title,relation) DO UPDATE SET
                   target_id=excluded.target_id,strength=excluded.strength,explanation=excluded.explanation,status='accepted'""",
                (source_id, target_id, target_title, relation, strength, explanation),
            )

    def replace_agent_taxonomy_links(self, relations: list[dict[str, Any]]) -> int:
        """Atomically replace only hierarchy edges owned by the knowledge gardener."""
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM links WHERE relation='contains' AND explanation LIKE 'agent_taxonomy:%'"
            )
            for item in relations:
                conn.execute(
                    """INSERT INTO links(source_id,target_id,target_title,relation,strength,explanation,status,evidence_json)
                       VALUES(?,?,?,'contains',?,?,'accepted','[]')
                       ON CONFLICT(source_id,target_title,relation) DO UPDATE SET
                       target_id=excluded.target_id,strength=excluded.strength,
                       explanation=excluded.explanation,status='accepted'""",
                    (
                        int(item["parent_id"]), int(item["child_id"]), str(item["child_title"]),
                        float(item.get("confidence", 0.75)),
                        "agent_taxonomy:" + str(item.get("reason", "经概念角色与前置关系判断"))[:300],
                    ),
                )
        return len(relations)

    def clear_notes_by_source(self, source: str) -> int:
        with self.connect() as conn:
            ids = [row["id"] for row in conn.execute("SELECT id FROM notes WHERE source=?", (source,))]
            if not ids:
                return 0
            marks = ",".join("?" for _ in ids)
            conn.execute(f"DELETE FROM links WHERE source_id IN ({marks}) OR target_id IN ({marks})", (*ids, *ids))
            conn.execute(f"DELETE FROM notes WHERE id IN ({marks})", ids)
            return len(ids)

    def review_link(self, link_id: int, accepted: bool) -> bool:
        status = "accepted" if accepted else "rejected"
        with self.connect() as conn:
            row = conn.execute("SELECT id,target_title,status FROM links WHERE id=? AND relation='semantic'", (link_id,)).fetchone()
            if not row:
                return False
            conn.execute("UPDATE links SET status=?,reviewed_at=? WHERE id=?", (status, utc_now(), link_id))
            conn.execute(
                "INSERT INTO activities(action,detail,xp,created_at) VALUES('review_link',?,?,?)",
                (f"{status}:{row['target_title']}", 6, utc_now()),
            )
        return True

    def graph(self) -> dict[str, list[dict[str, Any]]]:
        with self.connect() as conn:
            nodes = []
            for row in conn.execute(
                """SELECT id,title,kind,tags_json,updated_at,activation_score,
                          access_count,last_accessed_at FROM notes
                   WHERE source!='pdf' AND kind NOT IN ('source','raw')
                   ORDER BY updated_at DESC LIMIT 300"""
            ):
                item = dict(row)
                item["tags"] = json.loads(item.pop("tags_json") or "[]")
                nodes.append(item)
            def canonical_title(value: str) -> str:
                normalized = unicodedata.normalize("NFKC", value).lower()
                normalized = re.sub(r"\s*[（(][^（）()]{1,80}[）)]\s*$", "", normalized)
                compact = re.sub(r"[\s·•_—–\-:：]+", "", normalized)
                aliases = {
                    "attention": "注意力机制", "snn": "脉冲神经网络",
                    "surrogategradient": "代理梯度", "backpropagation": "反向传播",
                    "heaviside": "阶跃函数",
                }
                return aliases.get(compact, compact)

            kind_priority = {"domain": 8, "moc": 7, "concept": 6, "knowledge": 5, "spark": 4, "bridge": 3, "interest": 2}
            grouped: dict[str, list[dict[str, Any]]] = {}
            for node in nodes:
                grouped.setdefault(canonical_title(node["title"]), []).append(node)
            alias: dict[int, int] = {}
            merged_nodes = []
            for group in grouped.values():
                representative = max(group, key=lambda item: (kind_priority.get(item["kind"], 0), item["updated_at"]))
                merged = dict(representative)
                merged["title"] = min((item["title"] for item in group), key=lambda value: (len(value), value))
                merged["tags"] = sorted({tag for item in group for tag in item["tags"]})
                merged["merged_ids"] = [item["id"] for item in group]
                merged_nodes.append(merged)
                alias.update({item["id"]: representative["id"] for item in group})
            nodes = merged_nodes
            valid_ids = {n["id"] for n in nodes}
            node_map = {n["id"]: n for n in nodes}
            edge_rows = conn.execute(
                "SELECT id,source_id,target_id,target_title,relation,strength,explanation,status,evidence_json FROM links WHERE target_id IS NOT NULL AND status!='rejected'"
            ).fetchall()
            edges = []
            for row in edge_rows:
                source_id = alias.get(row["source_id"])
                target_id = alias.get(row["target_id"])
                if source_id not in valid_ids or target_id not in valid_ids or source_id == target_id:
                    continue
                item = dict(row)
                item["source_id"] = source_id
                item["target_id"] = target_id
                item["target_title"] = node_map[target_id]["title"]
                item["evidence"] = json.loads(item.pop("evidence_json") or "[]")
                source_tags = set(node_map[item["source_id"]]["tags"])
                target_tags = set(node_map[item["target_id"]]["tags"])
                source_kind = node_map[item["source_id"]]["kind"]
                target_kind = node_map[item["target_id"]]["kind"]
                item["cross_domain"] = bool(
                    item["relation"] == "semantic" and (
                        (source_tags and target_tags and not source_tags.intersection(target_tags))
                        or source_kind != target_kind
                    )
                )
                duplicate = next((edge for edge in edges if edge["source_id"] == source_id and edge["target_id"] == target_id and edge["relation"] == item["relation"]), None)
                if duplicate:
                    if (item["status"] == "accepted", item["strength"]) > (duplicate["status"] == "accepted", duplicate["strength"]):
                        edges.remove(duplicate)
                        edges.append(item)
                else:
                    edges.append(item)
        return {"nodes": nodes, "edges": edges}

    def add_card(self, card: dict[str, Any]) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """INSERT INTO cards(concept,frontier_title,frontier_url,textbook_refs_json,explanation,questions_json,created_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (
                    card["concept"], card.get("frontier_title", "前沿材料"), card.get("frontier_url", ""),
                    json.dumps(card.get("textbook_refs", []), ensure_ascii=False), card["explanation"],
                    json.dumps(card.get("questions", []), ensure_ascii=False), utc_now(),
                ),
            )
            return int(cur.lastrowid)

    def purge_frontier(self, frontier_title: str) -> dict[str, int]:
        """Remove regenerated cards and their tasks while preserving source material."""
        with self.connect() as conn:
            card_rows = conn.execute(
                "SELECT id,concept FROM cards WHERE frontier_title=?", (frontier_title,)
            ).fetchall()
            card_ids = [row["id"] for row in card_rows]
            if not card_ids:
                return {"cards": 0, "tasks": 0}
            card_id_set = set(card_ids)
            concepts = {row["concept"] for row in card_rows}
            task_ids = []
            for row in conn.execute("SELECT id,concept,payload_json FROM tasks"):
                try:
                    payload = json.loads(row["payload_json"] or "{}")
                except json.JSONDecodeError:
                    payload = {}
                # Older quiz tasks did not carry card_id, so concept matching is
                # retained as a one-time backwards-compatible cleanup path.
                if payload.get("card_id") in card_id_set or row["concept"] in concepts:
                    task_ids.append(row["id"])
            if task_ids:
                marks = ",".join("?" for _ in task_ids)
                conn.execute(f"DELETE FROM tasks WHERE id IN ({marks})", task_ids)
            marks = ",".join("?" for _ in card_ids)
            conn.execute(f"DELETE FROM cards WHERE id IN ({marks})", card_ids)
        return {"cards": len(card_ids), "tasks": len(task_ids)}

    def list_cards(self, limit: int = 30) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM cards ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["textbook_refs"] = json.loads(item.pop("textbook_refs_json"))
            item["questions"] = json.loads(item.pop("questions_json"))
            result.append(item)
        return result

    def add_task(self, title: str, task_type: str, concept: str, payload: dict[str, Any], due_at: str, xp: int) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO tasks(title,task_type,concept,payload_json,due_at,xp) VALUES(?,?,?,?,?,?)",
                (title, task_type, concept, json.dumps(payload, ensure_ascii=False), due_at, xp),
            )
            return int(cur.lastrowid)

    def list_tasks(self, include_done: bool = False, limit: int = 100) -> list[dict[str, Any]]:
        where = "" if include_done else "WHERE status='pending'"
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM tasks {where} ORDER BY due_at ASC,id DESC LIMIT ?", (limit,)
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result

    def get_task(self, task_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if not row:
            return None
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json") or "{}")
        return item

    def record_review(
        self,
        task_id: int,
        quality: int,
        feedback: str,
        answer: str,
        recommended_interval_days: int | None = None,
    ) -> dict[str, Any]:
        """Finish one attempt and schedule the next spaced-repetition review."""
        quality = max(0, min(3, int(quality)))
        task = self.get_task(task_id)
        if not task or task["status"] != "pending":
            raise ValueError("复习任务不存在或已经完成")
        payload = dict(task["payload"])
        previous_interval = int(payload.get("interval_days", 0))
        review_count = int(payload.get("review_count", 0))
        if recommended_interval_days is not None:
            next_interval = max(1, min(90, int(recommended_interval_days)))
            next_count = review_count + int(quality > 0)
        elif quality == 0:
            next_interval = 1
            next_count = 0
        else:
            multipliers = {1: 1.5, 2: 2.5, 3: 4.0}
            next_interval = max(1, round((previous_interval or 1) * multipliers[quality]))
            next_interval = min(next_interval, 90)
            next_count = review_count + 1
        earned_xp = {0: 2, 1: 6, 2: task["xp"], 3: task["xp"] + 5}[quality]
        completed_at = utc_now()
        due = (datetime.now(timezone.utc) + timedelta(days=next_interval)).isoformat(timespec="seconds")
        payload.update({
            "review_count": next_count,
            "interval_days": next_interval,
            "last_answer": answer[:1000],
            "last_feedback": feedback[:1000],
            "last_quality": quality,
        })
        with self.connect() as conn:
            conn.execute(
                "UPDATE tasks SET status='done',completed_at=? WHERE id=?",
                (completed_at, task_id),
            )
            conn.execute(
                """INSERT INTO tasks(title,task_type,concept,payload_json,due_at,status,xp)
                   VALUES(?,?,?,?,?,'pending',?)""",
                (task["title"], task["task_type"], task["concept"], json.dumps(payload, ensure_ascii=False), due, task["xp"]),
            )
            conn.execute(
                "INSERT INTO activities(action,detail,xp,created_at) VALUES('review',?,?,?)",
                (f"{task['title']}｜质量 {quality}/3", earned_xp, completed_at),
            )
        return {
            "quality": quality,
            "feedback": feedback,
            "earned_xp": earned_xp,
            "next_interval_days": next_interval,
            "next_due": due,
        }

    def complete_task(self, task_id: int) -> bool:
        now = utc_now()
        with self.connect() as conn:
            row = conn.execute("SELECT title,xp,status FROM tasks WHERE id=?", (task_id,)).fetchone()
            if not row or row["status"] == "done":
                return False
            conn.execute("UPDATE tasks SET status='done',completed_at=? WHERE id=?", (now, task_id))
            conn.execute(
                "INSERT INTO activities(action,detail,xp,created_at) VALUES('complete_task',?,?,?)",
                (row["title"], row["xp"], now),
            )
        return True

    def add_activity(self, action: str, detail: str = "", xp: int = 0) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO activities(action,detail,xp,created_at) VALUES(?,?,?,?)",
                (action, detail, xp, utc_now()),
            )

    def stats(self) -> dict[str, Any]:
        with self.connect() as conn:
            notes = conn.execute("SELECT COUNT(*) n FROM notes").fetchone()["n"]
            concepts = conn.execute("SELECT COUNT(*) n FROM cards").fetchone()["n"]
            pending = conn.execute("SELECT COUNT(*) n FROM tasks WHERE status='pending'").fetchone()["n"]
            completed = conn.execute("SELECT COUNT(*) n FROM tasks WHERE status='done'").fetchone()["n"]
            xp = conn.execute("SELECT COALESCE(SUM(xp),0) n FROM activities").fetchone()["n"]
            links = conn.execute("SELECT COUNT(*) n FROM links").fetchone()["n"]
            proposed = conn.execute("SELECT COUNT(*) n FROM links WHERE status='proposed'").fetchone()["n"]
        level = int(xp // 100) + 1
        return {
            "notes": notes, "concepts": concepts, "pending_tasks": pending, "completed_tasks": completed,
            "xp": xp, "level": level, "level_progress": xp % 100, "links": links, "proposed_links": proposed,
        }

    def add_feed(self, name: str, url: str) -> int:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO feeds(name,url) VALUES(?,?) ON CONFLICT(url) DO UPDATE SET name=excluded.name,enabled=1",
                (name, url),
            )
            return int(conn.execute("SELECT id FROM feeds WHERE url=?", (url,)).fetchone()["id"])

    def list_feeds(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM feeds ORDER BY id DESC")]

    def touch_feed(self, feed_id: int) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE feeds SET last_checked_at=? WHERE id=?", (utc_now(), feed_id))

    def create_wechat_candidate(
        self,
        *,
        title: str,
        talker: str,
        time_range: str,
        contact: dict[str, Any],
        query: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        clean_messages = []
        seen_message_ids: set[str] = set()
        for index, item in enumerate(messages):
            if not str(item.get("content", "")).strip():
                continue
            source_id = str(item.get("source_message_id") or f"message-{index}")
            if source_id in seen_message_ids:
                continue
            seen_message_ids.add(source_id)
            clean_messages.append(item)
            if len(clean_messages) >= 300:
                break
        if not clean_messages:
            raise ValueError("请至少选择一条有内容的微信消息")
        candidate_id = str(uuid.uuid4())
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO wechat_candidates(
                       candidate_id,title,talker,time_range,contact_json,query_json,message_count,created_at
                   ) VALUES(?,?,?,?,?,?,?,?)""",
                (
                    candidate_id, title.strip() or "微信讨论候选", talker.strip(), time_range.strip(),
                    json.dumps(contact or {}, ensure_ascii=False), json.dumps(query or {}, ensure_ascii=False),
                    len(clean_messages), now,
                ),
            )
            for index, item in enumerate(clean_messages):
                source_id = str(item.get("source_message_id") or f"message-{index}")
                source_json = json.dumps(item.get("source") or {}, ensure_ascii=False)
                if len(source_json) > 100_000:
                    source_json = json.dumps({
                        "truncated": True, "source_message_id": source_id,
                        "reason": "原始消息元数据超过知识花园单条保存上限",
                    }, ensure_ascii=False)
                conn.execute(
                    """INSERT INTO wechat_candidate_messages(
                           candidate_id,source_message_id,sender,sent_at,content,message_type,source_json
                       ) VALUES(?,?,?,?,?,?,?)""",
                    (
                        candidate_id, source_id, str(item.get("sender", ""))[:200],
                        str(item.get("sent_at", ""))[:100], str(item.get("content", ""))[:10000],
                        str(item.get("message_type", "text"))[:80],
                        source_json,
                    ),
                )
            event_id = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO learning_events(
                       event_id,surface,event_type,source_kind,payload_json,created_at
                   ) VALUES(?,?,?,?,?,?)""",
                (
                    event_id, "wechat_import", "wechat_candidate_created", "explicit",
                    json.dumps({"candidate_id": candidate_id, "message_count": len(clean_messages)}, ensure_ascii=False), now,
                ),
            )
        return self.get_wechat_candidate(candidate_id) or {}

    def list_wechat_candidates(self, limit: int = 30) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM wechat_candidates ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 100)),)
            ).fetchall()
        return [self._decode_wechat_candidate(dict(row)) for row in rows]

    def get_wechat_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM wechat_candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
            if not row:
                return None
            messages = conn.execute(
                """SELECT source_message_id,sender,sent_at,content,message_type
                   FROM wechat_candidate_messages WHERE candidate_id=? ORDER BY id""",
                (candidate_id,),
            ).fetchall()
        result = self._decode_wechat_candidate(dict(row))
        result["messages"] = [dict(item) for item in messages]
        return result

    @staticmethod
    def _decode_wechat_candidate(item: dict[str, Any]) -> dict[str, Any]:
        for key in ("contact_json", "query_json"):
            try:
                item[key[:-5]] = json.loads(item.pop(key))
            except (json.JSONDecodeError, TypeError):
                item[key[:-5]] = {}
        return item

    def review_wechat_candidate(self, candidate_id: str, accepted: bool, raw_path: str = "") -> dict[str, Any]:
        status = "accepted" if accepted else "rejected"
        now = utc_now()
        with self.connect() as conn:
            row = conn.execute("SELECT status FROM wechat_candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
            if not row:
                raise ValueError("微信候选不存在")
            if row["status"] != "pending":
                raise ValueError("这条微信候选已经审核过")
            conn.execute(
                "UPDATE wechat_candidates SET status=?,raw_path=?,reviewed_at=? WHERE candidate_id=?",
                (status, raw_path, now, candidate_id),
            )
            conn.execute(
                """INSERT INTO learning_events(
                       event_id,surface,event_type,source_kind,payload_json,created_at
                   ) VALUES(?,?,?,?,?,?)""",
                (
                    str(uuid.uuid4()), "wechat_import", f"wechat_candidate_{status}", "explicit",
                    json.dumps({"candidate_id": candidate_id, "raw_path": raw_path}, ensure_ascii=False), now,
                ),
            )
        return self.get_wechat_candidate(candidate_id) or {}
