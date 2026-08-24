from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from core.context import GardenContext
from core.storage import GardenStore, utc_now


MASTERY_DIMENSIONS = ("recognition", "explanation", "application", "transfer")


def parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def exponential_retention(elapsed_days: float, stability_days: float) -> float:
    """Approximate retention with an exponential forgetting curve.

    Stability is personalized by later review outcomes. This is a scheduling
    estimate, not a psychological diagnosis or a claim that one universal curve
    describes every learner.
    """
    if elapsed_days <= 0:
        return 1.0
    return max(0.0, min(1.0, math.exp(-elapsed_days / max(0.25, stability_days))))


def note_activation(note: dict[str, Any], *, now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    last_accessed = parse_utc(note.get("last_accessed_at")) or parse_utc(note.get("updated_at")) or now
    elapsed = max(0.0, (now - last_accessed).total_seconds() / 86_400)
    stability = float(note.get("stability_days", 14.0) or 14.0)
    retention = exponential_retention(elapsed, stability)
    importance = float(note.get("base_importance", 0.5) or 0.5)
    usage_bonus = min(0.2, math.log1p(int(note.get("access_count", 0) or 0)) * 0.04)
    cached = float(note.get("activation_score", 0.5) or 0.5)
    value = 0.55 * importance * retention + 0.25 * cached + usage_bonus
    return round(max(0.05, min(1.0, value)), 4)


class LearningMemoryService:
    """Evidence-first experience memory and knowledge-evolution service."""

    def __init__(self, store: GardenStore):
        self.store = store

    @staticmethod
    def new_id(prefix: str) -> str:
        return f"{prefix}-{uuid4().hex}"

    def begin_turn(
        self, question: str, session_id: str | None = None, *, capability: str = "gardener_chat",
    ) -> dict[str, str]:
        session_id = (session_id or "").strip() or self.new_id("session")
        request_id = self.new_id("request")
        message_id = self.new_id("message")
        now = utc_now()
        with self.store.connect() as conn:
            conn.execute(
                """INSERT INTO sessions(session_id,title,updated_at)
                   VALUES(?,?,?)
                   ON CONFLICT(session_id) DO UPDATE SET updated_at=excluded.updated_at,status='active'""",
                (session_id, question.strip()[:80], now),
            )
            conn.execute(
                """INSERT INTO session_messages(
                       message_id,session_id,request_id,role,capability,content,created_at
                   ) VALUES(?,?,?,?,?,?,?)""",
                (message_id, session_id, request_id, "user", capability, question.strip(), now),
            )
        return {"session_id": session_id, "request_id": request_id, "message_id": message_id}

    def complete_turn(self, context: GardenContext, result: dict[str, Any]) -> dict[str, Any]:
        assistant_id = self.new_id("message")
        now = utc_now()
        with self.store.connect() as conn:
            conn.execute(
                """INSERT INTO session_messages(
                       message_id,session_id,request_id,role,capability,content,metadata_json,created_at
                   ) VALUES(?,?,?,?,?,?,?,?)""",
                (
                    assistant_id, context.session_id, context.request_id, "assistant",
                    context.active_capability, str(result.get("answer", "")),
                    json.dumps({
                        "evidence_layer": result.get("evidence_layer", "none"),
                        "citation_ids": [item.get("id") for item in result.get("citations", [])],
                        "personalization": result.get("personalization", {}),
                    }, ensure_ascii=False),
                    now,
                ),
            )
            conn.execute(
                "UPDATE sessions SET updated_at=? WHERE session_id=?",
                (now, context.session_id),
            )

        intent = result.get("intent") or {}
        concepts = [str(item) for item in intent.get("concepts", []) if str(item).strip()]
        event_id = self.record_event(
            surface="gardener_chat",
            event_type="question_asked",
            source_kind="observed",
            session_id=context.session_id,
            message_id=context.current_message.message_id,
            concepts=concepts,
            payload={
                "primary_intent": intent.get("primary_intent", "unknown"),
                "task_demand": intent.get("task_demand", "understand"),
                "teaching_move": (result.get("teaching_strategy") or {}).get("teaching_move"),
                "evidence_sufficient": bool((result.get("evidence_review") or {}).get("sufficient")),
                "quality_passed": bool((result.get("quality_review") or {}).get("passed", True)),
                "request_id": context.request_id,
            },
        )
        citation_ids = [
            int(item["id"]) for item in result.get("citations", [])
            if str(item.get("id", "")).isdigit()
        ]
        if citation_ids:
            self.record_knowledge_access(
                citation_ids,
                session_id=context.session_id,
                message_id=assistant_id,
                reason="gardener_answer_evidence",
            )
        reflection = self.reflect(force=False)
        return {
            "assistant_message_id": assistant_id,
            "question_event_id": event_id,
            "reflection": reflection,
        }

    def record_event(
        self,
        *,
        surface: str,
        event_type: str,
        source_kind: str,
        payload: dict[str, Any] | None = None,
        concepts: list[str] | None = None,
        concept_note_ids: list[int] | None = None,
        session_id: str | None = None,
        message_id: str | None = None,
    ) -> str:
        event_id = self.new_id("event")
        with self.store.connect() as conn:
            conn.execute(
                """INSERT INTO learning_events(
                       event_id,session_id,message_id,surface,event_type,source_kind,payload_json
                   ) VALUES(?,?,?,?,?,?,?)""",
                (
                    event_id, session_id, message_id, surface, event_type, source_kind,
                    json.dumps(payload or {}, ensure_ascii=False),
                ),
            )
            pairs: dict[str, int | None] = {}
            for title in concepts or []:
                title = title.strip()
                if not title:
                    continue
                note = conn.execute(
                    "SELECT id FROM notes WHERE title=? ORDER BY kind='concept' DESC LIMIT 1", (title,)
                ).fetchone()
                pairs[title] = int(note["id"]) if note else None
            for note_id in concept_note_ids or []:
                note = conn.execute("SELECT id,title FROM notes WHERE id=?", (note_id,)).fetchone()
                if note:
                    pairs[str(note["title"])] = int(note["id"])
            for concept_key, note_id in pairs.items():
                conn.execute(
                    """INSERT OR IGNORE INTO event_concepts(
                           event_id,concept_key,concept_note_id
                       ) VALUES(?,?,?)""",
                    (event_id, concept_key, note_id),
                )
        return event_id

    def session_history(self, session_id: str, limit: int = 10) -> list[dict[str, Any]]:
        with self.store.connect() as conn:
            rows = conn.execute(
                """SELECT message_id,role,content,capability,created_at
                   FROM session_messages WHERE session_id=?
                   ORDER BY created_at DESC,rowid DESC LIMIT ?""",
                (session_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def active_memory_context(
        self,
        concepts: list[str] | None = None,
        *,
        surface: str = "gardener_chat",
        task_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return only scoped, time-adjusted learner hypotheses with provenance.

        Memory claims are hypotheses, not profile labels.  A claim from another
        surface/concept/task is excluded instead of being made globally relevant
        by keyword overlap.  Every returned claim carries the evidence a user can
        inspect and later contradict.
        """
        concepts = [item for item in (concepts or []) if item]
        concept_keys = {item.strip().casefold() for item in concepts if item.strip()}
        task_key_set = {str(item).strip().casefold() for item in (task_keys or []) if str(item).strip()}
        with self.store.connect() as conn:
            claim_rows = [dict(row) for row in conn.execute(
                """SELECT claim_id,layer,dimension,scope_type,scope_key,claim_text,confidence
                          ,source_kind,updated_at
                   FROM memory_claims
                   WHERE owner_id='local' AND status='active'
                     AND (valid_until IS NULL OR valid_until>?)
                   ORDER BY layer DESC,confidence DESC,updated_at DESC LIMIT 12""",
                (utc_now(),),
            )]
            mastery = []
            if concepts:
                marks = ",".join("?" for _ in concepts)
                rows = conn.execute(
                    f"""SELECT * FROM concept_mastery
                        WHERE owner_id='local' AND concept_key IN ({marks})""",
                    concepts,
                ).fetchall()
                mastery = [self._mastery_snapshot(dict(row)) for row in rows]
        now = datetime.now(timezone.utc)
        claims = []
        for claim in claim_rows:
            scope_type = str(claim.get("scope_type") or "global").casefold()
            scope_key = str(claim.get("scope_key") or "").strip().casefold()
            if scope_type == "surface" and scope_key not in {"", surface.casefold()}:
                continue
            if scope_type == "concept" and scope_key not in concept_keys:
                continue
            if scope_type == "task" and scope_key not in task_key_set:
                continue
            if scope_type not in {"global", "surface", "concept", "task"}:
                continue
            updated = parse_utc(claim.get("updated_at")) or now
            elapsed = max(0.0, (now - updated).total_seconds() / 86_400)
            if claim["source_kind"] == "explicit":
                stability = 365.0
            else:
                stability = 180.0 if int(claim["layer"]) == 3 else 45.0
            effective = float(claim["confidence"]) * exponential_retention(elapsed, stability)
            claim["effective_confidence"] = round(effective, 4)
            with self.store.connect() as conn:
                evidence_rows = conn.execute(
                    """SELECT e.id,e.event_id,e.source_claim_id,e.message_id,e.relation,e.weight,
                              ev.surface,ev.event_type,ev.source_kind AS event_source_kind,
                              ev.created_at AS event_created_at,ev.payload_json,
                              msg.content AS message_content,
                              source.claim_text AS source_claim_text
                       FROM memory_claim_evidence e
                       LEFT JOIN learning_events ev ON ev.event_id=e.event_id
                       LEFT JOIN session_messages msg ON msg.message_id=e.message_id
                       LEFT JOIN memory_claims source ON source.claim_id=e.source_claim_id
                       WHERE e.claim_id=? ORDER BY e.created_at DESC,e.id DESC LIMIT 8""",
                    (claim["claim_id"],),
                ).fetchall()
            evidence = []
            support_weight = 0.0
            contradiction_weight = 0.0
            for row in evidence_rows:
                item = dict(row)
                weight = float(item.get("weight") or 0.0)
                if item.get("relation") == "contradicts":
                    contradiction_weight += weight
                else:
                    support_weight += weight
                payload = json.loads(item.get("payload_json") or "{}")
                observation = (
                    str(item.get("message_content") or "").strip()
                    or str(payload.get("observation") or payload.get("feedback_note") or "").strip()
                    or str(item.get("source_claim_text") or "").strip()
                    or str(item.get("event_type") or "学习事件")
                )
                evidence.append({
                    "evidence_id": (
                        item.get("event_id") or item.get("message_id")
                        or item.get("source_claim_id") or f"evidence-{item['id']}"
                    ),
                    "relation": item.get("relation") or "supports",
                    "weight": weight,
                    "observation": observation[:240],
                    "surface": item.get("surface") or "memory",
                    "source_kind": item.get("event_source_kind") or claim.get("source_kind"),
                    "created_at": item.get("event_created_at") or claim.get("updated_at"),
                })
            claim["evidence"] = evidence
            claim["support_weight"] = round(support_weight, 3)
            claim["contradiction_weight"] = round(contradiction_weight, 3)
            if effective >= 0.35:
                claims.append(claim)
        claims.sort(key=lambda item: (item["layer"], item["effective_confidence"]), reverse=True)
        return {"claims": claims[:12], "concept_mastery": mastery}

    def record_personalization_feedback(
        self,
        *,
        request_id: str,
        helpful: bool,
        feedback_note: str = "",
    ) -> dict[str, Any]:
        """Persist an explicit correction and update only hypotheses used this turn."""
        request_id = request_id.strip()
        if not request_id:
            raise ValueError("缺少本轮 request_id")
        with self.store.connect() as conn:
            prior = conn.execute(
                """SELECT event_id FROM learning_events
                   WHERE event_type='personalization_feedback'
                     AND json_extract(payload_json,'$.request_id')=? LIMIT 1""",
                (request_id,),
            ).fetchone()
            assistant = conn.execute(
                """SELECT session_id,message_id,metadata_json FROM session_messages
                   WHERE request_id=? AND role='assistant'
                   ORDER BY created_at DESC,rowid DESC LIMIT 1""",
                (request_id,),
            ).fetchone()
        if prior:
            return {"recorded": False, "reason": "本轮已经反馈过", "event_id": prior["event_id"]}
        if not assistant:
            raise ValueError("没有找到这轮园丁回答")
        metadata = json.loads(assistant["metadata_json"] or "{}")
        plan = metadata.get("personalization") if isinstance(metadata.get("personalization"), dict) else {}
        used_claim_ids = [str(value) for value in plan.get("applied_claim_ids", []) if str(value)]
        strategy = str(plan.get("strategy_summary") or "本轮讲解方式").strip()
        task_key = str(plan.get("task_key") or "general").strip()
        payload = {
            "request_id": request_id,
            "helpful": bool(helpful),
            "feedback_note": feedback_note.strip()[:500],
            "strategy_summary": strategy,
            "applied_claim_ids": used_claim_ids,
            "observation": feedback_note.strip() or ("用户确认本轮讲解方式有帮助" if helpful else "用户明确否定本轮讲解方式"),
        }
        event_id = self.record_event(
            surface="gardener_chat",
            event_type="personalization_feedback",
            source_kind="explicit",
            session_id=assistant["session_id"],
            # A click is a new explicit user observation, not evidence that the
            # assistant's answer text itself expressed this preference.
            message_id=None,
            payload=payload,
        )
        changed: list[dict[str, Any]] = []
        with self.store.connect() as conn:
            for claim_id in used_claim_ids:
                claim = conn.execute(
                    "SELECT confidence,status FROM memory_claims WHERE claim_id=?",
                    (claim_id,),
                ).fetchone()
                if not claim:
                    continue
                relation = "supports" if helpful else "contradicts"
                conn.execute(
                    """INSERT OR IGNORE INTO memory_claim_evidence(
                           claim_id,event_id,relation,weight
                       ) VALUES(?,?,?,1.0)""",
                    (claim_id, event_id, relation),
                )
                confidence = float(claim["confidence"])
                confidence = min(0.98, confidence + 0.08) if helpful else max(0.05, confidence - 0.28)
                status = claim["status"]
                if not helpful and confidence < 0.55:
                    status = "candidate"
                conn.execute(
                    "UPDATE memory_claims SET confidence=?,status=?,updated_at=? WHERE claim_id=?",
                    (confidence, status, utc_now(), claim_id),
                )
                changed.append({"claim_id": claim_id, "confidence": round(confidence, 3), "status": status})

        # A positive confirmation may become a narrowly scoped, explicit teaching
        # hypothesis even when this turn intentionally used the standard fallback.
        created_claim_id = None
        if helpful and not used_claim_ids and strategy and strategy != "标准讲解（没有足够个性化证据）":
            claim_text = f"在 {task_key} 类问题中，用户确认“{strategy}”有帮助"
            self._upsert_claim(
                layer=2,
                dimension="teaching_preference",
                scope_type="task",
                scope_key=task_key,
                claim_text=claim_text,
                source_kind="explicit",
                confidence=0.92,
                status="active",
                event_ids=[event_id],
            )
            with self.store.connect() as conn:
                row = conn.execute(
                    """SELECT claim_id FROM memory_claims WHERE dimension='teaching_preference'
                       AND scope_type='task' AND scope_key=? AND claim_text=?
                       ORDER BY updated_at DESC LIMIT 1""",
                    (task_key, claim_text),
                ).fetchone()
            created_claim_id = row["claim_id"] if row else None
        return {
            "recorded": True,
            "event_id": event_id,
            "helpful": bool(helpful),
            "updated_claims": changed,
            "created_claim_id": created_claim_id,
            "message": (
                "已把这次确认作为窄范围教学证据" if helpful
                else "已写入反证并降低本轮相关假设的置信度"
            ),
        }

    def reflect(self, *, force: bool = False, min_events: int = 6) -> dict[str, Any]:
        last_reflection = self.store.setting("memory.last_reflection_at", "")
        with self.store.connect() as conn:
            if last_reflection:
                rows = conn.execute(
                    """SELECT * FROM learning_events
                       WHERE created_at>? AND event_type!='memory_reflection' AND surface!='inspiration'
                       ORDER BY created_at,event_id""",
                    (last_reflection,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM learning_events
                       WHERE event_type!='memory_reflection' AND surface!='inspiration'
                       ORDER BY created_at,event_id LIMIT 200"""
                ).fetchall()
        events = [dict(row) for row in rows]
        if len(events) < min_events and not force:
            return {"triggered": False, "events": len(events), "l2_created": 0, "l3_created": 0}

        proposals: list[dict[str, Any]] = []
        candidate_groups: dict[tuple[str, str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            payload = json.loads(event["payload_json"] or "{}")
            candidate = payload.get("memory_candidate")
            if not isinstance(candidate, dict):
                continue
            claim_text = str(candidate.get("claim_text", "")).strip()
            dimension = str(candidate.get("dimension", "")).strip()
            if not claim_text or not dimension:
                continue
            key = (
                dimension,
                str(candidate.get("scope_type", "surface")),
                str(candidate.get("scope_key", event["surface"])),
                claim_text,
                2,
            )
            candidate_groups[key].append(event)

        question_events = []
        for event in events:
            if event["event_type"] != "question_asked":
                continue
            payload = json.loads(event["payload_json"] or "{}")
            intent = str(payload.get("primary_intent", "unknown"))
            if intent != "unknown":
                question_events.append((intent, event))
        if len(question_events) >= 3:
            counts = Counter(intent for intent, _ in question_events)
            intent, count = counts.most_common(1)[0]
            if count / len(question_events) >= 0.6:
                intent_labels = {
                    "define": "定义辨析", "explain_mechanism": "机制解释", "apply": "实际应用",
                    "compare": "比较分析", "evaluate": "评价判断", "design": "整合设计",
                }
                claim_text = f"近期在问园丁场景中多次提出{intent_labels.get(intent, intent)}类问题"
                key = ("question_pattern", "surface", "gardener_chat", claim_text, 2)
                candidate_groups[key].extend(event for value, event in question_events if value == intent)

        created_l2 = 0
        activated_l2 = 0
        for (dimension, scope_type, scope_key, claim_text, layer), evidence in candidate_groups.items():
            explicit = any(item["source_kind"] == "explicit" for item in evidence)
            sessions = {item["session_id"] for item in evidence if item["session_id"]}
            count = len({item["event_id"] for item in evidence})
            if not explicit and count < 3:
                continue
            active = explicit or (count >= 5 and len(sessions) >= 2)
            confidence = 0.95 if explicit else min(0.9, 0.45 + count * 0.07 + min(0.1, len(sessions) * 0.03))
            was_created, was_activated = self._upsert_claim(
                layer=layer,
                dimension=dimension,
                scope_type=scope_type,
                scope_key=scope_key,
                claim_text=claim_text,
                source_kind="explicit" if explicit else "observed",
                confidence=confidence,
                status="active" if active else "candidate",
                event_ids=[item["event_id"] for item in evidence],
            )
            created_l2 += int(was_created)
            activated_l2 += int(was_activated)

        created_l3 = self._promote_cross_context_patterns()
        reflection_at = utc_now()
        self.record_event(
            surface="memory_system",
            event_type="memory_reflection",
            source_kind="system",
            payload={
                "window_start": last_reflection or None,
                "events_reviewed": len(events),
                "l2_created": created_l2,
                "l2_activated": activated_l2,
                "l3_created": created_l3,
            },
        )
        self.store.set_setting("memory.last_reflection_at", reflection_at)
        return {
            "triggered": True,
            "events": len(events),
            "l2_created": created_l2,
            "l2_activated": activated_l2,
            "l3_created": created_l3,
        }

    def _upsert_claim(
        self,
        *,
        layer: int,
        dimension: str,
        scope_type: str,
        scope_key: str,
        claim_text: str,
        source_kind: str,
        confidence: float,
        status: str,
        event_ids: list[str],
    ) -> tuple[bool, bool]:
        with self.store.connect() as conn:
            row = conn.execute(
                """SELECT claim_id,status FROM memory_claims
                   WHERE owner_id='local' AND layer=? AND dimension=? AND scope_type=?
                     AND scope_key=? AND claim_text=? AND status IN ('candidate','active')
                   LIMIT 1""",
                (layer, dimension, scope_type, scope_key, claim_text),
            ).fetchone()
            created = row is None
            previous_status = row["status"] if row else None
            claim_id = row["claim_id"] if row else self.new_id("claim")
            if row:
                conn.execute(
                    """UPDATE memory_claims SET confidence=max(confidence,?),
                       status=CASE WHEN status='active' THEN 'active' ELSE ? END,updated_at=?
                       WHERE claim_id=?""",
                    (confidence, status, utc_now(), claim_id),
                )
            else:
                conn.execute(
                    """INSERT INTO memory_claims(
                           claim_id,layer,dimension,scope_type,scope_key,claim_text,
                           source_kind,confidence,status,created_by
                       ) VALUES(?,?,?,?,?,?,?,?,?,'agent')""",
                    (
                        claim_id, layer, dimension, scope_type, scope_key, claim_text,
                        source_kind, confidence, status,
                    ),
                )
            for event_id in set(event_ids):
                conn.execute(
                    """INSERT OR IGNORE INTO memory_claim_evidence(
                           claim_id,event_id,relation,weight
                       ) VALUES(?,?,'supports',1.0)""",
                    (claim_id, event_id),
                )
        return created, previous_status != "active" and status == "active"

    def _promote_cross_context_patterns(self) -> int:
        """Promote only repeated, identical safe claims across three scopes to L3."""
        safe_dimensions = {"teaching_preference", "self_regulation"}
        with self.store.connect() as conn:
            rows = conn.execute(
                """SELECT c.*,COUNT(e.id) evidence_count
                   FROM memory_claims c
                   JOIN memory_claim_evidence e ON e.claim_id=c.claim_id
                   WHERE c.owner_id='local' AND c.layer=2 AND c.status='active'
                   GROUP BY c.claim_id"""
            ).fetchall()
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            item = dict(row)
            if item["dimension"] in safe_dimensions:
                groups[(item["dimension"], item["claim_text"])].append(item)
        created = 0
        for (dimension, claim_text), claims in groups.items():
            if len({item["scope_key"] for item in claims}) < 3:
                continue
            if sum(int(item["evidence_count"]) for item in claims) < 8:
                continue
            source_claim_ids = [item["claim_id"] for item in claims]
            with self.store.connect() as conn:
                existing = conn.execute(
                    """SELECT claim_id FROM memory_claims
                       WHERE owner_id='local' AND layer=3 AND dimension=? AND claim_text=?
                         AND status IN ('candidate','active')""",
                    (dimension, claim_text),
                ).fetchone()
                if existing:
                    claim_id = existing["claim_id"]
                else:
                    claim_id = self.new_id("claim")
                    conn.execute(
                        """INSERT INTO memory_claims(
                               claim_id,layer,dimension,scope_type,scope_key,claim_text,
                               source_kind,confidence,status,created_by
                           ) VALUES(?,3,?,'global','',?,'inferred',0.82,'active','agent')""",
                        (claim_id, dimension, claim_text),
                    )
                    created += 1
                for source_id in source_claim_ids:
                    conn.execute(
                        """INSERT OR IGNORE INTO memory_claim_evidence(
                               claim_id,source_claim_id,relation,weight
                           ) VALUES(?,?,'supports',1.0)""",
                        (claim_id, source_id),
                    )
        return created

    def record_knowledge_access(
        self,
        note_ids: list[int],
        *,
        session_id: str | None = None,
        message_id: str | None = None,
        reason: str = "retrieval",
    ) -> str | None:
        note_ids = sorted(set(int(value) for value in note_ids if int(value) > 0))
        if not note_ids:
            return None
        now = utc_now()
        with self.store.connect() as conn:
            valid_ids = [
                int(row["id"]) for row in conn.execute(
                    f"SELECT id FROM notes WHERE id IN ({','.join('?' for _ in note_ids)})", note_ids
                )
            ]
            for note_id in valid_ids:
                conn.execute(
                    """UPDATE notes SET access_count=access_count+1,last_accessed_at=?,
                       activation_score=min(1.0,activation_score+0.12),
                       stability_days=min(365.0,stability_days*1.08)
                       WHERE id=?""",
                    (now, note_id),
                )
        return self.record_event(
            surface="knowledge_retrieval",
            event_type="knowledge_access",
            source_kind="system",
            session_id=session_id,
            message_id=message_id,
            concept_note_ids=valid_ids,
            payload={"reason": reason, "note_ids": valid_ids},
        ) if valid_ids else None

    def refresh_knowledge_weights(self) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        with self.store.connect() as conn:
            rows = [dict(row) for row in conn.execute(
                """SELECT n.*,
                   (SELECT COUNT(*) FROM links l
                    WHERE l.source_id=n.id OR l.target_id=n.id) AS graph_degree
                   FROM notes n"""
            )]
            values = []
            for note in rows:
                value = note_activation(note, now=now)
                graph_bonus = min(0.15, math.log1p(int(note["graph_degree"])) * 0.035)
                value = round(min(1.0, value + graph_bonus), 4)
                conn.execute("UPDATE notes SET activation_score=? WHERE id=?", (value, note["id"]))
                values.append((value, note))
        values.sort(key=lambda item: item[0], reverse=True)
        return {
            "updated": len(values),
            "most_active": [item[1]["title"] for item in values[:5]],
            "compression_candidates": [
                {"id": item[1]["id"], "title": item[1]["title"], "activation": item[0]}
                for item in values if item[0] < 0.16 and not item[1]["compressed"]
            ][:20],
            "auto_compressed": 0,
        }

    @staticmethod
    def _review_dimension(task: dict[str, Any]) -> str:
        payload = task.get("payload", {})
        explicit = str(payload.get("mastery_dimension", ""))
        if explicit in MASTERY_DIMENSIONS:
            return explicit
        if task.get("task_type") == "quiz":
            return "recognition"
        question = str(payload.get("question") or "；".join(payload.get("questions", [])))
        if re.search(r"迁移|新情境|另一领域|跨领域", question):
            return "transfer"
        if re.search(r"应用|例子|场景|解决", question):
            return "application"
        return "explanation"

    def plan_mastery_update(
        self, task: dict[str, Any], quality: int, self_rating: int | None = None
    ) -> dict[str, Any]:
        quality = max(0, min(3, int(quality)))
        concept = str(task.get("concept", "")).strip() or "未命名概念"
        dimension = self._review_dimension(task)
        with self.store.connect() as conn:
            row = conn.execute(
                "SELECT * FROM concept_mastery WHERE owner_id='local' AND concept_key=?", (concept,)
            ).fetchone()
        existing = dict(row) if row else {}
        scores = {name: float(existing.get(f"{name}_score", 0.0) or 0.0) for name in MASTERY_DIMENSIONS}
        old_score = scores[dimension]
        if quality == 0:
            scores[dimension] = old_score * 0.55
        else:
            gain = {1: 0.18, 2: 0.32, 3: 0.46}[quality]
            scores[dimension] = old_score + (1.0 - old_score) * gain
        stability = float(existing.get("stability_days", 1.0) or 1.0)
        if quality == 0:
            stability = max(0.75, stability * 0.55)
        else:
            stability = min(365.0, stability * {1: 1.25, 2: 1.8, 3: 2.6}[quality] + 0.5)
        interval = max(1, min(90, round(-stability * math.log(0.72))))
        stage = "exposed"
        if scores["recognition"] >= 0.6:
            stage = "recognizes"
        if scores["explanation"] >= 0.6:
            stage = "explains"
        if scores["application"] >= 0.6:
            stage = "applies"
        if scores["transfer"] >= 0.6:
            stage = "transfers"
        confidence = sum(scores.values()) / len(scores)
        return {
            "concept": concept,
            "dimension": dimension,
            "quality": quality,
            "self_rating": None if self_rating is None else max(0, min(3, int(self_rating))),
            "scores": {key: round(value, 4) for key, value in scores.items()},
            "stage": stage,
            "confidence": round(confidence, 4),
            "stability_days": round(stability, 4),
            "next_interval_days": interval,
            "successful_reviews": int(existing.get("successful_reviews", 0) or 0) + int(quality > 0),
            "lapses": int(existing.get("lapses", 0) or 0) + int(quality == 0),
        }

    def apply_mastery_update(
        self,
        plan: dict[str, Any],
        *,
        answer: str,
        task_id: int,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        next_review = now + timedelta(days=int(plan["next_interval_days"]))
        concept = plan["concept"]
        event_id = self.record_event(
            surface="review",
            event_type="review_attempt",
            source_kind="observed",
            concepts=[concept],
            payload={
                "task_id": task_id,
                "quality": plan["quality"],
                "self_rating": plan.get("self_rating"),
                "calibration_gap": (
                    None if plan.get("self_rating") is None
                    else abs(int(plan["self_rating"]) - int(plan["quality"]))
                ),
                "dimension": plan["dimension"],
                "answer_excerpt": answer[:500],
            },
        )
        scores = plan["scores"]
        with self.store.connect() as conn:
            note = conn.execute(
                "SELECT id FROM notes WHERE title=? ORDER BY kind='concept' DESC LIMIT 1", (concept,)
            ).fetchone()
            conn.execute(
                """INSERT INTO concept_mastery(
                       owner_id,concept_key,concept_note_id,stage,confidence,last_evidence_at,
                       recognition_score,explanation_score,application_score,transfer_score,
                       stability_days,last_reviewed_at,next_review_at,successful_reviews,lapses
                   ) VALUES('local',?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(owner_id,concept_key) DO UPDATE SET
                       concept_note_id=COALESCE(excluded.concept_note_id,concept_mastery.concept_note_id),
                       stage=excluded.stage,confidence=excluded.confidence,
                       last_evidence_at=excluded.last_evidence_at,
                       recognition_score=excluded.recognition_score,
                       explanation_score=excluded.explanation_score,
                       application_score=excluded.application_score,
                       transfer_score=excluded.transfer_score,
                       stability_days=excluded.stability_days,last_reviewed_at=excluded.last_reviewed_at,
                       next_review_at=excluded.next_review_at,
                       successful_reviews=excluded.successful_reviews,lapses=excluded.lapses,
                       updated_at=excluded.updated_at""",
                (
                    concept, int(note["id"]) if note else None, plan["stage"], plan["confidence"], utc_now(),
                    scores["recognition"], scores["explanation"], scores["application"], scores["transfer"],
                    plan["stability_days"], utc_now(), next_review.isoformat(timespec="seconds"),
                    plan["successful_reviews"], plan["lapses"],
                ),
            )
            should_probe_transfer = (
                scores["explanation"] >= 0.6 and scores["transfer"] < 0.6
            )
            if should_probe_transfer:
                pending = conn.execute(
                    """SELECT id FROM tasks
                       WHERE status='pending' AND task_type='feynman' AND concept=? LIMIT 1""",
                    (concept,),
                ).fetchone()
                if not pending:
                    due = (now + timedelta(days=1)).isoformat(timespec="seconds")
                    payload = {
                        "question": (
                            f"假设你要把“{concept}”教给一个从未学过它的人："
                            "请不用原定义解释机制，再给一个新情境中的例子和一个失效边界。"
                        ),
                        "mastery_dimension": "transfer",
                        "trigger": "explanation_ready_transfer_unverified",
                    }
                    conn.execute(
                        """INSERT INTO tasks(
                               title,task_type,concept,payload_json,due_at,status,xp
                           ) VALUES(?, 'feynman', ?, ?, ?, 'pending', 18)""",
                        (
                            f"费曼复述：{concept}", concept,
                            json.dumps(payload, ensure_ascii=False), due,
                        ),
                    )
            conn.execute(
                """INSERT INTO concept_mastery_evidence(
                       owner_id,concept_key,event_id,dimension,outcome,weight,stage_after
                   ) VALUES('local',?,?,?,?,?,?)""",
                (
                    concept, event_id,
                    "recall" if plan["dimension"] == "recognition" else plan["dimension"],
                    "weakens" if plan["quality"] == 0 else "supports",
                    {0: 0.8, 1: 0.45, 2: 0.7, 3: 1.0}[plan["quality"]], plan["stage"],
                ),
            )
        return self.mastery_for(concept) or {}

    def _mastery_snapshot(self, row: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        reviewed_at = parse_utc(row.get("last_reviewed_at")) or parse_utc(row.get("created_at")) or now
        elapsed = max(0.0, (now - reviewed_at).total_seconds() / 86_400)
        retention = exponential_retention(elapsed, float(row.get("stability_days", 1.0) or 1.0))
        row["retention"] = round(retention, 4)
        row["effective_scores"] = {
            dimension: round(float(row.get(f"{dimension}_score", 0.0) or 0.0) * retention, 4)
            for dimension in MASTERY_DIMENSIONS
        }
        return row

    def mastery_for(self, concept: str, *, now: datetime | None = None) -> dict[str, Any] | None:
        with self.store.connect() as conn:
            row = conn.execute(
                "SELECT * FROM concept_mastery WHERE owner_id='local' AND concept_key=?", (concept,)
            ).fetchone()
        return self._mastery_snapshot(dict(row), now=now) if row else None

    def overview(self) -> dict[str, Any]:
        with self.store.connect() as conn:
            claims = [dict(row) for row in conn.execute(
                """SELECT claim_id,layer,dimension,scope_type,scope_key,claim_text,
                          confidence,status,updated_at
                   FROM memory_claims WHERE owner_id='local'
                   ORDER BY layer DESC,status='active' DESC,confidence DESC LIMIT 30"""
            )]
            mastery = [self._mastery_snapshot(dict(row)) for row in conn.execute(
                """SELECT * FROM concept_mastery WHERE owner_id='local'
                   ORDER BY next_review_at IS NULL,next_review_at LIMIT 30"""
            )]
            events = conn.execute(
                "SELECT COUNT(*) n FROM learning_events WHERE event_type!='memory_reflection'"
            ).fetchone()["n"]
        return {
            "experience_memory": {"claims": claims, "l1_event_count": events},
            "knowledge_memory": self.knowledge_overview(),
            "concept_mastery": mastery,
            "last_reflection_at": self.store.setting("memory.last_reflection_at", ""),
        }

    def knowledge_overview(self) -> dict[str, Any]:
        with self.store.connect() as conn:
            rows = [dict(row) for row in conn.execute(
                """SELECT id,title,activation_score,access_count,last_accessed_at,compressed
                   FROM notes ORDER BY activation_score DESC,last_accessed_at DESC LIMIT 100"""
            )]
        return {
            "most_active": rows[:5],
            "compression_candidates": [
                item for item in reversed(rows)
                if float(item["activation_score"]) < 0.16 and not item["compressed"]
            ][:20],
            "auto_compressed": 0,
        }
