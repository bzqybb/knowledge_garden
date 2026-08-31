from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

from core.config import llm_config
from core.context import ChatMessage, GardenContext, KnowledgeScope, LearnerSettings, ToolPolicy
from core.storage import GardenStore
from core.tracememo import tracememo_config


class ContextBuilder:
    """Build immutable request context from explicit settings and conversation facts.

    The builder is deliberately read-only. Persisting messages, retrieving inferred
    memory, and selecting teaching strategies belong to separate services/graph nodes.
    """

    def __init__(self, store: GardenStore):
        self.store = store

    def build(
        self,
        question: str,
        history: list[dict[str, Any]] | None = None,
        *,
        session_id: str,
        request_id: str,
        message_id: str,
        active_capability: str = "gardener_chat",
        turn_teaching_preferences: list[str] | tuple[str, ...] | None = None,
    ) -> GardenContext:
        question = question.strip()
        if not question:
            raise ValueError("请先写下你想问园丁的问题")

        messages: list[ChatMessage] = []
        for index, item in enumerate((history or [])[-10:]):
            role = str(item.get("role", ""))
            content = str(item.get("content", "")).strip()[:100_000]
            if role not in {"user", "assistant"} or not content:
                continue
            messages.append(ChatMessage(
                message_id=str(item.get("message_id") or f"history-{index}-{uuid4().hex}"),
                role=role,
                content=content,
                capability=str(item.get("capability") or active_capability),
                evidence_layer=(
                    str(item.get("evidence_layer") or "").strip()[:64] or None
                ),
            ))

        enabled_set = set(self.store.setting("enabled_tools", [
            "local_wiki", "obsidian", "wikipedia", "academic_search", "public_web",
            "understanding_model",
        ]))
        system_allowed = {"local_wiki", "obsidian"}
        network_disabled = os.getenv("GARDEN_DISABLE_NETWORK", "").strip().lower() in {"1", "true", "yes"}
        if not network_disabled:
            system_allowed.update({"wikipedia", "academic_search", "public_web"})
        if llm_config().enabled:
            system_allowed.add("understanding_model")
        # Saving the TraceMemo token in the Garden UI is the user's explicit
        # opt-in to mount the read-only WeChat tool.  This also upgrades stores
        # whose older enabled_tools setting predates the connector.
        try:
            trace_config = tracememo_config(
                str(self.store.setting("tracememo_base_url", "http://127.0.0.1:6131"))
            )
            if trace_config.enabled:
                enabled_set.add("tracememo_reader")
                system_allowed.add("tracememo_reader")
        except Exception:
            pass
        enabled = tuple(sorted(enabled_set))
        allowed = tuple(sorted(enabled_set & system_allowed))

        selected_note_ids = tuple(
            int(value) for value in self.store.setting("selected_note_ids", [])
            if str(value).isdigit() and int(value) > 0
        )
        selected_mocs = tuple(
            str(value).strip() for value in self.store.setting("selected_moc_titles", [])
            if str(value).strip()
        )
        saved_teaching_preferences = [
            str(value).strip()
            for value in self.store.setting("teaching_preferences", [])
            if str(value).strip()
        ]
        current_turn_preferences = [
            str(value).strip()[:500]
            for value in (turn_teaching_preferences or [])
            if str(value).strip()
        ]
        teaching_preferences = tuple(dict.fromkeys(
            saved_teaching_preferences + current_turn_preferences
        ))
        return GardenContext(
            request_id=request_id,
            session_id=session_id,
            current_message=ChatMessage(
                message_id=message_id,
                role="user",
                content=question,
                capability=active_capability,
            ),
            conversation_history=tuple(messages),
            active_capability=active_capability,
            learner_settings=LearnerSettings(
                declared_level=str(self.store.setting("learning_level", "本科入门")),
                grade_level=self.store.setting("grade_level", None),
                learning_goals=tuple(self.store.setting("learning_goals", [])),
                explicit_interests=tuple(self.store.setting("interests", [])),
                explicit_teaching_preferences=teaching_preferences,
            ),
            knowledge_scope=KnowledgeScope(
                vault_id=str(self.store.setting("vault_id", "default")),
                selected_note_ids=selected_note_ids,
                selected_moc_titles=selected_mocs,
                allow_raw_fallback=bool(self.store.setting("allow_raw_fallback", False)),
            ),
            tool_policy=ToolPolicy(
                user_enabled=enabled,
                allowed=allowed,
                mounted=allowed,
            ),
        )
