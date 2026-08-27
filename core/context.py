from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now_datetime() -> datetime:
    """Return an aware UTC timestamp for request-scoped context objects."""
    return datetime.now(timezone.utc)


class FrozenContextModel(BaseModel):
    """Shared rules for data that must not mutate while a graph run is active."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ChatMessage(FrozenContextModel):
    message_id: str = Field(min_length=1, max_length=128)
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=100_000)
    capability: str | None = Field(default=None, max_length=64)
    evidence_layer: str | None = Field(default=None, max_length=64)
    created_at: datetime = Field(default_factory=utc_now_datetime)

    @field_validator("message_id", "content")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("created_at")
    @classmethod
    def require_aware_datetime(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(timezone.utc)


class LearnerSettings(FrozenContextModel):
    """Only user-declared or explicitly confirmed settings belong here.

    Observations such as cognitive load, inferred level, or teaching preference are
    learning events or memory claims. They must not silently become settings.
    """

    declared_level: str = Field(default="本科入门", max_length=64)
    grade_level: str | None = Field(default=None, max_length=64)
    learning_goals: tuple[str, ...] = ()
    explicit_interests: tuple[str, ...] = ()
    explicit_teaching_preferences: tuple[str, ...] = ()


class KnowledgeScope(FrozenContextModel):
    """Logical knowledge boundary; physical Obsidian paths stay in configuration."""

    vault_id: str = Field(default="default", min_length=1, max_length=128)
    selected_note_ids: tuple[int, ...] = ()
    selected_moc_titles: tuple[str, ...] = ()
    allow_raw_fallback: bool = False

    @field_validator("selected_note_ids")
    @classmethod
    def note_ids_must_be_positive(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(note_id <= 0 for note_id in value):
            raise ValueError("selected_note_ids must contain positive ids")
        return value


class ToolPolicy(FrozenContextModel):
    """A request can only mount tools allowed by the current policy."""

    user_enabled: tuple[str, ...] = ()
    allowed: tuple[str, ...] = ()
    mounted: tuple[str, ...] = ()

    @model_validator(mode="after")
    def mounted_tools_must_be_allowed(self) -> "ToolPolicy":
        unexpected = set(self.mounted) - set(self.allowed)
        if unexpected:
            raise ValueError(f"mounted tools are not allowed: {sorted(unexpected)}")
        return self


class GardenContext(FrozenContextModel):
    """Immutable input shared by every Agent participating in one request.

    This object deliberately excludes retrieved documents, Agent diagnoses,
    teaching strategies, generated answers, and memory writes. Those are mutable
    graph state or persisted evidence, not request context.
    """

    request_id: str = Field(min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)
    owner_id: str = Field(default="local", min_length=1, max_length=128)
    current_message: ChatMessage
    conversation_history: tuple[ChatMessage, ...] = ()
    active_capability: str = Field(default="gardener_chat", min_length=1, max_length=64)
    response_language: str = Field(default="zh-CN", min_length=2, max_length=16)
    learner_settings: LearnerSettings = Field(default_factory=LearnerSettings)
    knowledge_scope: KnowledgeScope = Field(default_factory=KnowledgeScope)
    tool_policy: ToolPolicy = Field(default_factory=ToolPolicy)
    created_at: datetime = Field(default_factory=utc_now_datetime)

    @model_validator(mode="after")
    def current_message_must_be_user_input(self) -> "GardenContext":
        if self.current_message.role != "user":
            raise ValueError("current_message must have role='user'")
        return self

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(timezone.utc)
