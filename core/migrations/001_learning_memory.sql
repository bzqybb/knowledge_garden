-- Knowledge Garden learning and memory foundation.
--
-- Architecture rule:
-- * ECNUClaw-style educational dimensions may describe observations, but a
--   keyword match can only create a learning_event; it cannot become a profile.
-- * DeepTutor-style long-term memory is represented as claims with evidence,
--   confidence, scope, status, and lifecycle.
-- * Obsidian Markdown remains the human-readable knowledge layer. These tables
--   store interaction evidence and Agent state, not a replacement knowledge base.

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL DEFAULT 'local',
    title TEXT NOT NULL DEFAULT '',
    default_capability TEXT NOT NULL DEFAULT 'gardener_chat',
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'closed', 'archived')),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    ended_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_owner_updated
    ON sessions(owner_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS session_messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    request_id TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    capability TEXT NOT NULL DEFAULT 'gardener_chat',
    content TEXT NOT NULL CHECK (length(trim(content)) > 0),
    metadata_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(metadata_json)),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_session_messages_session_created
    ON session_messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_session_messages_request
    ON session_messages(request_id);

-- L1 append-only observations. They are evidence, not durable conclusions.
CREATE TABLE IF NOT EXISTS learning_events (
    event_id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL DEFAULT 'local',
    session_id TEXT,
    message_id TEXT,
    surface TEXT NOT NULL,
    event_type TEXT NOT NULL,
    source_kind TEXT NOT NULL
        CHECK (source_kind IN ('explicit', 'observed', 'inferred', 'system')),
    payload_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(payload_json)),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (message_id) REFERENCES session_messages(message_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_learning_events_owner_created
    ON learning_events(owner_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_learning_events_surface_type
    ON learning_events(surface, event_type);

CREATE TABLE IF NOT EXISTS event_concepts (
    event_id TEXT NOT NULL,
    concept_key TEXT NOT NULL,
    concept_note_id INTEGER,
    relation TEXT NOT NULL DEFAULT 'about',
    PRIMARY KEY (event_id, concept_key),
    FOREIGN KEY (event_id) REFERENCES learning_events(event_id) ON DELETE CASCADE,
    FOREIGN KEY (concept_note_id) REFERENCES notes(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_event_concepts_concept
    ON event_concepts(concept_key, concept_note_id);

-- L2/L3 durable memory. A claim is never valid merely because a keyword matched.
CREATE TABLE IF NOT EXISTS memory_claims (
    claim_id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL DEFAULT 'local',
    layer INTEGER NOT NULL CHECK (layer IN (2, 3)),
    dimension TEXT NOT NULL,
    scope_type TEXT NOT NULL DEFAULT 'global',
    scope_key TEXT NOT NULL DEFAULT '',
    claim_text TEXT NOT NULL CHECK (length(trim(claim_text)) > 0),
    source_kind TEXT NOT NULL
        CHECK (source_kind IN ('explicit', 'observed', 'inferred')),
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    status TEXT NOT NULL DEFAULT 'candidate'
        CHECK (status IN ('candidate', 'active', 'rejected', 'superseded', 'expired')),
    valid_from TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    valid_until TEXT,
    superseded_by TEXT,
    created_by TEXT NOT NULL DEFAULT 'agent'
        CHECK (created_by IN ('user', 'agent', 'system')),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CHECK (valid_until IS NULL OR valid_until > valid_from),
    CHECK (superseded_by IS NULL OR superseded_by <> claim_id),
    FOREIGN KEY (superseded_by) REFERENCES memory_claims(claim_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_claims_active_scope
    ON memory_claims(owner_id, status, layer, dimension, scope_type, scope_key);

-- Exactly one evidence source per row keeps provenance queryable and enforceable.
CREATE TABLE IF NOT EXISTS memory_claim_evidence (
    id INTEGER PRIMARY KEY,
    claim_id TEXT NOT NULL,
    event_id TEXT,
    source_claim_id TEXT,
    message_id TEXT,
    relation TEXT NOT NULL DEFAULT 'supports'
        CHECK (relation IN ('supports', 'contradicts', 'supersedes')),
    weight REAL NOT NULL DEFAULT 1.0
        CHECK (weight >= 0.0 AND weight <= 1.0),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CHECK (
        (event_id IS NOT NULL) +
        (source_claim_id IS NOT NULL) +
        (message_id IS NOT NULL) = 1
    ),
    FOREIGN KEY (claim_id) REFERENCES memory_claims(claim_id) ON DELETE CASCADE,
    FOREIGN KEY (event_id) REFERENCES learning_events(event_id) ON DELETE CASCADE,
    FOREIGN KEY (source_claim_id) REFERENCES memory_claims(claim_id) ON DELETE CASCADE,
    FOREIGN KEY (message_id) REFERENCES session_messages(message_id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_claim_evidence_event_unique
    ON memory_claim_evidence(claim_id, event_id, relation)
    WHERE event_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_claim_evidence_claim_unique
    ON memory_claim_evidence(claim_id, source_claim_id, relation)
    WHERE source_claim_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_claim_evidence_message_unique
    ON memory_claim_evidence(claim_id, message_id, relation)
    WHERE message_id IS NOT NULL;

-- Mastery is concept-specific learning state, not a general personality trait.
CREATE TABLE IF NOT EXISTS concept_mastery (
    owner_id TEXT NOT NULL DEFAULT 'local',
    concept_key TEXT NOT NULL,
    concept_note_id INTEGER,
    stage TEXT NOT NULL DEFAULT 'exposed'
        CHECK (stage IN ('unseen', 'exposed', 'recognizes', 'explains', 'applies', 'transfers')),
    confidence REAL NOT NULL DEFAULT 0.0
        CHECK (confidence >= 0.0 AND confidence <= 1.0),
    last_evidence_at TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (owner_id, concept_key),
    FOREIGN KEY (concept_note_id) REFERENCES notes(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_concept_mastery_stage
    ON concept_mastery(owner_id, stage, confidence);

CREATE TABLE IF NOT EXISTS concept_mastery_evidence (
    id INTEGER PRIMARY KEY,
    owner_id TEXT NOT NULL DEFAULT 'local',
    concept_key TEXT NOT NULL,
    event_id TEXT NOT NULL,
    dimension TEXT NOT NULL
        CHECK (dimension IN ('recall', 'explanation', 'application', 'transfer')),
    outcome TEXT NOT NULL
        CHECK (outcome IN ('supports', 'weakens')),
    weight REAL NOT NULL DEFAULT 1.0
        CHECK (weight >= 0.0 AND weight <= 1.0),
    stage_after TEXT NOT NULL
        CHECK (stage_after IN ('unseen', 'exposed', 'recognizes', 'explains', 'applies', 'transfers')),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (owner_id, concept_key, event_id, dimension),
    FOREIGN KEY (owner_id, concept_key)
        REFERENCES concept_mastery(owner_id, concept_key) ON DELETE CASCADE,
    FOREIGN KEY (event_id) REFERENCES learning_events(event_id) ON DELETE CASCADE
);
