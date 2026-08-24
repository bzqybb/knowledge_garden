-- TraceMemo is a read-only data adapter. Selected excerpts first become L1
-- candidates and can enter Obsidian only after an explicit review action.

CREATE TABLE IF NOT EXISTS wechat_candidates (
    candidate_id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL DEFAULT 'local',
    title TEXT NOT NULL CHECK (length(trim(title)) > 0),
    talker TEXT NOT NULL DEFAULT '',
    time_range TEXT NOT NULL DEFAULT '',
    contact_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(contact_json)),
    query_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(query_json)),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'accepted', 'rejected')),
    message_count INTEGER NOT NULL DEFAULT 0 CHECK (message_count >= 0),
    raw_path TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    reviewed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_wechat_candidates_status_created
    ON wechat_candidates(owner_id, status, created_at DESC);

CREATE TABLE IF NOT EXISTS wechat_candidate_messages (
    id INTEGER PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    source_message_id TEXT NOT NULL,
    sender TEXT NOT NULL DEFAULT '',
    sent_at TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL CHECK (length(trim(content)) > 0),
    message_type TEXT NOT NULL DEFAULT 'text',
    source_json TEXT NOT NULL DEFAULT '{}'
        CHECK (json_valid(source_json)),
    UNIQUE (candidate_id, source_message_id),
    FOREIGN KEY (candidate_id) REFERENCES wechat_candidates(candidate_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_wechat_candidate_messages_candidate
    ON wechat_candidate_messages(candidate_id, id);
