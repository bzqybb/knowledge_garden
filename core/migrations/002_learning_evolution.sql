-- Cached projections for knowledge activation and multidimensional mastery.
-- Source evidence remains in learning_events and the two evidence tables.

ALTER TABLE notes ADD COLUMN base_importance REAL NOT NULL DEFAULT 0.5
    CHECK (base_importance >= 0.0 AND base_importance <= 1.0);
ALTER TABLE notes ADD COLUMN activation_score REAL NOT NULL DEFAULT 0.5
    CHECK (activation_score >= 0.0 AND activation_score <= 1.0);
ALTER TABLE notes ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0
    CHECK (access_count >= 0);
ALTER TABLE notes ADD COLUMN last_accessed_at TEXT;
ALTER TABLE notes ADD COLUMN stability_days REAL NOT NULL DEFAULT 14.0
    CHECK (stability_days > 0.0);
ALTER TABLE notes ADD COLUMN compressed INTEGER NOT NULL DEFAULT 0
    CHECK (compressed IN (0, 1));

CREATE INDEX IF NOT EXISTS idx_notes_activation
    ON notes(compressed, activation_score DESC, last_accessed_at DESC);

ALTER TABLE concept_mastery ADD COLUMN recognition_score REAL NOT NULL DEFAULT 0.0
    CHECK (recognition_score >= 0.0 AND recognition_score <= 1.0);
ALTER TABLE concept_mastery ADD COLUMN explanation_score REAL NOT NULL DEFAULT 0.0
    CHECK (explanation_score >= 0.0 AND explanation_score <= 1.0);
ALTER TABLE concept_mastery ADD COLUMN application_score REAL NOT NULL DEFAULT 0.0
    CHECK (application_score >= 0.0 AND application_score <= 1.0);
ALTER TABLE concept_mastery ADD COLUMN transfer_score REAL NOT NULL DEFAULT 0.0
    CHECK (transfer_score >= 0.0 AND transfer_score <= 1.0);
ALTER TABLE concept_mastery ADD COLUMN stability_days REAL NOT NULL DEFAULT 1.0
    CHECK (stability_days > 0.0);
ALTER TABLE concept_mastery ADD COLUMN last_reviewed_at TEXT;
ALTER TABLE concept_mastery ADD COLUMN next_review_at TEXT;
ALTER TABLE concept_mastery ADD COLUMN successful_reviews INTEGER NOT NULL DEFAULT 0
    CHECK (successful_reviews >= 0);
ALTER TABLE concept_mastery ADD COLUMN lapses INTEGER NOT NULL DEFAULT 0
    CHECK (lapses >= 0);

CREATE INDEX IF NOT EXISTS idx_concept_mastery_review_due
    ON concept_mastery(owner_id, next_review_at, stage);
