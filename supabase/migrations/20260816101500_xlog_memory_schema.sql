-- xlog learning memory (Supabase / Postgres)
-- Project: xlog (husqkkwmnqssmjhvuftx)

CREATE TABLE IF NOT EXISTS rubric_versions (
  id BIGSERIAL PRIMARY KEY,
  version INTEGER NOT NULL,
  owner TEXT NOT NULL,
  criteria JSONB NOT NULL,
  preferences JSONB NOT NULL,
  notes TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'unknown',
  created_at BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback_events (
  id BIGSERIAL PRIMARY KEY,
  ts BIGINT NOT NULL,
  job_id TEXT NOT NULL,
  user_choice TEXT NOT NULL,
  user_comment TEXT NOT NULL DEFAULT '',
  judge_winner TEXT,
  agreement BOOLEAN,
  variants JSONB,
  judge_verdict JSONB,
  rubric_version INTEGER
);

CREATE TABLE IF NOT EXISTS style_references (
  id BIGSERIAL PRIMARY KEY,
  ts BIGINT NOT NULL,
  url TEXT NOT NULL,
  notes TEXT NOT NULL DEFAULT '',
  file_path TEXT,
  style_summary TEXT,
  hook_technique TEXT,
  pacing TEXT,
  caption_style TEXT,
  structure TEXT,
  learned_rules JSONB,
  style JSONB,
  rubric_version INTEGER
);

CREATE TABLE IF NOT EXISTS shorts_form_versions (
  id BIGSERIAL PRIMARY KEY,
  version INTEGER NOT NULL,
  source TEXT NOT NULL,
  structure JSONB NOT NULL,
  global_rules JSONB NOT NULL,
  created_at BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS learned_preferences (
  id BIGSERIAL PRIMARY KEY,
  rule TEXT NOT NULL UNIQUE,
  source TEXT NOT NULL,
  source_id TEXT,
  first_seen_at BIGINT NOT NULL,
  last_seen_at BIGINT NOT NULL,
  times_seen INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS ft_examples (
  id BIGSERIAL PRIMARY KEY,
  ts BIGINT NOT NULL,
  kind TEXT NOT NULL,
  source TEXT NOT NULL,
  source_id TEXT,
  messages JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS ft_jobs (
  id BIGSERIAL PRIMARY KEY,
  ts BIGINT NOT NULL,
  openai_job_id TEXT,
  status TEXT NOT NULL,
  model TEXT,
  example_count INTEGER NOT NULL DEFAULT 0,
  fingerprint TEXT,
  error TEXT
);

CREATE TABLE IF NOT EXISTS ft_active (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  model TEXT NOT NULL,
  job_id TEXT,
  updated_at BIGINT NOT NULL
);
