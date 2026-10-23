-- Persistent chat sessions for the agent: a session holds the conversation state (the chunks
-- the model has seen and the cumulative source list); its events are the conversation history,
-- stored as ADK event JSON. Only the backend's direct Postgres role reads and writes them.

BEGIN;

CREATE TABLE chat_session (
    id         text PRIMARY KEY,
    app_name   text NOT NULL,
    user_id    text NOT NULL,
    state      jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE chat_event (
    id              serial PRIMARY KEY,
    chat_session_id text NOT NULL REFERENCES chat_session(id) ON DELETE CASCADE,
    event           jsonb NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX chat_event_chat_session_id_idx ON chat_event (chat_session_id, id);

ALTER TABLE chat_session ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_event ENABLE ROW LEVEL SECURITY;

COMMIT;
