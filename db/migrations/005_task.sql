-- Tasks the agent works on for the user (for now only type 'report_creation'). The agent
-- creates a task or reopens an existing one from what the user says in the chat; the chat
-- session keeps the active task in its state. Only the backend's direct Postgres role reads
-- and writes them.

BEGIN;

CREATE TABLE task (
    id         integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    type       text NOT NULL CHECK (type IN ('report_creation')),
    title      text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE task ENABLE ROW LEVEL SECURITY;

COMMIT;
