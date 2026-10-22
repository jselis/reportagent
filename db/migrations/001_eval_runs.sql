-- Evaluation runs: one row per run (a pipeline + its hyperparameters), one result row per
-- case (and repeat) in that run. Aggregates are computed on read by eval_run_summary.

BEGIN;

-- Ground truth cases are never edited or deleted: a corrected case is added as a new row
-- and the old one is marked superseded. Results of superseded cases stay as history but
-- no longer count in the aggregates.
ALTER TABLE ground_truth_cases
    ADD COLUMN superseded_at timestamptz,
    ADD COLUMN superseded_by integer REFERENCES ground_truth_cases(id);

CREATE TABLE eval_runs (
    id          serial PRIMARY KEY,
    created_at  timestamptz NOT NULL DEFAULT now(),
    label       text,
    pipeline    text NOT NULL CHECK (pipeline IN ('rag', 'agent')),
    -- everything needed to reproduce the run: e.g. {"top_k": 5, "answer_model": "...", "repeats": 1}
    config      jsonb NOT NULL,
    status      text NOT NULL CHECK (status IN ('generating', 'generated', 'scoring', 'scored', 'failed')),
    judge_model text,
    scored_at   timestamptz
);

CREATE TABLE eval_run_results (
    id                    serial PRIMARY KEY,
    run_id                integer NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    case_id               integer NOT NULL REFERENCES ground_truth_cases(id),
    repeat_index          integer NOT NULL DEFAULT 0,
    generation_status     text NOT NULL CHECK (generation_status IN ('ok', 'failed')),
    error                 text,
    generated_at          timestamptz NOT NULL DEFAULT now(),
    actual_answer         text,
    -- chunks as [{"chunk_id": ..., "chunk": ...}]: everything that reached the LLM (recall basis) ...
    fed_chunks            jsonb,
    -- ... and what the pipeline decided is relevant: top-k for RAG, the cited chunks for the agent (precision basis)
    selected_chunks       jsonb,
    tokens_used           integer,
    response_time_seconds double precision,
    n_searches            integer,
    faithfulness          double precision,
    correctness           double precision,
    retrieval_recall      double precision,
    retrieval_precision   double precision,
    retrieval_f1          double precision,
    UNIQUE (run_id, case_id, repeat_index)
);

CREATE INDEX eval_run_results_case_id_idx ON eval_run_results (case_id);

-- Same security posture as the other tables: RLS on, no policies; only the backend's direct
-- Postgres role reads and writes.
ALTER TABLE eval_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE eval_run_results ENABLE ROW LEVEL SECURITY;

CREATE VIEW eval_run_summary WITH (security_invoker = true) AS
SELECT
    r.id AS run_id,
    count(DISTINCT res.case_id) FILTER (WHERE gt.superseded_at IS NULL) AS case_count,
    count(*) FILTER (WHERE gt.superseded_at IS NULL AND res.generation_status = 'failed') AS failed_count,
    (SELECT count(*) FROM ground_truth_cases g
      WHERE g.superseded_at IS NULL
        AND NOT EXISTS (SELECT 1 FROM eval_run_results x WHERE x.run_id = r.id AND x.case_id = g.id)
    ) AS missing_case_count,
    avg(res.faithfulness)          FILTER (WHERE gt.superseded_at IS NULL AND res.generation_status = 'ok') AS avg_faithfulness,
    avg(res.correctness)           FILTER (WHERE gt.superseded_at IS NULL AND res.generation_status = 'ok') AS avg_correctness,
    avg(res.retrieval_recall)      FILTER (WHERE gt.superseded_at IS NULL AND res.generation_status = 'ok') AS avg_retrieval_recall,
    avg(res.retrieval_precision)   FILTER (WHERE gt.superseded_at IS NULL AND res.generation_status = 'ok') AS avg_retrieval_precision,
    avg(res.retrieval_f1)          FILTER (WHERE gt.superseded_at IS NULL AND res.generation_status = 'ok') AS avg_retrieval_f1,
    avg(res.tokens_used)           FILTER (WHERE gt.superseded_at IS NULL AND res.generation_status = 'ok') AS avg_tokens_used,
    avg(res.response_time_seconds) FILTER (WHERE gt.superseded_at IS NULL AND res.generation_status = 'ok') AS avg_response_time_seconds,
    avg(res.n_searches)            FILTER (WHERE gt.superseded_at IS NULL AND res.generation_status = 'ok') AS avg_searches
FROM eval_runs r
LEFT JOIN eval_run_results res ON res.run_id = r.id
LEFT JOIN ground_truth_cases gt ON gt.id = res.case_id
GROUP BY r.id;

REVOKE ALL ON eval_run_summary FROM anon, authenticated;

-- Data migration: the evaluation results currently stored on ground_truth_cases become one
-- legacy run (plain RAG, top 5). Timestamps are approximate: the cases' updated_at stands in
-- for the generation time, which wasn't recorded.
WITH legacy_run AS (
    INSERT INTO eval_runs (created_at, label, pipeline, config, status, judge_model, scored_at)
    SELECT min(updated_at),
           'Legacy run migrated from ground_truth_cases (timestamps approximate)',
           'rag',
           '{"top_k": 5, "answer_model": "gpt-5.6-terra", "repeats": 1}'::jsonb,
           'scored',
           'gpt-5.6-terra',
           max(updated_at)
    FROM ground_truth_cases
    WHERE actual_answer IS NOT NULL AND actual_chunks IS NOT NULL
    RETURNING id
)
INSERT INTO eval_run_results (
    run_id, case_id, repeat_index, generation_status, generated_at,
    actual_answer, fed_chunks, selected_chunks,
    faithfulness, correctness, retrieval_recall, retrieval_precision, retrieval_f1
)
SELECT legacy_run.id, g.id, 0, 'ok', g.updated_at,
       g.actual_answer, g.actual_chunks, g.actual_chunks,
       g.faithfulness, g.correctness, g.retrieval_recall, g.retrieval_precision, g.retrieval_f1
FROM ground_truth_cases g, legacy_run
WHERE g.actual_answer IS NOT NULL AND g.actual_chunks IS NOT NULL;

COMMIT;
