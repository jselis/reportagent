-- Evaluation results now live in eval_run / eval_run_case (migrated in 001: all legacy results
-- were verified to exist unchanged in the legacy run). The old result columns on gt_case are
-- no longer read or written.

BEGIN;

ALTER TABLE gt_case
    DROP COLUMN actual_answer,
    DROP COLUMN actual_chunks,
    DROP COLUMN faithfulness,
    DROP COLUMN correctness,
    DROP COLUMN retrieval_recall,
    DROP COLUMN retrieval_precision,
    DROP COLUMN retrieval_f1;

COMMIT;
