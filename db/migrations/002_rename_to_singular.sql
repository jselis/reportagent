-- Naming convention: a table is named after the concept it holds, as a singular noun, and a
-- view ends in _vw. Foreign key columns are named after the table they point to.
--   ground_truth_cases    -> gt_case
--   eval_runs             -> eval_run
--   eval_run_results      -> eval_run_case   (columns run_id -> eval_run_id, case_id -> gt_case_id)
--   eval_run_summary      -> eval_run_summary_vw
-- Sequences, constraints, indexes and the trigger are renamed to match.

BEGIN;

DROP VIEW eval_run_summary;

ALTER TABLE ground_truth_cases RENAME TO gt_case;
ALTER TABLE eval_runs RENAME TO eval_run;
ALTER TABLE eval_run_results RENAME TO eval_run_case;

ALTER TABLE eval_run_case RENAME COLUMN run_id TO eval_run_id;
ALTER TABLE eval_run_case RENAME COLUMN case_id TO gt_case_id;

ALTER SEQUENCE ground_truth_cases_id_seq RENAME TO gt_case_id_seq;
ALTER SEQUENCE eval_runs_id_seq RENAME TO eval_run_id_seq;
ALTER SEQUENCE eval_run_results_id_seq RENAME TO eval_run_case_id_seq;

ALTER TABLE gt_case RENAME CONSTRAINT ground_truth_cases_pkey TO gt_case_pkey;
ALTER TABLE gt_case RENAME CONSTRAINT ground_truth_cases_superseded_by_fkey TO gt_case_superseded_by_fkey;
ALTER TRIGGER ground_truth_cases_set_updated_at ON gt_case RENAME TO gt_case_set_updated_at;

ALTER TABLE eval_run RENAME CONSTRAINT eval_runs_pkey TO eval_run_pkey;
ALTER TABLE eval_run RENAME CONSTRAINT eval_runs_pipeline_check TO eval_run_pipeline_check;
ALTER TABLE eval_run RENAME CONSTRAINT eval_runs_status_check TO eval_run_status_check;

ALTER TABLE eval_run_case RENAME CONSTRAINT eval_run_results_pkey TO eval_run_case_pkey;
ALTER TABLE eval_run_case RENAME CONSTRAINT eval_run_results_run_id_fkey TO eval_run_case_eval_run_id_fkey;
ALTER TABLE eval_run_case RENAME CONSTRAINT eval_run_results_case_id_fkey TO eval_run_case_gt_case_id_fkey;
ALTER TABLE eval_run_case RENAME CONSTRAINT eval_run_results_generation_status_check TO eval_run_case_generation_status_check;
ALTER TABLE eval_run_case RENAME CONSTRAINT eval_run_results_run_id_case_id_repeat_index_key TO eval_run_case_eval_run_id_gt_case_id_repeat_index_key;
ALTER INDEX eval_run_results_case_id_idx RENAME TO eval_run_case_gt_case_id_idx;

CREATE VIEW eval_run_summary_vw WITH (security_invoker = true) AS
SELECT
    r.id AS eval_run_id,
    count(DISTINCT erc.gt_case_id) FILTER (WHERE gt.superseded_at IS NULL) AS case_count,
    count(*) FILTER (WHERE gt.superseded_at IS NULL AND erc.generation_status = 'failed') AS failed_count,
    (SELECT count(*) FROM gt_case g
      WHERE g.superseded_at IS NULL
        AND NOT EXISTS (SELECT 1 FROM eval_run_case x WHERE x.eval_run_id = r.id AND x.gt_case_id = g.id)
    ) AS missing_case_count,
    avg(erc.faithfulness)          FILTER (WHERE gt.superseded_at IS NULL AND erc.generation_status = 'ok') AS avg_faithfulness,
    avg(erc.correctness)           FILTER (WHERE gt.superseded_at IS NULL AND erc.generation_status = 'ok') AS avg_correctness,
    avg(erc.retrieval_recall)      FILTER (WHERE gt.superseded_at IS NULL AND erc.generation_status = 'ok') AS avg_retrieval_recall,
    avg(erc.retrieval_precision)   FILTER (WHERE gt.superseded_at IS NULL AND erc.generation_status = 'ok') AS avg_retrieval_precision,
    avg(erc.retrieval_f1)          FILTER (WHERE gt.superseded_at IS NULL AND erc.generation_status = 'ok') AS avg_retrieval_f1,
    avg(erc.tokens_used)           FILTER (WHERE gt.superseded_at IS NULL AND erc.generation_status = 'ok') AS avg_tokens_used,
    avg(erc.response_time_seconds) FILTER (WHERE gt.superseded_at IS NULL AND erc.generation_status = 'ok') AS avg_response_time_seconds,
    avg(erc.n_searches)            FILTER (WHERE gt.superseded_at IS NULL AND erc.generation_status = 'ok') AS avg_searches
FROM eval_run r
LEFT JOIN eval_run_case erc ON erc.eval_run_id = r.id
LEFT JOIN gt_case gt ON gt.id = erc.gt_case_id
GROUP BY r.id;

REVOKE ALL ON eval_run_summary_vw FROM anon, authenticated;

COMMIT;
