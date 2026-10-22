from dataclasses import dataclass

from psycopg2.extras import Json

from app import agent, llm
from app.db import get_connection
from app.jobs import Job
from app.llm import run_llm_request
from app.models import (
    CorrectnessJudgement,
    EvalRunNotFoundError,
    EvalRunResponse,
    FaithfulnessJudgement,
    RetrievedChunk,
    StartEvalRunRequest,
)
from app.research import research

FAITHFULNESS_INSTRUCTIONS = """You assess whether an AI-generated answer is faithful to the document chunks it was based on.

Break the answer down into its individual atomic factual claims. Ignore citation markers like [1], [2], and do not treat a "the answer is not in the documents"-style refusal as a claim (return an empty claims list for it).

For each claim, decide whether it is directly supported by the provided chunks — `supported: true` if so, `false` if the chunks don't back it up or it isn't in them at all."""

CORRECTNESS_INSTRUCTIONS = """You judge whether a generated answer is correct compared to a reference (ground truth) answer, for the same question. Ignore citation markers like [1], [2] in the generated answer.

Score:
- 0: the generated answer is wrong or contradicts the reference answer.
- 0.5: partially correct — it captures the main point but has caveats, is incomplete, or has minor inaccuracies.
- 1: correct — semantically equivalent to the reference answer, even if worded differently."""


@dataclass
class _Generated:
    """What one pipeline run produced for one question."""

    answer: str
    fed_chunks: list[RetrievedChunk]  # everything that reached the LLM (recall basis)
    selected_chunks: list[RetrievedChunk]  # what the pipeline decided is relevant (precision basis)
    tokens_used: int
    response_time_seconds: float
    n_searches: int | None


def _generate_one(pipeline: str, config: dict, question: str) -> _Generated:
    if pipeline == "rag":
        response, chunks = research(question, config["top_k"])
        # Plain RAG feeds the top-k to the model and treats all of them as selected.
        return _Generated(
            answer=response.answer.text,
            fed_chunks=chunks,
            selected_chunks=chunks,
            tokens_used=response.tokens_used,
            response_time_seconds=response.response_time_seconds,
            n_searches=None,
        )
    if pipeline == "agent":
        run = agent.run_agent_with_chunks(question, config["max_llm_calls"], config["search_top_k"])
        return _Generated(
            answer=run.response.answer.text,
            fed_chunks=run.fed_chunks,
            selected_chunks=run.selected_chunks,
            tokens_used=run.response.tokens_used,
            response_time_seconds=run.response.response_time_seconds,
            n_searches=len(run.response.steps),
        )
    raise ValueError(f"Unknown pipeline: {pipeline}")


def _chunks_json(chunks: list[RetrievedChunk]) -> Json:
    return Json([{"chunk_id": c.id, "chunk": c.text} for c in chunks])


def _answer_model(pipeline: str) -> str:
    return llm.MODEL if pipeline == "rag" else agent.MODEL


def create_eval_run(request: StartEvalRunRequest) -> int:
    """Insert a new run (status 'generating') with everything needed to reproduce it."""
    config = request.config.model_dump(exclude={"pipeline"})
    config["answer_model"] = _answer_model(request.config.pipeline)
    config["repeats"] = request.repeats

    conn = get_connection()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO eval_run (label, pipeline, config, status)
                VALUES (%s, %s, %s, 'generating')
                RETURNING id
                """,
                (request.label, request.config.pipeline, Json(config)),
            )
            (eval_run_id,) = cur.fetchone()
    finally:
        conn.close()
    return eval_run_id


def require_eval_run(eval_run_id: int) -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM eval_run WHERE id = %s", (eval_run_id,))
            if cur.fetchone() is None:
                raise EvalRunNotFoundError(eval_run_id)
    finally:
        conn.close()


def _set_run_status(conn, eval_run_id: int, status: str) -> None:
    with conn, conn.cursor() as cur:
        cur.execute("UPDATE eval_run SET status = %s WHERE id = %s", (status, eval_run_id))


def run_generation(job: Job, eval_run_id: int) -> None:
    """Bring a run up to date: generate an answer for every active ground truth case (and
    repeat) that has no successful result in the run yet, using the run's stored config.
    Works for a fresh run, for cases added or corrected later, and to retry failures."""
    conn = get_connection()
    try:
        _set_run_status(conn, eval_run_id, "generating")
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT pipeline, config FROM eval_run WHERE id = %s", (eval_run_id,))
                pipeline, config = cur.fetchone()
                cur.execute("SELECT id, question FROM gt_case WHERE superseded_at IS NULL ORDER BY id")
                cases = cur.fetchall()
                cur.execute(
                    """
                    SELECT gt_case_id, repeat_index FROM eval_run_case
                    WHERE eval_run_id = %s AND generation_status = 'ok'
                    """,
                    (eval_run_id,),
                )
                done = set(cur.fetchall())

            todo = [
                (case_id, question, repeat_index)
                for case_id, question in cases
                for repeat_index in range(config["repeats"])
                if (case_id, repeat_index) not in done
            ]
            job.set_total(len(todo))

            for case_id, question, repeat_index in todo:
                try:
                    generated = _generate_one(pipeline, config, question)
                    row = (
                        "ok", None, generated.answer,
                        _chunks_json(generated.fed_chunks), _chunks_json(generated.selected_chunks),
                        generated.tokens_used, generated.response_time_seconds, generated.n_searches,
                    )
                    failed = False
                except Exception as exc:
                    row = ("failed", f"{type(exc).__name__}: {exc}"[:500], None, None, None, None, None, None)
                    failed = True

                with conn, conn.cursor() as cur:
                    # A retried (previously failed) row is reset, scores included.
                    cur.execute(
                        """
                        INSERT INTO eval_run_case (
                            eval_run_id, gt_case_id, repeat_index, generation_status, error,
                            actual_answer, fed_chunks, selected_chunks,
                            tokens_used, response_time_seconds, n_searches
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (eval_run_id, gt_case_id, repeat_index) DO UPDATE SET
                            generation_status = EXCLUDED.generation_status,
                            error = EXCLUDED.error,
                            generated_at = now(),
                            actual_answer = EXCLUDED.actual_answer,
                            fed_chunks = EXCLUDED.fed_chunks,
                            selected_chunks = EXCLUDED.selected_chunks,
                            tokens_used = EXCLUDED.tokens_used,
                            response_time_seconds = EXCLUDED.response_time_seconds,
                            n_searches = EXCLUDED.n_searches,
                            faithfulness = NULL, correctness = NULL,
                            retrieval_recall = NULL, retrieval_precision = NULL, retrieval_f1 = NULL
                        """,
                        (eval_run_id, case_id, repeat_index, *row),
                    )
                job.increment(failed=failed)
        except Exception:
            _set_run_status(conn, eval_run_id, "failed")
            raise
        _set_run_status(conn, eval_run_id, "generated")
    finally:
        conn.close()


def _score_faithfulness(answer: str, chunks: list[dict]) -> float:
    chunk_block = "\n\n".join(f"[{c['chunk_id']}]\n{c['chunk']}" for c in chunks)
    input_text = f"Document chunks:\n\n{chunk_block}\n\nGenerated answer:\n{answer}"

    result = run_llm_request(input_text, FAITHFULNESS_INSTRUCTIONS, FaithfulnessJudgement)
    judgement: FaithfulnessJudgement = result.parsed

    if not judgement.claims:
        return 1.0
    supported = sum(1 for c in judgement.claims if c.supported)
    return supported / len(judgement.claims)


def _score_correctness(question: str, actual_answer: str, expected_answer: str) -> float:
    input_text = (
        f"Question: {question}\n\n"
        f"Generated answer: {actual_answer}\n\n"
        f"Reference answer: {expected_answer}"
    )
    result = run_llm_request(input_text, CORRECTNESS_INSTRUCTIONS, CorrectnessJudgement)
    judgement: CorrectnessJudgement = result.parsed
    return judgement.score


def _score_retrieval(
    expected_chunks: list[dict], fed_chunks: list[dict], selected_chunks: list[dict]
) -> tuple[float, float, float]:
    """Recall is measured on everything that reached the LLM (what never reached it could
    not have been used); precision on what the pipeline selected as relevant."""
    expected_ids = {c["chunk_id"] for c in expected_chunks}
    fed_ids = {c["chunk_id"] for c in fed_chunks}
    selected_ids = {c["chunk_id"] for c in selected_chunks}

    recall = len(expected_ids & fed_ids) / len(expected_ids) if expected_ids else 0.0
    precision = len(expected_ids & selected_ids) / len(selected_ids) if selected_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return recall, precision, f1


def run_scoring(job: Job, eval_run_id: int, rescore_all: bool) -> None:
    """Score the run's generated answers for active ground truth cases on faithfulness,
    correctness and retrieval metrics. By default only answers without scores yet are
    scored; rescore_all overwrites existing scores (e.g. after improving the judge)."""
    conn = get_connection()
    try:
        _set_run_status(conn, eval_run_id, "scoring")
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT erc.id, g.question, g.expected_answer, g.expected_chunks,
                           erc.actual_answer, erc.fed_chunks, erc.selected_chunks
                    FROM eval_run_case erc
                    JOIN gt_case g ON g.id = erc.gt_case_id
                    WHERE erc.eval_run_id = %s
                      AND erc.generation_status = 'ok'
                      AND g.superseded_at IS NULL
                      AND (%s OR erc.faithfulness IS NULL OR erc.correctness IS NULL)
                    ORDER BY erc.id
                    """,
                    (eval_run_id, rescore_all),
                )
                rows = cur.fetchall()
            job.set_total(len(rows))

            for row_id, question, expected_answer, expected_chunks, actual_answer, fed, selected in rows:
                try:
                    faithfulness = _score_faithfulness(actual_answer, fed)
                    correctness = _score_correctness(question, actual_answer, expected_answer)
                    recall, precision, f1 = _score_retrieval(expected_chunks, fed, selected)

                    with conn, conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE eval_run_case
                            SET faithfulness = %s, correctness = %s,
                                retrieval_recall = %s, retrieval_precision = %s, retrieval_f1 = %s
                            WHERE id = %s
                            """,
                            (faithfulness, correctness, recall, precision, f1, row_id),
                        )
                    job.increment()
                except Exception:
                    job.increment(failed=True)
        except Exception:
            _set_run_status(conn, eval_run_id, "failed")
            raise

        with conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE eval_run SET status = 'scored', judge_model = %s, scored_at = now() WHERE id = %s",
                (llm.MODEL, eval_run_id),
            )
    finally:
        conn.close()


def list_eval_runs() -> list[EvalRunResponse]:
    """All evaluation runs, newest first, with aggregates from the eval_run_summary_vw view."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.id, r.created_at, r.label, r.pipeline, r.config, r.status, r.judge_model, r.scored_at,
                       s.case_count, s.failed_count, s.missing_case_count,
                       s.avg_faithfulness, s.avg_correctness,
                       s.avg_retrieval_recall, s.avg_retrieval_precision, s.avg_retrieval_f1,
                       s.avg_tokens_used, s.avg_response_time_seconds, s.avg_searches
                FROM eval_run r
                JOIN eval_run_summary_vw s ON s.eval_run_id = r.id
                ORDER BY r.created_at DESC, r.id DESC
                """
            )
            columns = [d[0] for d in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()
    return [EvalRunResponse(**dict(zip(columns, row))) for row in rows]
