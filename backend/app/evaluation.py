from psycopg2.extras import Json

from app.db import get_connection
from app.jobs import Job
from app.llm import run_llm_request
from app.models import CorrectnessJudgement, EvaluationSummaryResponse, FaithfulnessJudgement
from app.research import research

FAITHFULNESS_INSTRUCTIONS = """You assess whether an AI-generated answer is faithful to the document chunks it was based on.

Break the answer down into its individual atomic factual claims. Ignore citation markers like [1], [2], and do not treat a "the answer is not in the documents"-style refusal as a claim (return an empty claims list for it).

For each claim, decide whether it is directly supported by the provided chunks — `supported: true` if so, `false` if the chunks don't back it up or it isn't in them at all."""

CORRECTNESS_INSTRUCTIONS = """You judge whether a generated answer is correct compared to a reference (ground truth) answer, for the same question. Ignore citation markers like [1], [2] in the generated answer.

Score:
- 0: the generated answer is wrong or contradicts the reference answer.
- 0.5: partially correct — it captures the main point but has caveats, is incomplete, or has minor inaccuracies.
- 1: correct — semantically equivalent to the reference answer, even if worded differently."""


def run_generation(job: Job) -> None:
    """Re-run every ground truth case's question through the current retrieval + answer
    pipeline, storing the result as actual_answer/actual_chunks."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, question FROM ground_truth_cases")
            cases = cur.fetchall()
        job.set_total(len(cases))

        for case_id, question in cases:
            try:
                response, chunks = research(question)
                actual_chunks = [{"chunk_id": c.id, "chunk": c.text} for c in chunks]

                with conn, conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE ground_truth_cases
                        SET actual_answer = %s, actual_chunks = %s
                        WHERE id = %s
                        """,
                        (response.answer.text, Json(actual_chunks), case_id),
                    )
                job.increment()
            except Exception:
                job.increment(failed=True)
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


def _score_retrieval(expected_chunks: list[dict], actual_chunks: list[dict]) -> tuple[float, float, float]:
    expected_ids = {c["chunk_id"] for c in expected_chunks}
    actual_ids = {c["chunk_id"] for c in actual_chunks}
    overlap = expected_ids & actual_ids

    recall = len(overlap) / len(expected_ids) if expected_ids else 0.0
    precision = len(overlap) / len(actual_ids) if actual_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return recall, precision, f1


def run_scoring(job: Job) -> None:
    """Score every ground truth case that has already been through generation
    (has actual_answer/actual_chunks) on faithfulness, correctness, and retrieval metrics."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, question, expected_answer, expected_chunks, actual_answer, actual_chunks
                FROM ground_truth_cases
                WHERE actual_answer IS NOT NULL AND actual_chunks IS NOT NULL
                """
            )
            cases = cur.fetchall()
        job.set_total(len(cases))

        for case_id, question, expected_answer, expected_chunks, actual_answer, actual_chunks in cases:
            try:
                faithfulness = _score_faithfulness(actual_answer, actual_chunks)
                correctness = _score_correctness(question, actual_answer, expected_answer)
                recall, precision, f1 = _score_retrieval(expected_chunks, actual_chunks)

                with conn, conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE ground_truth_cases
                        SET faithfulness = %s, correctness = %s,
                            retrieval_recall = %s, retrieval_precision = %s, retrieval_f1 = %s
                        WHERE id = %s
                        """,
                        (faithfulness, correctness, recall, precision, f1, case_id),
                    )
                job.increment()
            except Exception:
                job.increment(failed=True)
    finally:
        conn.close()


def get_evaluation_summary() -> EvaluationSummaryResponse:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    count(*),
                    avg(faithfulness), avg(correctness),
                    avg(retrieval_recall), avg(retrieval_precision), avg(retrieval_f1)
                FROM ground_truth_cases
                """
            )
            count, avg_f, avg_c, avg_r, avg_p, avg_f1 = cur.fetchone()
    finally:
        conn.close()

    return EvaluationSummaryResponse(
        case_count=count,
        avg_faithfulness=avg_f,
        avg_correctness=avg_c,
        avg_retrieval_recall=avg_r,
        avg_retrieval_precision=avg_p,
        avg_retrieval_f1=avg_f1,
    )
