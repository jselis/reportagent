from psycopg2.extras import Json

from app.db import get_connection
from app.research import research


def run_evaluation() -> tuple[list[int], list[int]]:
    """Re-run every ground truth case's question through the current retrieval +
    answer pipeline, storing the result as actual_answer/actual_chunks for comparison
    against the curated expected_answer/expected_chunks. Returns (evaluated, failed) ids."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, question FROM ground_truth_cases")
            cases = cur.fetchall()

        evaluated: list[int] = []
        failed: list[int] = []

        for case_id, question in cases:
            try:
                response, chunks = research(question)
            except Exception:
                failed.append(case_id)
                continue

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
            evaluated.append(case_id)

        return evaluated, failed
    finally:
        conn.close()
