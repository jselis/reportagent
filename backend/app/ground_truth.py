from psycopg2.extras import Json

from app.db import get_connection
from app.models import CreateGroundTruthRequest, GroundTruthCaseResponse


def create_ground_truth_case(request: CreateGroundTruthRequest) -> GroundTruthCaseResponse:
    conn = get_connection()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ground_truth_cases (question, expected_answer, expected_chunks)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (
                    request.question,
                    request.expected_answer,
                    Json([chunk.model_dump() for chunk in request.expected_chunks]),
                ),
            )
            (case_id,) = cur.fetchone()
    finally:
        conn.close()

    return GroundTruthCaseResponse(
        id=case_id,
        question=request.question,
        expected_answer=request.expected_answer,
        expected_chunks=request.expected_chunks,
    )
