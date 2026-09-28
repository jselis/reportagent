from app.db import get_connection
from app.models import CreateGroundTruthRequest, GroundTruthCaseResponse


def create_ground_truth_case(request: CreateGroundTruthRequest) -> GroundTruthCaseResponse:
    conn = get_connection()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ground_truth_cases (question, expected_answer, expected_document_ids)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (request.question, request.expected_answer, request.expected_document_ids),
            )
            (case_id,) = cur.fetchone()
    finally:
        conn.close()

    return GroundTruthCaseResponse(
        id=case_id,
        question=request.question,
        expected_answer=request.expected_answer,
        expected_document_ids=request.expected_document_ids,
    )
