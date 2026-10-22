from psycopg2.extras import Json

from app.db import get_connection
from app.models import (
    ActiveGroundTruthCaseNotFoundError,
    CreateGroundTruthRequest,
    GroundTruthCaseListItem,
    GroundTruthCaseResponse,
)


def _mark_superseded(cur, case_id: int, superseded_by: int | None) -> None:
    cur.execute(
        """
        UPDATE gt_case SET superseded_at = now(), superseded_by = %s
        WHERE id = %s AND superseded_at IS NULL
        """,
        (superseded_by, case_id),
    )
    if cur.rowcount == 0:
        raise ActiveGroundTruthCaseNotFoundError(case_id)


def create_ground_truth_case(request: CreateGroundTruthRequest) -> GroundTruthCaseResponse:
    """Insert a new case. If it corrects an existing one, that one is marked outdated in
    the same transaction (cases are never edited or deleted)."""
    conn = get_connection()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO gt_case (question, expected_answer, expected_chunks)
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
            if request.replaces_id is not None:
                _mark_superseded(cur, request.replaces_id, case_id)
    finally:
        conn.close()

    return GroundTruthCaseResponse(
        id=case_id,
        question=request.question,
        expected_answer=request.expected_answer,
        expected_chunks=request.expected_chunks,
    )


def supersede_ground_truth_case(case_id: int) -> None:
    """Mark a case outdated without a replacement."""
    conn = get_connection()
    try:
        with conn, conn.cursor() as cur:
            _mark_superseded(cur, case_id, None)
    finally:
        conn.close()


def list_ground_truth_cases() -> list[GroundTruthCaseListItem]:
    """All cases, active ones first, then outdated ones."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, question, expected_answer, jsonb_array_length(expected_chunks),
                       created_at, superseded_at, superseded_by
                FROM gt_case
                ORDER BY superseded_at IS NOT NULL, id
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [
        GroundTruthCaseListItem(
            id=id_, question=question, expected_answer=expected_answer, expected_chunk_count=count,
            created_at=created_at, superseded_at=superseded_at, superseded_by=superseded_by,
        )
        for id_, question, expected_answer, count, created_at, superseded_at, superseded_by in rows
    ]
