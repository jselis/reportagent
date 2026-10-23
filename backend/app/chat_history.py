from app.agent import APP_NAME, TURNS_KEY, USER_ID
from app.db import get_connection
from app.models import (
    AgentSessionListResponse,
    AgentSessionNotFoundError,
    AgentSessionResponse,
    AgentSessionSummary,
    AgentSessionTurn,
)


def list_agent_sessions() -> AgentSessionListResponse:
    """The Agent tab's past chats, most recently active first. A chat without a saved
    turn (its first question failed) isn't listed."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, created_at, updated_at,
                       jsonb_array_length(state -> %s), state -> %s -> 0 ->> 'question'
                FROM chat_session
                WHERE app_name = %s AND user_id = %s AND state ? %s
                ORDER BY updated_at DESC
                """,
                (TURNS_KEY, TURNS_KEY, APP_NAME, USER_ID, TURNS_KEY),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return AgentSessionListResponse(
        sessions=[
            AgentSessionSummary(id=id_, created_at=created, updated_at=updated, turn_count=count, title=title)
            for id_, created, updated, count, title in rows
        ]
    )


def get_agent_session(session_id: str) -> AgentSessionResponse:
    """A past chat with all its turns, as it was when it was last used."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT state FROM chat_session WHERE id = %s AND app_name = %s AND user_id = %s",
                (session_id, APP_NAME, USER_ID),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        raise AgentSessionNotFoundError(session_id)
    return AgentSessionResponse(
        session_id=session_id,
        turns=[AgentSessionTurn(**turn) for turn in row[0].get(TURNS_KEY, [])],
    )
