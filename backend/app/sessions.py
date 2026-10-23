import asyncio
import time
import uuid
from typing import Any, Optional

from google.adk.events import Event
from google.adk.sessions import Session
from google.adk.sessions.base_session_service import (
    BaseSessionService,
    GetSessionConfig,
    ListSessionsResponse,
)
from google.adk.sessions.state import State
from psycopg2.extras import Json, execute_values

from app.db import get_connection


class ChatSessionService(BaseSessionService):
    """ADK session service for ONE turn of a persisted chat. The chat is loaded from our own
    Postgres tables once (chat_session for the state, chat_event for the conversation history),
    the turn runs on the in-memory session, and save_turn() then writes everything the turn
    added in a single transaction. Nothing reads the database during a turn, so writing as the
    run goes would only add round trips; and a turn that fails saves nothing, so a chat never
    contains half a turn. Only session-scoped state is supported."""

    def __init__(self) -> None:
        self._session: Session | None = None
        self._persisted_events = 0  # how many of the session's events are already in the database

    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Session:
        """A new chat, in memory only: its row is written by save_turn()."""
        self._session = Session(
            id=session_id or str(uuid.uuid4()), app_name=app_name, user_id=user_id,
            state=state or {}, events=[], last_update_time=time.time(),
        )
        self._persisted_events = 0
        return self._session

    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Optional[Session]:
        """This turn's session: loaded from the database the first time, then served from memory."""
        if self._session is None or self._session.id != session_id:
            self._session = await asyncio.to_thread(self._load, app_name, user_id, session_id)
            self._persisted_events = len(self._session.events) if self._session else 0
        return self._session

    async def list_sessions(self, *, app_name: str, user_id: Optional[str] = None) -> ListSessionsResponse:
        raise NotImplementedError("Chats are listed by app.chat_history")

    async def delete_session(self, *, app_name: str, user_id: str, session_id: str) -> None:
        raise NotImplementedError("Chats are deleted by app.chat_history")

    async def append_event(self, session: Session, event: Event) -> Event:
        """Update the in-memory session as the base class does; saving happens in save_turn()."""
        if event.partial:
            return event
        event = await super().append_event(session, event)
        session.last_update_time = time.time()
        return event

    def save_turn(self, turns_key: str, turn: dict) -> None:
        """Add the turn record to the session state, then write the new events and the state."""
        session = self._session
        session.state[turns_key] = [*session.state.get(turns_key, []), turn]
        persistent_state = {k: v for k, v in session.state.items() if not k.startswith(State.TEMP_PREFIX)}
        new_events = session.events[self._persisted_events:]

        conn = get_connection()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_session (id, app_name, user_id, state) VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET state = EXCLUDED.state, updated_at = now()
                    """,
                    (session.id, session.app_name, session.user_id, Json(persistent_state)),
                )
                if new_events:
                    execute_values(
                        cur,
                        "INSERT INTO chat_event (chat_session_id, event) VALUES %s",
                        [(session.id, Json(e.model_dump(mode="json", exclude_none=True))) for e in new_events],
                    )
        finally:
            conn.close()
        self._persisted_events = len(session.events)

    # --- blocking database access, run in a worker thread ---

    def _load(self, app_name: str, user_id: str, session_id: str) -> Optional[Session]:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT state, extract(epoch FROM updated_at) FROM chat_session
                    WHERE id = %s AND app_name = %s AND user_id = %s
                    """,
                    (session_id, app_name, user_id),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                state, updated_at = row
                cur.execute(
                    "SELECT event FROM chat_event WHERE chat_session_id = %s ORDER BY id", (session_id,)
                )
                events = [Event.model_validate(r[0]) for r in cur.fetchall()]
        finally:
            conn.close()
        return Session(
            id=session_id, app_name=app_name, user_id=user_id, state=state, events=events,
            last_update_time=float(updated_at),
        )
