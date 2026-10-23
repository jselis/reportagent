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
from psycopg2.extras import Json

from app.db import get_connection


class PostgresSessionService(BaseSessionService):
    """ADK session service that keeps chat sessions in our own Postgres tables
    (chat_session for the session state, chat_event for the conversation history), so a chat
    survives restarts and redeploys. Only session-scoped state is supported."""

    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Session:
        return await asyncio.to_thread(
            self._create_session, app_name, user_id, state or {}, session_id or str(uuid.uuid4())
        )

    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Optional[Session]:
        return await asyncio.to_thread(self._get_session, app_name, user_id, session_id, config)

    async def list_sessions(self, *, app_name: str, user_id: Optional[str] = None) -> ListSessionsResponse:
        return await asyncio.to_thread(self._list_sessions, app_name, user_id)

    async def delete_session(self, *, app_name: str, user_id: str, session_id: str) -> None:
        await asyncio.to_thread(self._delete_session, app_name, user_id, session_id)

    async def append_event(self, session: Session, event: Event) -> Event:
        """Update the in-memory session as the base class does, then persist the event and
        the resulting (non-temporary) session state."""
        if event.partial:
            return event
        event = await super().append_event(session, event)
        persistent_state = {k: v for k, v in session.state.items() if not k.startswith(State.TEMP_PREFIX)}
        await asyncio.to_thread(self._persist_event, session.id, event, persistent_state)
        session.last_update_time = time.time()
        return event

    # --- blocking database access, run in a worker thread ---

    def _create_session(self, app_name: str, user_id: str, state: dict, session_id: str) -> Session:
        conn = get_connection()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chat_session (id, app_name, user_id, state) VALUES (%s, %s, %s, %s)",
                    (session_id, app_name, user_id, Json(state)),
                )
        finally:
            conn.close()
        return Session(
            id=session_id, app_name=app_name, user_id=user_id, state=state, events=[],
            last_update_time=time.time(),
        )

    def _get_session(
        self, app_name: str, user_id: str, session_id: str, config: Optional[GetSessionConfig]
    ) -> Optional[Session]:
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

        if config is not None:
            if config.after_timestamp is not None:
                events = [e for e in events if e.timestamp >= config.after_timestamp]
            if config.num_recent_events is not None:
                events = events[-config.num_recent_events:] if config.num_recent_events > 0 else []

        return Session(
            id=session_id, app_name=app_name, user_id=user_id, state=state, events=events,
            last_update_time=float(updated_at),
        )

    def _list_sessions(self, app_name: str, user_id: Optional[str]) -> ListSessionsResponse:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, user_id, extract(epoch FROM updated_at) FROM chat_session
                    WHERE app_name = %s AND (%s::text IS NULL OR user_id = %s)
                    ORDER BY updated_at
                    """,
                    (app_name, user_id, user_id),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        return ListSessionsResponse(
            sessions=[
                Session(id=id_, app_name=app_name, user_id=uid, state={}, events=[], last_update_time=float(ts))
                for id_, uid, ts in rows
            ]
        )

    def _delete_session(self, app_name: str, user_id: str, session_id: str) -> None:
        conn = get_connection()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chat_session WHERE id = %s AND app_name = %s AND user_id = %s",
                    (session_id, app_name, user_id),
                )
        finally:
            conn.close()

    def _persist_event(self, session_id: str, event: Event, state: dict) -> None:
        conn = get_connection()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chat_event (chat_session_id, event) VALUES (%s, %s)",
                    (session_id, Json(event.model_dump(mode="json", exclude_none=True))),
                )
                cur.execute(
                    "UPDATE chat_session SET state = %s, updated_at = now() WHERE id = %s",
                    (Json(state), session_id),
                )
        finally:
            conn.close()
