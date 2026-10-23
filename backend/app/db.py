import queue
import threading
import time

import psycopg2
from psycopg2 import extensions

from app.config import settings

# Opening a Postgres connection to Supabase takes ~0.3 s (TCP, TLS, auth), and the backend does
# many small queries per request, so connections are kept open and reused.
MAX_CONNECTIONS = 10  # open at the same time; further callers wait for a free one
MAX_IDLE_SECONDS = 300  # an idle connection older than this is closed instead of reused, since the server or a proxy may have dropped it
KEEPALIVE_IDLE_SECONDS = 30

_idle: queue.LifoQueue = queue.LifoQueue()
_slots = threading.BoundedSemaphore(MAX_CONNECTIONS)


class _PooledConnection(extensions.connection):
    """A connection whose close() hands it back to the pool; callers keep the usual
    `conn = get_connection()` ... `finally: conn.close()` shape."""

    in_pool = False
    idle_since = 0.0

    def close(self) -> None:
        if self.in_pool:
            return
        _release(self)

    def close_for_real(self) -> None:
        extensions.connection.close(self)


def _release(conn: _PooledConnection) -> None:
    try:
        if not conn.closed and conn.get_transaction_status() != extensions.TRANSACTION_STATUS_IDLE:
            conn.rollback()  # the caller left a transaction open (read-only callers never commit)
    except psycopg2.Error:
        conn.close_for_real()
    conn.in_pool = True
    if conn.closed:
        _slots.release()
        return
    conn.idle_since = time.monotonic()
    _idle.put(conn)
    _slots.release()


def get_connection() -> _PooledConnection:
    """A connection from the pool, opened if none is idle. Give it back with close()."""
    _slots.acquire()
    try:
        while True:
            try:
                conn = _idle.get_nowait()
            except queue.Empty:
                conn = psycopg2.connect(
                    settings.supabase_db_url,
                    connection_factory=_PooledConnection,
                    keepalives=1,
                    keepalives_idle=KEEPALIVE_IDLE_SECONDS,
                )
                break
            if not conn.closed and time.monotonic() - conn.idle_since < MAX_IDLE_SECONDS:
                break
            conn.close_for_real()
    except BaseException:
        _slots.release()
        raise
    conn.in_pool = False
    return conn
