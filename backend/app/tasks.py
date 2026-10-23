from app.db import get_connection
from app.models import Task

_COLUMNS = "id, type, title, created_at"


def create_task(task_type: str, title: str) -> Task:
    conn = get_connection()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO task (type, title) VALUES (%s, %s) RETURNING {_COLUMNS}",
                (task_type, title),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    return Task(id=row[0], type=row[1], title=row[2], created_at=row[3])


def list_tasks() -> list[Task]:
    """All tasks, newest first."""
    conn = get_connection()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(f"SELECT {_COLUMNS} FROM task ORDER BY created_at DESC, id DESC")
            rows = cur.fetchall()
    finally:
        conn.close()
    return [Task(id=r[0], type=r[1], title=r[2], created_at=r[3]) for r in rows]


def get_task(task_id: int) -> Task | None:
    conn = get_connection()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(f"SELECT {_COLUMNS} FROM task WHERE id = %s", (task_id,))
            row = cur.fetchone()
    finally:
        conn.close()
    return Task(id=row[0], type=row[1], title=row[2], created_at=row[3]) if row else None
