from app.db import get_connection
from app.rag import index


def upsert_document_metadata(document_id: str, topic: str) -> None:
    """Record (or update) a document's metadata, marking it as in sync with Pinecone."""
    conn = get_connection()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO document_metadata (document_id, topic, synced_at)
                VALUES (%s, %s, now())
                ON CONFLICT (document_id)
                DO UPDATE SET topic = EXCLUDED.topic, synced_at = now()
                """,
                (document_id, topic),
            )
    finally:
        conn.close()


def sync_pending_metadata() -> tuple[list[str], list[str]]:
    """Push metadata changes made directly in the database to every chunk of the
    affected documents in Pinecone. Returns (synced, failed) document ids."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT document_id, topic FROM document_metadata
                WHERE sync_needed_at IS NOT NULL
                  AND (synced_at IS NULL OR sync_needed_at > synced_at)
                """
            )
            pending = cur.fetchall()

        synced: list[str] = []
        failed: list[str] = []

        for document_id, topic in pending:
            try:
                index.update(filter={"doc_id": {"$eq": document_id}}, set_metadata={"topic": topic})
            except Exception:
                failed.append(document_id)
                continue

            with conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE document_metadata SET synced_at = now() WHERE document_id = %s",
                    (document_id,),
                )
            synced.append(document_id)

        return synced, failed
    finally:
        conn.close()
