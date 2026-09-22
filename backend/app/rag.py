from pinecone import Pinecone

from app.config import settings
from app.llm import client as openai_client
from app.models import RetrievedChunk

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

CHUNK_SIZE = 1000  # characters per chunk — placeholder; revisit once real documents are used
CHUNK_OVERLAP = 100

_pc = Pinecone(api_key=settings.pinecone_api_key)
index = _pc.Index(settings.pinecone_index_name)


def _chunk_text(content: str) -> list[str]:
    """Naive fixed-size character chunker. Placeholder — swap for a smarter
    (sentence/paragraph-aware, token-based) strategy once we have real documents."""
    chunks = []
    start = 0
    while start < len(content):
        end = start + CHUNK_SIZE
        chunks.append(content[start:end])
        start = end - CHUNK_OVERLAP
    return chunks


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via OpenAI. Pinecone never sees the OpenAI key —
    embedding happens here, in our own code, before anything reaches Pinecone."""
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def ingest_document(doc_id: str, filename: str, content: str) -> int:
    """Chunk, embed, and upsert a document into Pinecone. Returns the number of chunks ingested."""
    chunks = _chunk_text(content)
    embeddings = _embed(chunks)

    vectors = [
        {
            "id": f"{doc_id}-{i}",
            "values": embedding,
            "metadata": {
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i,
                "text": chunk,
            },
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    index.upsert(vectors=vectors)
    return len(vectors)


def retrieve(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    """Embed the query and return the top-k most similar chunks from Pinecone."""
    query_embedding = _embed([query])[0]

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    return [
        RetrievedChunk(
            doc_id=match.metadata["doc_id"],
            filename=match.metadata["filename"],
            chunk_index=match.metadata["chunk_index"],
            text=match.metadata["text"],
            score=match.score,
        )
        for match in results.matches
    ]
