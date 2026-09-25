from app.llm import run_llm_request
from app.models import (
    Answer,
    AskResponse,
    Citation,
    DocumentSource,
    GroundedAnswer,
    RetrievedChunk,
    Source,
)
from app.rag import retrieve

TOP_K = 5
SNIPPET_FALLBACK_CHARS = 200

INSTRUCTIONS = """You answer questions using ONLY the numbered document chunks provided in the user's message.

Rules:
- Base your answer strictly on the information in the provided chunks. Do not use outside knowledge, and do not guess.
- If the chunks do not contain the answer, say clearly that the answer is not in the relevant matches in the documents. In that case return no citations and a low confidence.
- Otherwise, answer directly and concisely, in the language of the question.
- In `citations`, list each chunk you actually used: `chunk_ref` is the chunk's number (the N in "[N]"), and `quote` is the shortest verbatim excerpt (one or two sentences, copied exactly) from that chunk that supports your answer.
- Set `confidence` to how well the chunks support your answer."""


def _build_input(question: str, chunks: list[RetrievedChunk]) -> str:
    if chunks:
        chunk_block = "\n\n".join(f"[{n}]\n{c.text}" for n, c in enumerate(chunks, start=1))
    else:
        chunk_block = "(no chunks were retrieved)"

    return f"Document chunks:\n\n{chunk_block}\n\nQuestion: {question}"


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _snippet(chunk: RetrievedChunk, quote: str) -> str:
    """Use the model's quote only if it really appears in the chunk; otherwise
    fall back to the start of the chunk, so a snippet is never invented."""
    normalized_quote = _normalize(quote)
    if normalized_quote and normalized_quote in _normalize(chunk.text):
        return normalized_quote
    return _normalize(chunk.text)[:SNIPPET_FALLBACK_CHARS]


def _build_sources(citations: list[Citation], chunks: list[RetrievedChunk]) -> list[Source]:
    sources: list[Source] = []
    seen: set[int] = set()
    for citation in citations:
        if citation.chunk_ref in seen or not 1 <= citation.chunk_ref <= len(chunks):
            continue
        seen.add(citation.chunk_ref)
        chunk = chunks[citation.chunk_ref - 1]
        sources.append(
            DocumentSource(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                snippet=_snippet(chunk, citation.quote),
            )
        )
    return sources


def research_and_answer(question: str) -> AskResponse:
    """Answer a question using only the top-matching document chunks from Pinecone."""
    chunks = retrieve(question, top_k=TOP_K)
    result = run_llm_request(_build_input(question, chunks), INSTRUCTIONS, GroundedAnswer)
    grounded: GroundedAnswer = result.parsed

    return AskResponse(
        answer=Answer(
            text=grounded.text,
            sources=_build_sources(grounded.citations, chunks),
            confidence=grounded.confidence,
        ),
        tokens_used=result.tokens_used,
        response_time_seconds=result.response_time_seconds,
        ttft_seconds=result.ttft_seconds,
    )
