import re

from app.llm import run_llm_request
from app.models import (
    Answer,
    AskResponse,
    DocumentSource,
    GroundedAnswer,
    RetrievedChunk,
    Source,
)
from app.rag import retrieve

TOP_K = 5
SNIPPET_FALLBACK_CHARS = 200

INSTRUCTIONS = """You answer questions using ONLY the document chunks provided in the user's message. Each chunk is labelled with its id, like [id: some-document-3].

Rules:
- Base your answer strictly on the information in the provided chunks. Do not use outside knowledge, and do not guess.
- If the chunks do not contain the answer, say clearly that the answer is not in the relevant matches in the documents. In that case return no citations and put no reference markers in the text. Use a low confidence.
- Otherwise, answer directly and concisely, in the language of the question.
- Right after each statement that comes from a chunk, put that chunk's exact id in square brackets, copied from its label, e.g. "The tower is 330 meters tall. [some-document-3]". If a statement is supported by several chunks, use one bracket per id, e.g. [doc-a-0][doc-b-2]. Never invent ids and never use plain numbers as markers.
- In `citations`, list each distinct chunk you used, once: `source_id` is the chunk's id copied exactly, and `quote` is the shortest verbatim excerpt (one or two sentences, copied exactly) from that chunk that supports your statement.
- Set `confidence` to how well the chunks support your answer."""

_MARKER = re.compile(r"\[([^\[\]]+)\]")


def _build_input(question: str, chunks: list[RetrievedChunk]) -> str:
    if chunks:
        chunk_block = "\n\n".join(f"[id: {c.id}]\n{c.text}" for c in chunks)
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


def _build_answer(grounded: GroundedAnswer, chunks: list[RetrievedChunk]) -> Answer:
    """Replace the chunk ids the model wrote in the text (e.g. [2-1]) by 1-based
    numbers in order of first appearance, and build `sources` in that same order.
    Deterministic: [n] always matches sources[n-1], each carrying its real index id."""
    by_id = {c.id: c for c in chunks}
    quote_by_id: dict[str, str] = {}
    for citation in grounded.citations:
        quote_by_id.setdefault(citation.source_id, citation.quote)

    sources: list[Source] = []
    number_of: dict[str, int] = {}

    def _replace(match: re.Match) -> str:
        chunk = by_id.get(match.group(1).strip())
        if chunk is None:
            return match.group(0)  # not a known chunk id: leave the text untouched
        if chunk.id not in number_of:
            sources.append(
                DocumentSource(
                    id=chunk.id,
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    snippet=_snippet(chunk, quote_by_id.get(chunk.id, "")),
                )
            )
            number_of[chunk.id] = len(sources)
        return f"[{number_of[chunk.id]}]"

    text = _MARKER.sub(_replace, grounded.text)
    return Answer(text=text, sources=sources, confidence=grounded.confidence)


def research_and_answer(question: str) -> AskResponse:
    """Answer a question using only the top-matching document chunks from Pinecone."""
    chunks = retrieve(question, top_k=TOP_K)
    result = run_llm_request(_build_input(question, chunks), INSTRUCTIONS, GroundedAnswer)

    return AskResponse(
        answer=_build_answer(result.parsed, chunks),
        tokens_used=result.tokens_used,
        response_time_seconds=result.response_time_seconds,
        ttft_seconds=result.ttft_seconds,
        raw_llm_output=result.raw_output,
    )
