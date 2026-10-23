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
MIN_QUOTE_FRAGMENT_CHARS = 5

INSTRUCTIONS = """You answer questions using ONLY the document chunks provided in the user's message. Each chunk is labelled with its id, like [id: some-document-3].

Rules:
- Base your answer strictly on the information in the provided chunks. Do not use outside knowledge, and do not guess.
- If the chunks do not contain the answer, say clearly that the answer is not in the relevant matches in the documents. In that case return no citations and put no reference markers in the text. Use a low confidence.
- Otherwise, answer directly and concisely, in the language of the question.
- Right after each statement that comes from a chunk, put that chunk's exact id in square brackets, copied from its label, e.g. "The tower is 330 meters tall. [some-document-3]". If a statement is supported by several chunks, use one bracket per id, e.g. [doc-a-0][doc-b-2]. Never invent ids and never use plain numbers as markers.
- In `citations`, list each distinct chunk you used, once: `source_id` is the chunk's id copied exactly, and `quote` is a short verbatim excerpt (copied exactly) from that chunk that supports your statement. Keep only the passages that matter: you may skip irrelevant stretches with "...", e.g. "The tower was designed by Gustave Eiffel ... and is 330 meters tall". Every part between the "..." must be copied exactly, in the original order.
- Set `confidence` to how well the chunks support your answer."""

_MARKER = re.compile(r"\[([^\[\]]+)\]")
_ELLIPSIS = re.compile(r"\s*(?:\.{3,}|…)\s*")
UNSUPPORTED_MARKER = "[UNSUPPORTED]"


def _build_input(question: str, chunks: list[RetrievedChunk]) -> str:
    if chunks:
        chunk_block = "\n\n".join(f"[id: {c.id}]\n{c.text}" for c in chunks)
    else:
        chunk_block = "(no chunks were retrieved)"

    return f"Document chunks:\n\n{chunk_block}\n\nQuestion: {question}"


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _quote_in_chunk(quote: str, chunk_text: str) -> bool:
    """True if the quote's fragments, separated by "..." (or "…") omissions, all appear verbatim
    in the chunk, in that order and without overlapping. A fragment must be long enough to
    mean something, so a quote like "... a ..." cannot match any chunk."""
    fragments = [f for f in _ELLIPSIS.split(_normalize(quote)) if f]
    if not fragments or any(len(f) < MIN_QUOTE_FRAGMENT_CHARS for f in fragments):
        return False
    pattern = ".*?".join(re.escape(f) for f in fragments)
    return re.search(pattern, _normalize(chunk_text)) is not None


def _snippet(chunk: RetrievedChunk, quote: str) -> str:
    """Use the model's quote only if it really appears in the chunk (omissions allowed);
    otherwise fall back to the start of the chunk, so a snippet is never invented."""
    if _quote_in_chunk(quote, chunk.text):
        return _normalize(quote)
    return _normalize(chunk.text)[:SNIPPET_FALLBACK_CHARS]


def build_answer(
    grounded: GroundedAnswer, chunks: list[RetrievedChunk], previous_sources: list[Source]
) -> tuple[Answer, list[str]]:
    """Replace the chunk ids the model wrote in the text (e.g. [2-1]) by numbers, and build
    `sources`. Numbers are stable: previous_sources (from earlier turns of a chat, in number
    order) keep their numbers, and a chunk cited for the first time gets the next number, so
    `sources` is cumulative and [n] always matches sources[n-1]. A marker that is not the id of
    a chunk the model was shown is replaced by [UNSUPPORTED]. Returns the answer and the ids of
    the chunks cited in this answer."""
    by_id = {c.id: c for c in chunks}
    quote_by_id: dict[str, str] = {}
    for citation in grounded.citations:
        quote_by_id.setdefault(citation.source_id, citation.quote)

    sources: list[Source] = list(previous_sources)
    number_of: dict[str, int] = {s.id: i + 1 for i, s in enumerate(sources) if s.type == "document"}
    cited_ids: list[str] = []

    def _replace(match: re.Match) -> str:
        chunk = by_id.get(match.group(1).strip())
        if chunk is None:
            return UNSUPPORTED_MARKER
        if chunk.id not in cited_ids:
            cited_ids.append(chunk.id)
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
    return Answer(text=text, sources=sources, confidence=grounded.confidence), cited_ids


def research(question: str, top_k: int) -> tuple[AskResponse, list[RetrievedChunk]]:
    """Answer a question using only the top_k best-matching document chunks from Pinecone.

    Returns the response alongside the full set of chunks retrieved for it (not just
    the ones the model ended up citing), for callers that need that (e.g. evaluation)."""
    chunks = retrieve(question, top_k=top_k)
    result = run_llm_request(_build_input(question, chunks), INSTRUCTIONS, GroundedAnswer)

    response = AskResponse(
        answer=build_answer(result.parsed, chunks, [])[0],
        tokens_used=result.tokens_used,
        response_time_seconds=result.response_time_seconds,
        ttft_seconds=result.ttft_seconds,
        raw_llm_output=result.raw_output,
    )
    return response, chunks


def research_and_answer(question: str) -> AskResponse:
    """Answer a question using only the top-matching document chunks from Pinecone."""
    response, _ = research(question, TOP_K)
    return response
