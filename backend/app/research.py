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
- In `text`, put numeric reference markers like [1], [2] directly after the statements they support. A marker's number is the position of the matching entry in your `citations` list: the first citation is [1], the second is [2], and so on. Number citations in the order they are first used in the text.
- In `citations`, list each distinct chunk you actually used, once: `source_id` is the chunk's id copied exactly from its label, and `quote` is the shortest verbatim excerpt (one or two sentences, copied exactly) from that chunk that supports the statement. Every citation must be referenced by a marker in the text.
- Set `confidence` to how well the chunks support your answer."""

_MARKER = re.compile(r"\[(\d+)\]")


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
    """Turn the model's output into an Answer whose [n] markers are guaranteed to
    match the n-th entry of `sources`, each carrying its real index id."""
    by_id = {c.id: c for c in chunks}
    sources: list[Source] = []
    number_of_source: dict[str, int] = {}
    renumber: dict[int, int] = {}  # model's citation number -> final source number

    for model_number, citation in enumerate(grounded.citations, start=1):
        chunk = by_id.get(citation.source_id)
        if chunk is None:
            continue  # the model cited an id that was never retrieved
        if chunk.id not in number_of_source:
            sources.append(
                DocumentSource(
                    id=chunk.id,
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    snippet=_snippet(chunk, citation.quote),
                )
            )
            number_of_source[chunk.id] = len(sources)
        renumber[model_number] = number_of_source[chunk.id]

    dropped_marker = False

    def _replace(match: re.Match) -> str:
        nonlocal dropped_marker
        final_number = renumber.get(int(match.group(1)))
        if final_number is None:
            dropped_marker = True
            return ""
        return f"[{final_number}]"

    text = _MARKER.sub(_replace, grounded.text)
    if dropped_marker:
        text = re.sub(r"\s+([.,;:!?])", r"\1", re.sub(r" {2,}", " ", text)).strip()

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
    )
