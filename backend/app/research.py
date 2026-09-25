from app.llm import run_llm_request
from app.models import Answer, AskResponse, RetrievedChunk
from app.rag import retrieve

TOP_K = 5

INSTRUCTIONS = """You answer questions using ONLY the document chunks provided in the user's message.

Rules:
- Base your answer strictly on the information in the provided chunks. Do not use outside knowledge, and do not guess.
- If the chunks do not contain the answer, say clearly that the answer is not in the relevant matches in the documents. In that case return no sources and a low confidence.
- Otherwise, answer directly and concisely, in the language of the question.
- In `sources`, list only the chunks you actually used, with the title formatted as "<doc_id> (chunk <chunk_index>)" and url set to null.
- Set `confidence` to how well the chunks support your answer."""


def _build_input(question: str, chunks: list[RetrievedChunk]) -> str:
    if chunks:
        chunk_block = "\n\n".join(
            f"[doc_id={c.doc_id}, chunk_index={c.chunk_index}]\n{c.text}" for c in chunks
        )
    else:
        chunk_block = "(no chunks were retrieved)"

    return f"Document chunks:\n\n{chunk_block}\n\nQuestion: {question}"


def research_and_answer(question: str) -> AskResponse:
    """Answer a question using only the top-matching document chunks from Pinecone."""
    chunks = retrieve(question, top_k=TOP_K)
    result = run_llm_request(_build_input(question, chunks), INSTRUCTIONS, Answer)
    return AskResponse(
        answer=result.parsed,
        tokens_used=result.tokens_used,
        response_time_seconds=result.response_time_seconds,
        ttft_seconds=result.ttft_seconds,
    )
