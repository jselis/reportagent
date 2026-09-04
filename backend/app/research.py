from app.llm import run_llm_request
from app.models import Answer, AskResponse

INSTRUCTIONS = "Answer the user's question directly and concisely."


def research_and_answer(question: str) -> AskResponse:
    """Ask the model a question and return a direct answer."""
    result = run_llm_request(question, INSTRUCTIONS, Answer)
    return AskResponse(
        answer=result.parsed,
        tokens_used=result.tokens_used,
        response_time_seconds=result.response_time_seconds,
        ttft_seconds=result.ttft_seconds,
    )
