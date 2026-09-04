from app.llm import run_llm_request
from app.models import Summary, SummarizeResponse

INSTRUCTIONS = "Summarize the user's text concisely, preserving the key points."


def summarize_text(text: str) -> SummarizeResponse:
    """Summarize the given text."""
    result = run_llm_request(text, INSTRUCTIONS, Summary)
    return SummarizeResponse(
        summary=result.parsed,
        tokens_used=result.tokens_used,
        response_time_seconds=result.response_time_seconds,
        ttft_seconds=result.ttft_seconds,
    )
