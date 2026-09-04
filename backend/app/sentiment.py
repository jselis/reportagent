from app.llm import run_llm_request
from app.models import Sentiment, SentimentResponse

INSTRUCTIONS = (
    "Classify the overall sentiment of the user's text as positive, negative, "
    "or neutral, and provide your confidence in that classification as a number "
    "between 0 and 1."
)


def analyze_sentiment(text: str) -> SentimentResponse:
    """Analyze the sentiment of the given text."""
    result = run_llm_request(text, INSTRUCTIONS, Sentiment)
    return SentimentResponse(
        sentiment=result.parsed,
        tokens_used=result.tokens_used,
        response_time_seconds=result.response_time_seconds,
        ttft_seconds=result.ttft_seconds,
    )
