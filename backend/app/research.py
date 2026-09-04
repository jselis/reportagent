import time

from openai import OpenAI

from app.config import settings
from app.models import Answer, AskResponse

MODEL = "gpt-5.6-terra"

client = OpenAI(api_key=settings.openai_api_key)


def research_and_answer(question: str) -> AskResponse:
    """Ask the model to research the question via web search and answer it."""
    start = time.perf_counter()
    ttft = None

    with client.responses.stream(
        model=MODEL,
        #tools=[{"type": "web_search_preview"}],
        input=question,
        text_format=Answer,
    ) as stream:
        for event in stream:
            if ttft is None and event.type == "response.output_text.delta":
                ttft = time.perf_counter() - start
        response = stream.get_final_response()

    elapsed = time.perf_counter() - start

    return AskResponse(
        answer=response.output_parsed,
        tokens_used=response.usage.total_tokens,
        response_time_seconds=elapsed,
        ttft_seconds=ttft,
    )
