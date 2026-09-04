import time

import openai
from openai import OpenAI

from app.config import settings
from app.models import Answer, AskResponse, ResearchConnectionError, ResearchTimeoutError

MODEL = "gpt-5.6-terra"
TIMEOUT_SECONDS = 15.0

client = OpenAI(api_key=settings.openai_api_key)

# Debug-only fault injection flags, toggled via /api/debug/* endpoints (see main.py).
# Only reachable when settings.debug is True.
_force_timeout = False
_force_connection_error = False


def research_and_answer(question: str) -> AskResponse:
    """Ask the model to research the question via web search and answer it."""
    if _force_timeout:
        raise ResearchTimeoutError("Timeout forced for testing")
    if _force_connection_error:
        raise ResearchConnectionError("Connection error forced for testing")

    start = time.perf_counter()
    ttft = None

    try:
        with client.responses.stream(
            model=MODEL,
            #tools=[{"type": "web_search_preview"}],
            input=question,
            text_format=Answer,
            timeout=TIMEOUT_SECONDS,
        ) as stream:
            for event in stream:
                if ttft is None and event.type == "response.output_text.delta":
                    ttft = time.perf_counter() - start
            response = stream.get_final_response()
    except openai.APITimeoutError as exc:
        raise ResearchTimeoutError(
            f"OpenAI request did not complete within {TIMEOUT_SECONDS}s"
        ) from exc
    except openai.APIConnectionError as exc:
        raise ResearchConnectionError(
            "Failed to establish a connection to OpenAI"
        ) from exc

    elapsed = time.perf_counter() - start

    return AskResponse(
        answer=response.output_parsed,
        tokens_used=response.usage.total_tokens,
        response_time_seconds=elapsed,
        ttft_seconds=ttft,
    )
