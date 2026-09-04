import time

import openai
from openai import OpenAI
from pydantic import BaseModel

from app.config import settings
from app.models import (
    LLMResult,
    ResearchAPIError,
    ResearchConnectionError,
    ResearchTimeoutError,
)

MODEL = "gpt-5.6-terra"
TIMEOUT_SECONDS = 15.0

client = OpenAI(api_key=settings.openai_api_key)

# Debug-only fault injection flags, toggled via /api/debug/* endpoints (see main.py).
# Only reachable when settings.debug is True.
_force_timeout = False
_force_connection_error = False
_force_api_error: dict | None = None  # {"status_code": int, "message": str}


def run_llm_request(
    input_text: str,
    instructions: str,
    text_format: type[BaseModel],
) -> LLMResult:
    """Send a structured-output request to the LLM.

    Shared by every endpoint that talks to OpenAI: handles timeouts, connection
    failures, and API errors uniformly, and measures time-to-first-token via streaming.
    """
    if _force_timeout:
        raise ResearchTimeoutError("Timeout forced for testing")
    if _force_connection_error:
        raise ResearchConnectionError("Connection error forced for testing")
    if _force_api_error is not None:
        raise ResearchAPIError(**_force_api_error)

    start = time.perf_counter()
    ttft = None

    try:
        with client.responses.stream(
            model=MODEL,
            instructions=instructions,
            input=input_text,
            text_format=text_format,
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
    except openai.APIStatusError as exc:
        raise ResearchAPIError(status_code=exc.status_code, message=exc.message) from exc

    elapsed = time.perf_counter() - start

    return LLMResult(
        parsed=response.output_parsed,
        tokens_used=response.usage.total_tokens,
        response_time_seconds=elapsed,
        ttft_seconds=ttft,
    )
