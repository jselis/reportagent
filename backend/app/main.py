from typing import Callable, TypeVar

from fastapi import FastAPI, HTTPException

from app import llm
from app.config import settings
from app.models import (
    AnalyzeSentimentRequest,
    AskRequest,
    AskResponse,
    ResearchAPIError,
    ResearchConnectionError,
    ResearchTimeoutError,
    SentimentResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from app.research import research_and_answer
from app.sentiment import analyze_sentiment
from app.summarize import summarize_text

app = FastAPI(
    title="reportagent",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

T = TypeVar("T")


def _run_or_raise_http(fn: Callable[..., T], *args) -> T:
    """Call an LLM-backed endpoint function, translating its typed errors to HTTP."""
    try:
        return fn(*args)
    except ResearchTimeoutError:
        raise HTTPException(
            status_code=504,
            detail="The research service took too long to respond. Please try again.",
        )
    except ResearchConnectionError:
        raise HTTPException(
            status_code=503,
            detail="The research service is currently unreachable. Please try again shortly.",
        )
    except ResearchAPIError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=f'The research service gave a server error {exc.status_code}: "{exc.message}". Please try again when this problem is solved.',
        )


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    return _run_or_raise_http(research_and_answer, request.question)


@app.post("/api/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest) -> SummarizeResponse:
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    return _run_or_raise_http(summarize_text, request.text)


@app.post("/api/analyze-sentiment", response_model=SentimentResponse)
def analyze_sentiment_endpoint(request: AnalyzeSentimentRequest) -> SentimentResponse:
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    return _run_or_raise_http(analyze_sentiment, request.text)


if settings.debug:

    @app.post("/api/debug/force-timeout")
    def set_force_timeout(enabled: bool = True) -> dict:
        llm._force_timeout = enabled
        return {"force_timeout": enabled}

    @app.post("/api/debug/force-connection-error")
    def set_force_connection_error(enabled: bool = True) -> dict:
        llm._force_connection_error = enabled
        return {"force_connection_error": enabled}

    @app.post("/api/debug/force-api-error")
    def set_force_api_error(
        enabled: bool = True,
        status_code: int = 500,
        message: str = "Simulated OpenAI error",
    ) -> dict:
        llm._force_api_error = (
            {"status_code": status_code, "message": message} if enabled else None
        )
        return {"force_api_error": llm._force_api_error}
