from fastapi import FastAPI, HTTPException

from app import research
from app.config import settings
from app.models import (
    AskRequest,
    AskResponse,
    ResearchAPIError,
    ResearchConnectionError,
    ResearchTimeoutError,
)
from app.research import research_and_answer

app = FastAPI(
    title="reportagent",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    try:
        return research_and_answer(request.question)
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
        raise HTTPException(status_code=exc.status_code, detail=exc.message)


if settings.debug:

    @app.post("/api/debug/force-timeout")
    def set_force_timeout(enabled: bool = True) -> dict:
        research._force_timeout = enabled
        return {"force_timeout": enabled}

    @app.post("/api/debug/force-connection-error")
    def set_force_connection_error(enabled: bool = True) -> dict:
        research._force_connection_error = enabled
        return {"force_connection_error": enabled}

    @app.post("/api/debug/force-api-error")
    def set_force_api_error(
        enabled: bool = True,
        status_code: int = 500,
        message: str = "Simulated OpenAI error",
    ) -> dict:
        research._force_api_error = (
            {"status_code": status_code, "message": message} if enabled else None
        )
        return {"force_api_error": research._force_api_error}
