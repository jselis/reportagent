from fastapi import FastAPI, HTTPException

from app.models import AskRequest, AskResponse
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

    return research_and_answer(request.question)
