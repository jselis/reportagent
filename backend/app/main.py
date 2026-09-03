from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.research import research_and_answer

app = FastAPI(
    title="reportagent",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    title: str | None = None
    url: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    result = research_and_answer(request.question)
    return AskResponse(**result)
