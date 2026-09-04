from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    title: str | None = None
    url: str | None = None


class Answer(BaseModel):
    text: str = Field(min_length = 1)
    sources: list[Source]
    confidence: float = Field(ge=0.0, le=1.0)


class AskResponse(BaseModel):
    answer: Answer
    tokens_used: int
    response_time_seconds: float
    ttft_seconds: float | None = None
