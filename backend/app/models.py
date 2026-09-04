from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    title: str | None = None
    url: str | None = None


class Answer(BaseModel):
    text: str
    sources: list[Source]


class AskResponse(BaseModel):
    answer: Answer
    tokens_used: int
    response_time_seconds: float
    ttft_seconds: float | None = None
