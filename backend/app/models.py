from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    title: str | None = None
    url: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
