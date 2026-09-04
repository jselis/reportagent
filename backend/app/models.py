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


class ResearchError(Exception):
    """Base class for research-layer failures."""


class ResearchTimeoutError(ResearchError):
    """Raised when the OpenAI call doesn't complete within the allowed time."""


class ResearchConnectionError(ResearchError):
    """Raised when the OpenAI call fails to even establish a connection."""


class ResearchAPIError(ResearchError):
    """Raised when OpenAI responds with an HTTP error status (4xx/5xx)."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"OpenAI returned {status_code}: {message}")
