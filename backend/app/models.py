from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str


class SummarizeRequest(BaseModel):
    text: str


class AnalyzeSentimentRequest(BaseModel):
    text: str


class DocumentSource(BaseModel):
    type: Literal["document"] = "document"
    id: str  # globally unique: the chunk's id in the Pinecone index (document_id-chunk_id)
    document_id: str
    chunk_id: int
    snippet: str


class WebSource(BaseModel):
    type: Literal["web"] = "web"
    title: str | None = None
    url: str
    snippet: str | None = None


Source = Annotated[DocumentSource | WebSource, Field(discriminator="type")]


class Answer(BaseModel):
    text: str = Field(min_length=1)
    sources: list[Source]
    confidence: float = Field(ge=0.0, le=1.0)


class Citation(BaseModel):
    """What the LLM returns per used chunk: the chunk's index id and a verbatim quote."""

    source_id: str
    quote: str


class GroundedAnswer(BaseModel):
    """The LLM-facing output schema. The server turns citations into real Source objects."""

    text: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[Citation]


class Summary(BaseModel):
    text: str = Field(min_length=1)


class Sentiment(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)


class IngestRequest(BaseModel):
    document_id: str
    text: str


class RetrievedChunk(BaseModel):
    id: str  # the chunk's id in the Pinecone index (document_id-chunk_id)
    document_id: str
    chunk_id: int
    text: str
    score: float


class IngestResponse(BaseModel):
    document_id: str
    chunks_ingested: int


class LLMResponseMeta(BaseModel):
    tokens_used: int
    response_time_seconds: float
    ttft_seconds: float | None = None


class AskResponse(LLMResponseMeta):
    answer: Answer


class SummarizeResponse(LLMResponseMeta):
    summary: Summary


class SentimentResponse(LLMResponseMeta):
    sentiment: Sentiment


@dataclass
class LLMResult:
    """Internal transport between the generic LLM engine and its per-endpoint callers."""

    parsed: BaseModel
    tokens_used: int
    response_time_seconds: float
    ttft_seconds: float | None


class ResearchError(Exception):
    """Base class for LLM-request failures."""


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
