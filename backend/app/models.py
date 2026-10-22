from dataclasses import dataclass
from datetime import datetime
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
    topic: str


class ExtractTextResponse(BaseModel):
    text: str


class SyncMetadataResponse(BaseModel):
    synced: list[str]
    failed: list[str]


class CorpusDocument(BaseModel):
    document_id: str
    topic: str
    created_at: datetime
    synced_at: datetime | None
    sync_needed: bool


class CorpusResponse(BaseModel):
    documents: list[CorpusDocument]


class FaithfulnessClaim(BaseModel):
    claim: str
    supported: bool


class FaithfulnessJudgement(BaseModel):
    """The LLM judge's output: the generated answer broken into atomic claims,
    each marked as supported or not by the retrieved chunks. The faithfulness
    score itself (supported / total) is computed server-side, not by the model."""

    claims: list[FaithfulnessClaim]


class CorrectnessJudgement(BaseModel):
    score: Literal[0.0, 0.5, 1.0]


class JobStartResponse(BaseModel):
    started: bool


class JobStatusResponse(BaseModel):
    status: Literal["idle", "running", "done"]
    total: int
    done: int
    failed: int


class EvaluationSummaryResponse(BaseModel):
    case_count: int
    avg_faithfulness: float | None
    avg_correctness: float | None
    avg_retrieval_recall: float | None
    avg_retrieval_precision: float | None
    avg_retrieval_f1: float | None


class EvalRunResponse(BaseModel):
    """One evaluation run with its aggregates over the currently active ground truth cases."""

    id: int
    created_at: datetime
    label: str | None
    pipeline: Literal["rag", "agent"]
    config: dict
    status: Literal["generating", "generated", "scoring", "scored", "failed"]
    judge_model: str | None
    scored_at: datetime | None
    case_count: int
    failed_count: int
    missing_case_count: int  # active cases that have no result in this run yet
    avg_faithfulness: float | None
    avg_correctness: float | None
    avg_retrieval_recall: float | None
    avg_retrieval_precision: float | None
    avg_retrieval_f1: float | None
    avg_tokens_used: float | None
    avg_response_time_seconds: float | None
    avg_searches: float | None


class EvalRunsResponse(BaseModel):
    runs: list[EvalRunResponse]


class ExpectedChunk(BaseModel):
    chunk_id: str
    chunk: str


class CreateGroundTruthRequest(BaseModel):
    question: str
    expected_answer: str
    expected_chunks: list[ExpectedChunk]


class GroundTruthCaseResponse(BaseModel):
    id: int
    question: str
    expected_answer: str
    expected_chunks: list[ExpectedChunk]


class RetrievedChunk(BaseModel):
    id: str  # the chunk's id in the Pinecone index (document_id-chunk_id)
    document_id: str
    chunk_id: int
    text: str
    score: float
    topic: str | None = None  # None for chunks ingested before this field existed


class IngestResponse(BaseModel):
    document_id: str
    chunks_ingested: int


class LLMResponseMeta(BaseModel):
    tokens_used: int
    response_time_seconds: float
    ttft_seconds: float | None = None
    raw_llm_output: str | None = None  # debug only: the model's output exactly as returned


class AskResponse(LLMResponseMeta):
    answer: Answer


class AgentStep(BaseModel):
    """One tool call the agent made, for the inspection view."""

    tool: str
    arguments: dict


class AgentAskResponse(AskResponse):
    steps: list[AgentStep]


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
    raw_output: str | None = None


class ResearchError(Exception):
    """Base class for LLM-request failures."""


class ResearchTimeoutError(ResearchError):
    """Raised when the OpenAI call doesn't complete within the allowed time."""


class ResearchConnectionError(ResearchError):
    """Raised when the OpenAI call fails to even establish a connection."""


class AgentDidNotAnswerError(ResearchError):
    """Raised when the agent run ends without a valid submit_answer call."""


class ResearchAPIError(ResearchError):
    """Raised when OpenAI responds with an HTTP error status (4xx/5xx)."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"OpenAI returned {status_code}: {message}")
