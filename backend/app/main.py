from functools import partial
from typing import Callable, TypeVar

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app import llm
from app.agent import run_agent
from app.chat_history import delete_agent_session, get_agent_session, list_agent_sessions
from app.config import settings
from app.documents import extract_pdf_text
from app.evaluation import create_eval_run, list_eval_runs, require_eval_run, run_generation, run_scoring
from app.ground_truth import create_ground_truth_case, list_ground_truth_cases, supersede_ground_truth_case
from app.jobs import Job
from app.metadata import list_corpus, sync_pending_metadata, upsert_document_metadata
from app.models import (
    ActiveGroundTruthCaseNotFoundError,
    EvalRunNotFoundError,
    GroundTruthCaseListResponse,
    StartEvalRunRequest,
    StartEvalRunResponse,
    AgentAskRequest,
    AgentAskResponse,
    AgentSessionListResponse,
    AgentSessionNotFoundError,
    AgentSessionResponse,
    AgentDidNotAnswerError,
    CorpusResponse,
    AnalyzeSentimentRequest,
    AskRequest,
    AskResponse,
    CreateGroundTruthRequest,
    EvalRunsResponse,
    ExtractTextResponse,
    GroundTruthCaseResponse,
    IngestRequest,
    IngestResponse,
    JobStartResponse,
    JobStatusResponse,
    ResearchAPIError,
    ResearchConnectionError,
    ResearchTimeoutError,
    RetrievedChunk,
    SentimentResponse,
    SummarizeRequest,
    SummarizeResponse,
    SyncMetadataResponse,
)
from app.rag import ingest_document, retrieve
from app.research import research_and_answer
from app.sentiment import analyze_sentiment
from app.summarize import summarize_text

app = FastAPI(
    title="reportagent",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
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
    except AgentDidNotAnswerError:
        raise HTTPException(
            status_code=502,
            detail="The agent did not produce a valid answer, possibly because it ran out of search steps. Try rephrasing the question.",
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


@app.post("/api/agent/ask", response_model=AgentAskResponse)
def agent_ask(request: AgentAskRequest) -> AgentAskResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    try:
        return _run_or_raise_http(run_agent, request.question, request.session_id)
    except AgentSessionNotFoundError:
        raise HTTPException(status_code=404, detail="This chat no longer exists. Start a new chat.")


@app.get("/api/agent/sessions", response_model=AgentSessionListResponse)
def agent_sessions() -> AgentSessionListResponse:
    return list_agent_sessions()


@app.get("/api/agent/sessions/{session_id}", response_model=AgentSessionResponse)
def agent_session(session_id: str) -> AgentSessionResponse:
    try:
        return get_agent_session(session_id)
    except AgentSessionNotFoundError:
        raise HTTPException(status_code=404, detail="That chat doesn't exist.")


@app.delete("/api/agent/sessions/{session_id}", status_code=204)
def delete_session(session_id: str) -> None:
    try:
        delete_agent_session(session_id)
    except AgentSessionNotFoundError:
        raise HTTPException(status_code=404, detail="That chat doesn't exist.")


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


@app.post("/api/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    if not request.document_id.strip():
        raise HTTPException(status_code=400, detail="document_id must not be empty")
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    chunks_ingested = ingest_document(request)
    upsert_document_metadata(request.document_id, request.topic)
    return IngestResponse(document_id=request.document_id, chunks_ingested=chunks_ingested)


@app.post("/api/sync-metadata", response_model=SyncMetadataResponse)
def sync_metadata() -> SyncMetadataResponse:
    synced, failed = sync_pending_metadata()
    return SyncMetadataResponse(synced=synced, failed=failed)


@app.get("/api/corpus", response_model=CorpusResponse)
def corpus() -> CorpusResponse:
    return CorpusResponse(documents=list_corpus())


@app.get("/api/retrieve", response_model=list[RetrievedChunk])
def retrieve_chunks(q: str, top_n: int = 20, topic: str | None = None) -> list[RetrievedChunk]:
    if not q.strip():
        raise HTTPException(status_code=400, detail="q must not be empty")

    return retrieve(q, top_k=top_n, topic=topic)


@app.post("/api/ground-truth", response_model=GroundTruthCaseResponse)
def create_ground_truth(request: CreateGroundTruthRequest) -> GroundTruthCaseResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    if not request.expected_answer.strip():
        raise HTTPException(status_code=400, detail="expected_answer must not be empty")
    if not request.expected_chunks:
        raise HTTPException(status_code=400, detail="select at least one chunk")

    try:
        return create_ground_truth_case(request)
    except ActiveGroundTruthCaseNotFoundError:
        raise HTTPException(status_code=404, detail="The case to replace doesn't exist or is already outdated.")


@app.get("/api/ground-truth", response_model=GroundTruthCaseListResponse)
def ground_truth_cases() -> GroundTruthCaseListResponse:
    return GroundTruthCaseListResponse(cases=list_ground_truth_cases())


@app.post("/api/ground-truth/{case_id}/supersede", status_code=204)
def supersede_ground_truth(case_id: int) -> None:
    try:
        supersede_ground_truth_case(case_id)
    except ActiveGroundTruthCaseNotFoundError:
        raise HTTPException(status_code=404, detail="That case doesn't exist or is already outdated.")


_generation_job = Job()
_scoring_job = Job()


def _require_run_or_404(eval_run_id: int) -> None:
    try:
        require_eval_run(eval_run_id)
    except EvalRunNotFoundError:
        raise HTTPException(status_code=404, detail=f"Evaluation run {eval_run_id} doesn't exist.")


@app.post("/api/evaluation/runs", response_model=StartEvalRunResponse)
def start_eval_run(request: StartEvalRunRequest) -> StartEvalRunResponse:
    """Create a run and start generating its answers in the background."""
    if _generation_job.snapshot()["status"] == "running":
        raise HTTPException(status_code=409, detail="A generation job is already running.")

    eval_run_id = create_eval_run(request)
    _generation_job.start(partial(run_generation, eval_run_id=eval_run_id))
    return StartEvalRunResponse(eval_run_id=eval_run_id)


@app.get("/api/evaluation/runs", response_model=EvalRunsResponse)
def evaluation_runs() -> EvalRunsResponse:
    return EvalRunsResponse(runs=list_eval_runs())


@app.post("/api/evaluation/runs/{eval_run_id}/generate", response_model=JobStartResponse)
def update_eval_run(eval_run_id: int) -> JobStartResponse:
    """Generate answers for the active cases the run doesn't have a successful result for yet."""
    _require_run_or_404(eval_run_id)
    return JobStartResponse(started=_generation_job.start(partial(run_generation, eval_run_id=eval_run_id)))


@app.post("/api/evaluation/runs/{eval_run_id}/score", response_model=JobStartResponse)
def score_eval_run(eval_run_id: int, rescore_all: bool) -> JobStartResponse:
    """Score a run's answers: only the unscored ones, or all of them with rescore_all."""
    _require_run_or_404(eval_run_id)
    return JobStartResponse(
        started=_scoring_job.start(partial(run_scoring, eval_run_id=eval_run_id, rescore_all=rescore_all))
    )


@app.get("/api/evaluation/generate/status", response_model=JobStatusResponse)
def generation_status() -> JobStatusResponse:
    return JobStatusResponse(**_generation_job.snapshot())


@app.get("/api/evaluation/score/status", response_model=JobStatusResponse)
def scoring_status() -> JobStatusResponse:
    return JobStatusResponse(**_scoring_job.snapshot())


@app.post("/api/extract-text", response_model=ExtractTextResponse)
async def extract_text(file: UploadFile = File(...)) -> ExtractTextResponse:
    if file.content_type != "application/pdf" and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_bytes = await file.read()
    try:
        text = extract_pdf_text(file_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read this PDF file")

    return ExtractTextResponse(text=text)


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
