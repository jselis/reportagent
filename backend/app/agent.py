import asyncio
import time
from dataclasses import dataclass, field

import openai
from google.adk.agents import Agent
from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.run_config import RunConfig
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from google.genai import types
from pydantic import ValidationError

from app.config import settings
from app.models import (
    AgentAskResponse,
    AgentDidNotAnswerError,
    AgentStep,
    AskResponse,
    Citation,
    GroundedAnswer,
    ResearchAPIError,
    ResearchConnectionError,
    ResearchTimeoutError,
    RetrievedChunk,
)
from app.rag import retrieve
from app.research import build_answer

MODEL = "openai/gpt-5.6-terra"  # LiteLLM model string: provider prefix + OpenAI model name
LLM_CALL_TIMEOUT_SECONDS = 30.0
# Settings of the Agent tab / /api/agent/ask; evaluation runs pass their own.
MAX_LLM_CALLS = 10  # upper bound on model calls per question: searches + the final answer
TOP_K = 5  # chunks returned per search
APP_NAME = "reportagent"
USER_ID = "user"

INSTRUCTION = """You answer questions using ONLY document chunks that you retrieve with the search_documents tool. Each returned chunk has an id, like some-document-3.

Process:
- Call search_documents to find chunks. If the results are thin or miss part of the question, search again with a different query. You may split a complex question into several searches.
- You have at most {max_model_calls} model calls in total. Every search and the final submit_answer each use one, so the last call must be submit_answer. Do not keep searching until the budget runs out: after a few attempts, decide whether the answer can be found reliably in the documents. If it cannot, call submit_answer and say that it is not in the relevant matches in the documents.
- When you have enough, call submit_answer exactly once. That ends the run. Never reply with plain text: your answer must go through submit_answer.

Rules for the answer:
- Base your answer strictly on the retrieved chunks. Do not use outside knowledge, and do not guess.
- If the chunks do not contain the answer, say clearly in `text` that the answer is not in the relevant matches in the documents. In that case pass no citations and put no reference markers in the text. Use a low confidence.
- Otherwise, answer directly and concisely, in the language of the question.
- Right after each statement that comes from a chunk, put that chunk's exact id in square brackets, copied from the tool result, e.g. "The tower is 330 meters tall. [some-document-3]". If a statement is supported by several chunks, use one bracket per id, e.g. [doc-a-0][doc-b-2]. Never invent ids and never use plain numbers as markers.
- In `citations`, list each distinct chunk you used, once: `source_id` is the chunk's id copied exactly, and `quote` is the shortest verbatim excerpt (one or two sentences, copied exactly) from that chunk that supports your statement.
- Set `confidence` to how well the chunks support your answer."""


@dataclass
class _RunState:
    """What one agent run collects on the side, outside the model's context."""

    chunks: dict[str, RetrievedChunk] = field(default_factory=dict)  # by chunk id, in retrieval order
    steps: list[AgentStep] = field(default_factory=list)
    answer: GroundedAnswer | None = None


def _build_agent(state: _RunState, max_llm_calls: int, search_top_k: int) -> Agent:
    """Build the agent with tools bound to this run's state."""

    def search_documents(query: str) -> dict:
        """Search the ingested documents and return the best-matching chunks.

        The search is semantic (embedding-based), not keyword-based: write a natural-language
        query that describes the meaning of what you look for. Quotes, boolean operators and
        exact-phrase syntax have no effect.

        Args:
            query: A natural-language description of what to find.
        """
        state.steps.append(AgentStep(tool="search_documents", arguments={"query": query}))
        found = retrieve(query, top_k=search_top_k)
        for chunk in found:
            state.chunks.setdefault(chunk.id, chunk)
        return {"chunks": [{"id": c.id, "text": c.text} for c in found]}

    def submit_answer(
        text: str, confidence: float, citations: list[Citation], tool_context: ToolContext
    ) -> dict:
        """Submit the final answer. Call this exactly once, as the last step.

        Args:
            text: The answer, with a chunk id in square brackets after each statement.
            confidence: A number from 0 to 1: how well the chunks support the answer.
            citations: One entry per distinct chunk used: its id and a verbatim supporting quote.
        """
        try:
            state.answer = GroundedAnswer(text=text, confidence=confidence, citations=citations)
        except ValidationError as exc:
            return {"error": f"Invalid answer, fix it and call submit_answer again: {exc}"}
        tool_context.actions.skip_summarization = True  # the run ends here, no further model call
        return {"status": "submitted"}

    return Agent(
        name="report_agent",
        model=LiteLlm(
            model=MODEL, api_key=settings.openai_api_key, timeout=LLM_CALL_TIMEOUT_SECONDS
        ),
        instruction=INSTRUCTION.format(max_model_calls=max_llm_calls),
        tools=[search_documents, submit_answer],
    )


async def _run(question: str, state: _RunState, max_llm_calls: int, search_top_k: int) -> int:
    """Run the agent to completion and return the total tokens used."""
    service = InMemorySessionService()
    runner = Runner(agent=_build_agent(state, max_llm_calls, search_top_k), app_name=APP_NAME, session_service=service)
    session = await service.create_session(app_name=APP_NAME, user_id=USER_ID)
    message = types.Content(role="user", parts=[types.Part(text=question)])

    tokens_used = 0
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session.id,
        new_message=message,
        run_config=RunConfig(max_llm_calls=max_llm_calls),
    ):
        if event.usage_metadata and event.usage_metadata.total_token_count:
            tokens_used += event.usage_metadata.total_token_count
    return tokens_used


@dataclass
class AgentRun:
    """An agent response plus what the run saw and chose, for evaluation."""

    response: AgentAskResponse
    fed_chunks: list[RetrievedChunk]  # every chunk returned by any search: all that reached the model
    selected_chunks: list[RetrievedChunk]  # the chunks the final answer cites


def run_agent_with_chunks(question: str, max_llm_calls: int, search_top_k: int) -> AgentRun:
    """Answer a question with an agent that searches the documents as often as it needs."""
    state = _RunState()
    start = time.perf_counter()

    try:
        tokens_used = asyncio.run(_run(question, state, max_llm_calls, search_top_k))
    except LlmCallsLimitExceededError as exc:
        raise AgentDidNotAnswerError(f"The agent used all {max_llm_calls} allowed LLM calls") from exc
    except openai.APITimeoutError as exc:
        raise ResearchTimeoutError(
            f"An LLM call did not complete within {LLM_CALL_TIMEOUT_SECONDS}s"
        ) from exc
    except openai.APIConnectionError as exc:
        raise ResearchConnectionError("Failed to establish a connection to OpenAI") from exc
    except openai.APIStatusError as exc:
        raise ResearchAPIError(status_code=exc.status_code, message=exc.message) from exc

    elapsed = time.perf_counter() - start

    if state.answer is None:
        raise AgentDidNotAnswerError("The agent run ended without a valid submit_answer call")

    fed_chunks = list(state.chunks.values())
    answer = build_answer(state.answer, fed_chunks)
    cited_ids = {source.id for source in answer.sources if source.type == "document"}

    base = AskResponse(answer=answer, tokens_used=tokens_used, response_time_seconds=elapsed)
    return AgentRun(
        response=AgentAskResponse(**base.model_dump(), steps=state.steps),
        fed_chunks=fed_chunks,
        selected_chunks=[c for c in fed_chunks if c.id in cited_ids],
    )


def run_agent(question: str) -> AgentAskResponse:
    """The Agent tab's entry point: the agent with this module's default settings."""
    return run_agent_with_chunks(question, MAX_LLM_CALLS, TOP_K).response
