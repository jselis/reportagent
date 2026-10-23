import asyncio
import time
from dataclasses import dataclass, field
from typing import get_args

import openai

from app import tracing  # sets up tracing; must be imported before google.adk is used
from google.adk.agents import Agent
from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.run_config import RunConfig
from google.adk.events import Event, EventActions
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.tools import ToolContext
from google.genai import types
from pydantic import ValidationError

from app.config import settings
from app.models import (
    AgentAskResponse,
    AgentDidNotAnswerError,
    AgentSessionNotFoundError,
    AgentStep,
    Answer,
    AskResponse,
    Citation,
    DocumentSource,
    GroundedAnswer,
    ResearchAPIError,
    ResearchConnectionError,
    ResearchTimeoutError,
    RetrievedChunk,
    Task,
    TaskType,
)
from app.rag import retrieve
from app.research import build_answer
from app.sessions import PostgresSessionService
from app.tasks import create_task, get_task, list_tasks

MODEL = "openai/gpt-5.6-terra"  # LiteLLM model string: provider prefix + OpenAI model name
LLM_CALL_TIMEOUT_SECONDS = 30.0
# Settings of the Agent tab / /api/agent/ask; evaluation runs pass their own.
MAX_LLM_CALLS = 10  # upper bound on model calls per question (per turn): searches + the final answer
TOP_K = 5  # chunks returned per search
APP_NAME = "reportagent"
USER_ID = "user"

# Session state keys. Both are saved with the chat session, so they cover the whole conversation.
SEEN_CHUNKS_KEY = "seen_chunks"  # every chunk the model was shown: id -> chunk
SOURCES_KEY = "sources"  # cumulative source list; a source's number is its position + 1
ACTIVE_TASK_KEY = "active_task"  # the task this chat operates under, if any
TURNS_KEY = "turns"  # each turn's question and full response, so a chat can be reopened as it was

# Chats of the Agent tab live in Postgres; evaluation runs use a fresh in-memory session each.
_chat_sessions = PostgresSessionService()

INSTRUCTION = """You answer questions using ONLY document chunks that you retrieve with the search_documents tool. Each returned chunk has an id, like some-document-3.

The conversation may contain earlier questions and answers. Answer the latest question, using the earlier turns to understand what it refers to. Chunks you retrieved in earlier turns are still in the conversation: you may cite them by their id if they are enough, or search again.

Process:
- Call search_documents to find chunks. If the results are thin or miss part of the question, search again with a different query. You may split a complex question into several searches.
- For each question you have at most {max_model_calls} model calls in total. Every search and the final submit_answer each use one, so the last call must be submit_answer. Do not keep searching until the budget runs out: after a few attempts, decide whether the answer can be found reliably in the documents. If it cannot, call submit_answer and say that it is not in the relevant matches in the documents.
- When you have enough, call submit_answer exactly once. That ends the run. Never reply with plain text: your answer must go through submit_answer.

Rules for the answer:
- Base your answer strictly on the retrieved chunks. Do not use outside knowledge, and do not guess.
- If the chunks do not contain the answer, say clearly in `text` that the answer is not in the relevant matches in the documents. In that case pass no citations and put no reference markers in the text. Use a low confidence.
- Otherwise, answer directly and concisely, in the language of the question.
- Right after each statement that comes from a chunk, put that chunk's exact id in square brackets, copied from the tool result, e.g. "The tower is 330 meters tall. [some-document-3]". If a statement is supported by several chunks, use one bracket per id, e.g. [doc-a-0][doc-b-2]. Always write the chunk id, never a number: the numbers you may see in the interface are not ids. Never invent ids.
- In `citations`, list each distinct chunk you used, once: `source_id` is the chunk's id copied exactly, and `quote` is a short verbatim excerpt (copied exactly) from that chunk that supports your statement. Keep only the passages that matter: you may skip irrelevant stretches with "...", e.g. "The tower was designed by Gustave Eiffel ... and is 330 meters tall". Every part between the "..." must be copied exactly, in the original order.
- Set `confidence` to how well the chunks support your answer."""

TASK_INSTRUCTION = """

Tasks:
Besides answering questions, you work on tasks for the user. The only task type for now is `report_creation`: the user is working on a report. A task is the context of this chat; the active task, if any, is stated at the end of these instructions.
- If the user says they want to start or create a new report, call create_task with task_type `report_creation` and a short title taken from what the user said. Do not create a task when the user only asks a question.
- If the user refers to a report that may already exist (for example to continue or reopen it), call list_tasks, then open_task with the id of the matching task. If no task matches, tell the user that and ask whether to create a new one; do not create one unprompted.
- Never create a task that already exists, and do not create or open a task that is already the active one.
- After creating or opening a task, call submit_answer to confirm it to the user in one sentence, with no citations, no reference markers and a high confidence. Do not search the documents for this."""


@dataclass
class _RunState:
    """What one agent run collects on the side, outside the model's context."""

    session_id: str = ""
    chunks: dict[str, RetrievedChunk] = field(default_factory=dict)  # every chunk seen in the session, in order
    steps: list[AgentStep] = field(default_factory=list)  # this run's tool calls only
    task: Task | None = None  # the chat's active task; the model may change it during the run
    answer: Answer | None = None
    cited_ids: list[str] = field(default_factory=list)  # chunks cited in this run's answer


def _build_agent(state: _RunState, max_llm_calls: int, search_top_k: int, use_tasks: bool) -> Agent:
    """Build the agent with tools bound to this run's state."""

    def search_documents(query: str, tool_context: ToolContext) -> dict:
        """Search the ingested documents and return the best-matching chunks.

        The search is semantic (embedding-based), not keyword-based: write a natural-language
        query that describes the meaning of what you look for. Quotes, boolean operators and
        exact-phrase syntax have no effect.

        Args:
            query: A natural-language description of what to find.
        """
        found = retrieve(query, top_k=search_top_k)
        state.steps.append(AgentStep(tool="search_documents", arguments={"query": query}, chunks=found))
        for chunk in found:
            state.chunks.setdefault(chunk.id, chunk)
        # Save everything the model has been shown in this session: only these ids are valid citations.
        tool_context.state[SEEN_CHUNKS_KEY] = {cid: c.model_dump() for cid, c in state.chunks.items()}
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
            grounded = GroundedAnswer(text=text, confidence=confidence, citations=citations)
        except ValidationError as exc:
            return {"error": f"Invalid answer, fix it and call submit_answer again: {exc}"}

        previous_sources = [DocumentSource(**s) for s in tool_context.state.get(SOURCES_KEY, [])]
        state.answer, state.cited_ids = build_answer(grounded, list(state.chunks.values()), previous_sources)
        tool_context.state[SOURCES_KEY] = [s.model_dump() for s in state.answer.sources]
        tool_context.actions.skip_summarization = True  # the run ends here, no further model call
        return {"status": "submitted"}

    def _activate(task: Task, tool_context: ToolContext) -> dict:
        state.task = task
        tool_context.state[ACTIVE_TASK_KEY] = task.model_dump(mode="json")
        return {"status": "active", "task": task.model_dump(mode="json")}

    def create_task_tool(task_type: str, title: str, tool_context: ToolContext) -> dict:
        """Create a new task and make it the active one of this chat.

        Args:
            task_type: The kind of task. Only "report_creation" exists.
            title: A short title for the task, taken from what the user said.
        """
        state.steps.append(
            AgentStep(tool="create_task", arguments={"task_type": task_type, "title": title}, chunks=[])
        )
        if task_type not in get_args(TaskType):
            return {"error": f"Unknown task_type {task_type!r}; use one of {list(get_args(TaskType))}"}
        return _activate(create_task(task_type, title), tool_context)

    def list_tasks_tool() -> dict:
        """List all existing tasks, newest first, to find the one the user refers to."""
        state.steps.append(AgentStep(tool="list_tasks", arguments={}, chunks=[]))
        return {"tasks": [t.model_dump(mode="json") for t in list_tasks()]}

    def open_task_tool(task_id: int, tool_context: ToolContext) -> dict:
        """Make an existing task the active one of this chat.

        Args:
            task_id: The id of a task returned by list_tasks.
        """
        state.steps.append(AgentStep(tool="open_task", arguments={"task_id": task_id}, chunks=[]))
        task = get_task(task_id)
        if task is None:
            return {"error": f"There is no task with id {task_id}"}
        return _activate(task, tool_context)

    # The model sees the tools under these names; the _tool suffix only avoids clashing with
    # the imported database functions.
    create_task_tool.__name__ = "create_task"
    list_tasks_tool.__name__ = "list_tasks"
    open_task_tool.__name__ = "open_task"

    def instruction(_: ReadonlyContext) -> str:
        text = INSTRUCTION.format(max_model_calls=max_llm_calls)
        if not use_tasks:
            return text
        if state.task is None:
            active = "No task is active."
        else:
            active = f"Active task: #{state.task.id}, {state.task.type}, \"{state.task.title}\"."
        return text + TASK_INSTRUCTION + "\n\n" + active

    tools = [search_documents, submit_answer]
    if use_tasks:
        tools += [create_task_tool, list_tasks_tool, open_task_tool]

    return Agent(
        name="report_agent",
        model=LiteLlm(
            model=MODEL, api_key=settings.openai_api_key, timeout=LLM_CALL_TIMEOUT_SECONDS
        ),
        instruction=instruction,
        tools=tools,
    )


async def _run(
    question: str,
    state: _RunState,
    max_llm_calls: int,
    search_top_k: int,
    service: BaseSessionService,
    session_id: str | None,
    use_tasks: bool,
) -> int:
    """Run one turn of the agent in a new or an existing session; return the total tokens used."""
    if session_id is None:
        session = await service.create_session(app_name=APP_NAME, user_id=USER_ID)
    else:
        session = await service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
        if session is None:
            raise AgentSessionNotFoundError(session_id)

    state.session_id = session.id
    state.chunks = {cid: RetrievedChunk.model_validate(c) for cid, c in session.state.get(SEEN_CHUNKS_KEY, {}).items()}

    active_task = session.state.get(ACTIVE_TASK_KEY)
    state.task = Task.model_validate(active_task) if active_task else None

    runner = Runner(agent=_build_agent(state, max_llm_calls, search_top_k, use_tasks), app_name=APP_NAME, session_service=service)
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
    fed_chunks: list[RetrievedChunk]  # every chunk the model was shown (in the whole session)
    selected_chunks: list[RetrievedChunk]  # the chunks this run's answer cites


def _execute(
    question: str,
    max_llm_calls: int,
    search_top_k: int,
    trace: tracing.TraceAttributes,
    service: BaseSessionService,
    session_id: str | None,
    use_tasks: bool,
) -> AgentRun:
    state = _RunState()
    start = time.perf_counter()

    try:
        with tracing.trace_attributes(trace):
            tokens_used = asyncio.run(_run(question, state, max_llm_calls, search_top_k, service, session_id, use_tasks))
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
    base = AskResponse(answer=state.answer, tokens_used=tokens_used, response_time_seconds=elapsed)
    return AgentRun(
        response=AgentAskResponse(**base.model_dump(), steps=state.steps, session_id=state.session_id, task=state.task),
        fed_chunks=fed_chunks,
        selected_chunks=[state.chunks[cid] for cid in state.cited_ids],
    )


def run_agent_with_chunks(
    question: str, max_llm_calls: int, search_top_k: int, trace: tracing.TraceAttributes
) -> AgentRun:
    """Answer a single question in a fresh in-memory session, as evaluation runs do."""
    return _execute(question, max_llm_calls, search_top_k, trace, InMemorySessionService(), None, use_tasks=False)


async def _save_turn(session_id: str, question: str, response: AgentAskResponse) -> None:
    """Append the turn to the chat's saved turns (kept in session state, next to the sources)."""
    session = await _chat_sessions.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
    turns = [*session.state.get(TURNS_KEY, []), {"question": question, "response": response.model_dump(mode="json")}]
    event = Event(author=APP_NAME, actions=EventActions(state_delta={TURNS_KEY: turns}))
    await _chat_sessions.append_event(session, event)


def run_agent(question: str, session_id: str | None) -> AgentAskResponse:
    """The Agent tab's entry point: one turn of a persisted chat (a new chat if session_id is
    None), with this module's default settings."""
    trace = tracing.TraceAttributes(
        name="agent-ask", environment="interactive", tags=["interactive"], metadata={}
    )
    response = _execute(question, MAX_LLM_CALLS, TOP_K, trace, _chat_sessions, session_id, use_tasks=True).response
    asyncio.run(_save_turn(response.session_id, question, response))
    return response
