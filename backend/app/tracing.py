"""Langfuse tracing for the agent.

Importing this module (before anything from google.adk runs) sets up tracing. The two objects
below meet through the global OpenTelemetry tracer provider: creating the Langfuse client
registers the provider and attaches Langfuse's span exporter to it, and the instrumentor
makes every ADK agent run, LLM call and tool call emit a span on that provider. Langfuse
sends spans in the background and flushes the remaining ones when the process exits.
"""

from contextlib import contextmanager
from dataclasses import dataclass

from langfuse import Langfuse, propagate_attributes
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

from app.config import settings

langfuse = Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    base_url=settings.langfuse_base_url,
)

GoogleADKInstrumentor().instrument()


@dataclass
class TraceAttributes:
    """How a traced agent run shows up in Langfuse, so runs can be told apart and filtered."""

    name: str  # trace name
    environment: str  # Langfuse environment, a top-level filter: "interactive" or "evaluation"
    tags: list[str]
    metadata: dict[str, str]  # e.g. eval_run_id, to filter all traces of one evaluation run


@contextmanager
def trace_attributes(attributes: TraceAttributes):
    """Apply the attributes to every span created inside the block."""
    with propagate_attributes(
        trace_name=attributes.name,
        environment=attributes.environment,
        tags=attributes.tags,
        metadata=attributes.metadata,
    ):
        yield
