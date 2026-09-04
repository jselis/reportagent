import time

from openai import OpenAI

from app.config import settings
from app.models import Answer, AskResponse, Source

MODEL = "gpt-5.6-terra"

client = OpenAI(api_key=settings.openai_api_key)


def research_and_answer(question: str) -> AskResponse:
    """Ask the model to research the question via web search and answer it."""
    start = time.perf_counter()
    ttft = None

    with client.responses.stream(
        model=MODEL,
        #tools=[{"type": "web_search_preview"}],
        input=question,
    ) as stream:
        for event in stream:
            if ttft is None and event.type == "response.output_text.delta":
                ttft = time.perf_counter() - start
        response = stream.get_final_response()

    elapsed = time.perf_counter() - start

    sources = []
    for item in response.output:
        if item.type != "message":
            continue
        for content in item.content:
            for annotation in getattr(content, "annotations", []) or []:
                if annotation.type == "url_citation":
                    sources.append(
                        Source(title=annotation.title, url=annotation.url)
                    )

    answer = Answer(text=response.output_text, sources=sources)

    return AskResponse(
        answer=answer,
        tokens_used=response.usage.total_tokens,
        response_time_seconds=elapsed,
        ttft_seconds=ttft,
    )
