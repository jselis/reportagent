from openai import OpenAI

from app.config import settings

client = OpenAI(api_key=settings.openai_api_key)


def research_and_answer(question: str) -> dict:
    """Ask the model to research the question via web search and answer it."""
    response = client.responses.create(
        model=settings.openai_model,
        tools=[{"type": "web_search_preview"}],
        input=question,
    )

    sources = []
    for item in response.output:
        if item.type != "message":
            continue
        for content in item.content:
            for annotation in getattr(content, "annotations", []) or []:
                if annotation.type == "url_citation":
                    sources.append(
                        {"title": annotation.title, "url": annotation.url}
                    )

    return {
        "answer": response.output_text,
        "sources": sources,
    }
