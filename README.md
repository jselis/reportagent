# reportagent

Capstone project for the **AI Agent Engineering** course by Maven.

## Purpose

This project builds an **agentic RAG (Retrieval-Augmented Generation) system** that goes beyond simple question answering. In addition to retrieving and grounding answers in a knowledge base, the agent is capable of **generating structured reports** — well-organized documents synthesized from the data exposed through the RAG pipeline.

## Goals

- Implement a RAG pipeline for retrieving relevant context from a knowledge base
- Build an agentic layer on top of RAG that can reason about *what* to retrieve and *how* to use it
- Extend the agent's capabilities to produce structured, report-style outputs (not just conversational answers)

## Running the API

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then set OPENAI_API_KEY
uvicorn app.main:app --reload
```

### `POST /ask`

Takes a question, researches it using the OpenAI web search tool, and returns an answer with cited sources.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the latest developments in agentic RAG?"}'
```
