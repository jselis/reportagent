# reportagent

Capstone project for the **AI Agent Engineering** course by Maven.

## Purpose

This project builds an **agentic RAG (Retrieval-Augmented Generation) system** that goes beyond simple question answering. In addition to retrieving and grounding answers in a knowledge base, the agent is capable of **generating structured reports** — well-organized documents synthesized from the data exposed through the RAG pipeline.

## Goals

- Implement a RAG pipeline for retrieving relevant context from a knowledge base
- Build an agentic layer on top of RAG that can reason about *what* to retrieve and *how* to use it
- Extend the agent's capabilities to produce structured, report-style outputs (not just conversational answers)

## Architecture

```
                ┌─────────────┐
    :80 ──────▶ │   Traefik   │  reverse proxy
                └──────┬──────┘
                       │
          ┌────────────┴────────────┐
          │                         │
   PathPrefix `/`            PathPrefix `/api`
          │                         │
 ┌────────▼────────┐      ┌─────────▼─────────┐
 │     frontend     │      │      backend       │
 │  React (Vite)    │      │      FastAPI       │
 └──────────────────┘      └────────────────────┘
```

- **traefik** — reverse proxy / router. Routes `/` to the frontend and `/api/*` to the backend (stripping the `/api` prefix before it reaches FastAPI). Dashboard at `localhost:8080`.
- **backend** — the FastAPI app from before (`/ask` endpoint), now living in [backend/app](backend/app).
- **frontend** — a React (Vite) scaffold in [frontend/src](frontend/src). Currently just a text input + button, not yet wired to the backend.
- A database + database-manager service will be added later.

## Running with Docker

```bash
cp backend/.env.example backend/.env   # then set OPENAI_API_KEY
docker compose up --build
```

- Frontend: [http://localhost](http://localhost)
- Backend (via Traefik): `http://localhost/api/...` — e.g. `POST http://localhost/api/ask`
- Backend docs directly: [http://localhost/api/docs](http://localhost/api/docs)
- Traefik dashboard: [http://localhost:8080](http://localhost:8080)

> **macOS + Docker Desktop note:** if Traefik can't reach the Docker socket (check `docker compose logs traefik`), your Docker Desktop context may not expose `/var/run/docker.sock` on the host. Find your real socket with `docker context inspect | grep Host`, then run compose with e.g. `DOCKER_SOCK=/Users/<you>/.docker/run/docker.sock docker compose up --build`.

### `POST /api/ask`

Takes a question, researches it using the OpenAI web search tool, and returns an answer with cited sources.

```bash
curl -X POST http://localhost/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the latest developments in agentic RAG?"}'
```

## Running without Docker

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then set OPENAI_API_KEY
uvicorn app.main:app --reload
```
