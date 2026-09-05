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

- **traefik** — reverse proxy / router. Routes `/` to the frontend and `/api/*` to the backend, passing the full path through unchanged. Dashboard at `localhost:8080`. Local-only — not used in cloud deploys (see below).
- **backend** — FastAPI app in [backend/app](backend/app), exposing `/api/ask`, `/api/summarize`, and `/api/analyze-sentiment`, all backed by a shared LLM engine ([backend/app/llm.py](backend/app/llm.py)).
- **frontend** — a React (Vite) app in [frontend/src](frontend/src) with a mode selector (Ask / Summarize / Analyze Sentiment) wired to the corresponding backend endpoint.
- A database + database-manager service will be added later.

## Running with Docker

```bash
cp .env.example .env                   # sets BACKEND_PORT
cp backend/.env.example backend/.env   # then set OPENAI_API_KEY
docker compose up --build
```

- Frontend: [http://localhost](http://localhost)
- Backend (via Traefik): `http://localhost/api/...` — e.g. `POST http://localhost/api/ask`
- Backend docs directly: [http://localhost/api/docs](http://localhost/api/docs)
- Traefik dashboard: [http://localhost:8080](http://localhost:8080)

> **Traefik can't reach the Docker socket?** Check `docker compose logs traefik` for repeated `Failed to retrieve information of the docker client and server host` errors. This has been observed when the Traefik image version is older than your local Docker daemon (`docker version --format '{{.Server.Version}}'`) — its bundled Docker API client fails to negotiate with a newer daemon. Bump the `traefik` image tag in `docker-compose.yml` to a more recent release.

### `POST /api/ask`

Takes a question, researches it using the OpenAI web search tool, and returns an answer with cited sources.

```bash
curl -X POST http://localhost/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the latest developments in agentic RAG?"}'
```

## Deploying to Railway / Render

Neither platform runs `docker-compose.yml` as one unit — each service (`backend`, `frontend`) is deployed independently with its own public domain, so there's no shared Traefik here. Instead:

- **backend** — deploy from `backend/Dockerfile` (root directory `backend/`). It already reads `PORT` from the environment (both platforms inject this automatically). Set these environment variables in the platform's dashboard:
  - `OPENAI_API_KEY`
  - `DEBUG=false` (never leave this `true` in a public deployment — it exposes fault-injection endpoints)
  - `ALLOWED_ORIGINS` — the frontend's public URL once you know it (comma-separated if more than one)
- **frontend** — deploy from `frontend/Dockerfile`, targeting the `prod` stage (build + serve static files), not `dev`. Set:
  - `VITE_API_BASE_URL` — the backend's public URL, **available at build time** (Vite bakes it into the static bundle; setting it only at runtime has no effect since the JS is already built)
  - `PORT` — usually auto-injected by the platform; the container binds to whatever it's given

Both env vars default to values that preserve local behavior (`ALLOWED_ORIGINS=http://localhost`, `VITE_API_BASE_URL=` empty → relative paths through Traefik) — they only need to change for a real deployment.

## Running without Docker

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then set OPENAI_API_KEY
uvicorn app.main:app --reload
```
