# rag-backend

Backend base for the Technical Knowledge Workspace portfolio project. The backend is a modular monolith for ContextVault with identity, workspaces, document management, and Phase 1 RAG vector search.

## Stack

- Python 3.12
- FastAPI + Uvicorn
- Pydantic Settings
- SQLAlchemy 2 async + asyncpg
- Alembic
- PostgreSQL + pgvector
- Redis
- LangChain text splitting
- Gemini embeddings
- pypdf
- Pytest
- Ruff
- Docker Compose

## Architecture

The codebase follows a modular monolith with hexagonal boundaries per module.

```text
src/rag_backend/
  api/
  core/
  shared/
  modules/
    identity/
    workspaces/
    documents/
    rag/
```

Each business module separates:

- `domain`: entities and core business concepts
- `application`: use cases, DTOs, and ports
- `infrastructure`: SQLAlchemy repositories and external adapters
- `interfaces`: FastAPI routers, request schemas, and HTTP dependencies

Cross-cutting concerns such as settings, database setup, logging, auth primitives, and JSON error handling live in `rag_backend/core`.

## Implemented

- FastAPI app with `/api/v1` router and `GET /health`
- Environment-based settings with `.env.example`
- Async SQLAlchemy session wiring
- Alembic environment and initial migration
- `pgvector` extension enablement in the baseline migration
- Identity module with `register`, `login`, and `me`
- Workspace module with create, list, and get-by-id
- Documents module with local file storage, metadata persistence, listing, retrieval, and logical deletion
- RAG Phase 1 with explicit document indexing and workspace-scoped semantic search

## RAG Phase 1

ContextVault uses a progressive RAG architecture. This backend currently implements only Phase 1:

- Extract text from uploaded `.txt`, `.md`, and `.pdf` documents
- Split document text into chunks
- Generate Gemini embeddings through `langchain-google-genai`
- Persist chunks and embeddings in PostgreSQL with pgvector
- Search semantically similar chunks inside an authenticated user's workspace

The current RAG search endpoint returns retrieved chunks only. It does not generate an LLM answer yet.

Planned later phases are intentionally out of scope here:

- LLM answer generation with sources
- Hybrid semantic plus lexical search
- Reranking
- Parent-child chunks
- Redis/background indexing workers

## Local Development

1. Copy `.env.example` to `.env`.
2. Install dependencies:

```bash
pip install -e .[dev]
```

3. Start infrastructure with Docker Compose:

```bash
docker compose up --build
```

4. Run migrations:

```bash
alembic upgrade head
```

5. Start the API locally if needed:

```bash
uvicorn rag_backend.main:app --reload
```

For local frontend work, the Vite app normally runs at `http://localhost:5173`. The backend must allow that origin through `CORS_ALLOWED_ORIGINS`. If the frontend port changes, update `CORS_ALLOWED_ORIGINS` to match the new origin.

## Local vs Docker Database Host

The database host depends on where the API or Alembic process is running.

- If you run the API or `alembic` from your local machine, use `localhost` as the database host.
- If you run the API or `alembic` inside Docker Compose, use `postgres` as the database host.

Local example:

```bash
postgresql+asyncpg://postgres:postgres@localhost:5432/rag_backend
```

Docker example:

```bash
postgresql+asyncpg://postgres:postgres@postgres:5432/rag_backend
```

Do not commit a real `.env` file with local or shared secrets.

## Quality Commands

Run tests:

```bash
pytest
```

Run linting:

```bash
ruff check .
```

## Environment Variables

Important settings are documented in `.env.example`:

- `DATABASE_URL`
- `ALEMBIC_DATABASE_URL`
- `JWT_SECRET_KEY`
- `REDIS_URL`
- `LOCAL_STORAGE_PATH`
- `GEMINI_API_KEY`
- `GEMINI_EMBEDDING_MODEL`
- `RAG_CHUNK_SIZE`
- `RAG_CHUNK_OVERLAP`
- `RAG_SEARCH_TOP_K`

## API Overview

- `GET /health`
- `POST /api/v1/auth/register`
- `POST /api/v1/auth/login`
- `GET /api/v1/auth/me`
- `POST /api/v1/workspaces`
- `GET /api/v1/workspaces`
- `GET /api/v1/workspaces/{workspace_id}`
- `POST /api/v1/workspaces/{workspace_id}/documents`
- `GET /api/v1/workspaces/{workspace_id}/documents`
- `GET /api/v1/workspaces/{workspace_id}/documents/{document_id}`
- `DELETE /api/v1/workspaces/{workspace_id}/documents/{document_id}`
- `POST /api/v1/workspaces/{workspace_id}/documents/{document_id}/index`
- `POST /api/v1/workspaces/{workspace_id}/rag/search`

Semantic search request:

```json
{
  "query": "deployment checklist",
  "top_k": 5
}
```

Semantic search response:

```json
{
  "query": "deployment checklist",
  "results": [
    {
      "chunk_id": "...",
      "document_id": "...",
      "content": "...",
      "score": 0.82,
      "metadata": {}
    }
  ]
}
```

## Design Notes

- Use cases do not depend directly on FastAPI or SQLAlchemy.
- Repository and storage behavior is exposed through application ports.
- IDs use UUIDs and persisted records include timestamps.
- Redis is included for future async workers but is not used yet.
- LangChain and Gemini are isolated behind RAG infrastructure adapters.
- Tests use deterministic fake embeddings and do not call Gemini.
- No chat endpoint, generated answer, hybrid search, reranking, or parent-child chunking is included yet.
