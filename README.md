# rag-backend

Backend base for the Technical Knowledge Workspace portfolio project. The backend is a modular monolith built for future RAG features, but this first phase only implements the platform foundation, identity, workspaces, document metadata management, and the initial RAG ports.

## Stack

- Python 3.12
- FastAPI + Uvicorn
- Pydantic Settings
- SQLAlchemy 2 async + asyncpg
- Alembic
- PostgreSQL + pgvector
- Redis
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

## Implemented in This Phase

- FastAPI app with `/api/v1` router and `GET /health`
- Environment-based settings with `.env.example`
- Async SQLAlchemy session wiring
- Alembic environment and initial migration
- `pgvector` extension enablement in the baseline migration
- Identity module with `register`, `login`, and `me`
- Workspace module with create, list, and get-by-id
- Documents module with local file storage, metadata persistence, listing, retrieval, and logical deletion
- Initial provider-agnostic RAG contracts only

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

## Design Notes

- Use cases do not depend directly on FastAPI or SQLAlchemy.
- Repository and storage behavior is exposed through application ports.
- IDs use UUIDs and persisted records include timestamps.
- Redis is included for future async workers but is not used yet.
- No LLM provider, embedding provider, LangChain, or full RAG implementation is included in this phase.
