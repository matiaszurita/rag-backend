# rag-backend

Backend base for the Technical Knowledge Workspace portfolio project. The backend is a modular monolith for ContextVault with identity, workspaces, document management, vector search, and source-backed RAG query answering.

## Stack

- Python 3.12
- FastAPI + Uvicorn
- Pydantic Settings
- SQLAlchemy 2 async + asyncpg
- Alembic
- PostgreSQL + pgvector + full-text search
- Redis
- LangChain text splitting
- Gemini embeddings and LLM answer generation
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
- RAG Phase 2 with source-backed question answering over retrieved chunks
- RAG Phase 4 backend retrieval with configurable vector, keyword, and hybrid search
- RAG Phase 5 foundation with optional no-op/fake reranking behind an application port

## RAG

ContextVault uses a progressive RAG architecture. This backend currently implements Phases 1, 2, and backend Phase 4 retrieval:

- Extract text from uploaded `.txt`, `.md`, and `.pdf` documents
- Split document text into chunks
- Generate Gemini embeddings through `langchain-google-genai`
- Persist chunks and embeddings in PostgreSQL with pgvector
- Search semantically similar chunks inside an authenticated user's workspace
- Search lexical/full-text matches inside chunk content for exact technical terms
- Combine vector and keyword retrieval with weighted reciprocal rank fusion in hybrid mode
- Optionally rerank retrieved candidates through a provider-neutral `RerankerPort`
- Ask a question against retrieved chunks and generate a Gemini Flash answer with sources

The `/rag/search` endpoint returns retrieved chunks only and is kept as a debug retrieval endpoint. It exposes retrieval mode, score breakdowns, and retrieval metadata. The `/rag/query` endpoint retrieves relevant chunks with the same retrieval modes, builds controlled context, calls the configured LLM, and returns an answer with sources.

Planned later phases are intentionally out of scope here:

- Real external reranking providers
- Parent-child chunks
- Conversation history and streaming
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
- `GEMINI_LLM_MODEL`
- `RAG_CHUNK_SIZE`
- `RAG_CHUNK_OVERLAP`
- `RAG_SEARCH_TOP_K`
- `RAG_ANSWER_MAX_CONTEXT_CHUNKS`
- `RAG_MIN_RELEVANCE_SCORE`
- `RAG_RETRIEVAL_MODE`
- `RAG_VECTOR_WEIGHT`
- `RAG_KEYWORD_WEIGHT`
- `RAG_VECTOR_CANDIDATES`
- `RAG_KEYWORD_CANDIDATES`
- `RAG_RERANKING_ENABLED`
- `RAG_RERANKING_PROVIDER`
- `RAG_RERANKING_CANDIDATES`

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
- `POST /api/v1/workspaces/{workspace_id}/rag/query`

Semantic search request:

```json
{
  "query": "deployment checklist",
  "top_k": 5,
  "retrieval_mode": "hybrid",
  "reranking_enabled": true
}
```

Semantic search response:

```json
{
  "query": "deployment checklist",
  "retrieval_mode": "hybrid",
  "results": [
    {
      "chunk_id": "...",
      "document_id": "...",
      "content": "...",
      "score": 0.82,
      "vector_score": 0.79,
      "keyword_score": 0.65,
      "rerank_score": null,
      "original_rank": null,
      "reranked_rank": null,
      "retrieval_source": "hybrid",
      "metadata": {}
    }
  ],
  "metadata": {
    "retrieval_mode": "hybrid",
    "vector_candidates": 20,
    "keyword_candidates": 20,
    "vector_results": 20,
    "keyword_results": 8,
    "deduplicated_results": 24,
    "final_results": 5,
    "fusion_algorithm": "weighted_rrf",
    "reranking_enabled": true,
    "reranking_provider": "noop",
    "reranking_applied": true,
    "reranking_candidates": 20,
    "candidates_before_rerank": 5
  }
}
```

RAG query request:

```json
{
  "question": "What does the deployment checklist require?",
  "top_k": 5,
  "retrieval_mode": "hybrid",
  "reranking_enabled": true
}
```

RAG query response:

```json
{
  "question": "What does the deployment checklist require?",
  "answer": "The deployment checklist requires ...",
  "sources": [
    {
      "chunk_id": "...",
      "document_id": "...",
      "filename": "deployment.md",
      "score": 0.82,
      "vector_score": 0.79,
      "keyword_score": 0.65,
      "rerank_score": null,
      "original_rank": null,
      "reranked_rank": null,
      "retrieval_source": "hybrid",
      "content_preview": "..."
    }
  ],
  "metadata": {
    "context_chunks_used": 1,
    "top_k": 5,
    "llm_model": "models/gemini-2.5-flash",
    "context_char_count": 912,
    "retrieval_mode": "hybrid",
    "fusion_algorithm": "weighted_rrf",
    "reranking_enabled": true,
    "reranking_provider": "noop",
    "reranking_applied": true
  }
}
```

Retrieval modes:

- `vector`: embeds the request text and retrieves chunks by pgvector similarity.
- `keyword`: retrieves chunks by PostgreSQL full-text search over `document_chunks.content` and does not require a query embedding.
- `hybrid`: retrieves vector and keyword candidates, deduplicates by chunk ID, and ranks with weighted reciprocal rank fusion.

The default retrieval mode is configured with `RAG_RETRIEVAL_MODE`. Optional reranking is configured with `RAG_RERANKING_ENABLED`, `RAG_RERANKING_PROVIDER`, and `RAG_RERANKING_CANDIDATES`. The only provider in this phase is `noop`, which keeps behavior deterministic and avoids external provider calls.

PostgreSQL full-text behavior is production-specific; the default test suite uses deterministic SQLite-compatible behavior and optional PostgreSQL integration tests can be used to validate real FTS matching.

## Design Notes

- Use cases do not depend directly on FastAPI or SQLAlchemy.
- Repository and storage behavior is exposed through application ports.
- IDs use UUIDs and persisted records include timestamps.
- Redis is included for future async workers but is not used yet.
- LangChain and Gemini are isolated behind RAG infrastructure adapters.
- Tests use deterministic fake embeddings and fake LLM providers and do not call Gemini.
- No frontend, conversation history, streaming, real reranking provider, or parent-child chunking is included yet.
