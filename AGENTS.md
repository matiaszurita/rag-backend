# ContextVault Agents Guide

## Project

ContextVault is a FastAPI backend for a RAG application. It is a modular monolith with hexagonal-style boundaries for identity, workspaces, documents, and RAG.

## Stack

- Python 3.12
- FastAPI
- SQLAlchemy async
- Alembic
- PostgreSQL with pgvector and full-text search
- Redis
- Gemini embeddings
- Gemini LLM
- Pytest
- Ruff

## Core Commands

- `ruff check .`
- `pytest`
- `alembic upgrade head`
- `docker compose up -d postgres redis`

## Repository Architecture

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

Each module should use these layer folders when applicable:

- `domain`: entities and core concepts.
- `application`: use cases, DTOs, and ports.
- `infrastructure`: repositories, provider adapters, storage, extraction, and framework integrations.
- `interfaces`: FastAPI routers, dependencies, and HTTP schemas.

## Layer Rules

- `domain` must not depend on frameworks, databases, providers, or HTTP concerns.
- `application` contains use cases and ports.
- `application` must not depend on FastAPI, SQLAlchemy, Gemini SDK, LangChain concrete classes, provider HTTP clients, or direct environment variables.
- `infrastructure` implements adapters and repositories behind application ports.
- `interfaces` contains FastAPI routers and HTTP schemas.
- Routers may compose concrete adapters and settings, but must not own business logic, retrieval logic, scoring, fusion, SQL, or provider calls.

## RAG Module Rules

- Document indexing belongs in `IndexDocumentService`.
- Retrieval orchestration belongs in `RetrievalService`.
- Search/debug retrieval use cases must delegate to `RetrievalService`.
- Query answering belongs in `QueryRagService`.
- Prompt construction belongs in `RagPromptBuilder`.
- Prompt builders only build prompts; they must not call Gemini, access the database, read settings, or know FastAPI.
- New retrieval strategies such as hybrid, reranking, and parent-child chunks must integrate through ports and application services, not directly in routers.
- Reranking should be introduced as a port and adapter, then composed around `RetrievalService`.
- Parent-child chunks must preserve compatibility with existing chunk retrieval and source response fields unless an API change is intentional and tested.

## Testing Rules

- Prefer unit tests with fakes for application services.
- Use integration tests only when a real database or provider boundary is necessary.
- Tests must not depend on real Gemini calls.
- Keep default tests deterministic and portable.
- Do not log API keys, JWT secrets, prompts containing secrets, or provider credentials.

## Migration Rules

- Every new table, column, index, or extension change must be represented in Alembic.
- Migrations should be reversible when reasonable.
- Be explicit and careful with pgvector and PostgreSQL-specific features.
- Do not change database schema during refactors unless fixing an existing defect.

## Settings And Environment Rules

- Every new environment variable must be added to `Settings`.
- Document new environment variables in `.env.example`.
- Do not hardcode API keys, model names, credentials, or environment-specific URLs in application code.
- Provider model names should come from settings unless a test fake owns the value.

## Security Rules

- Never print or commit API keys.
- Never expose JWT secrets.
- Never store secrets in the repository.
- External provider errors should be converted into controlled application errors.
- Avoid returning retrieved prompt context or secret-bearing payloads in error responses.

## Future Phase Rules

- Hybrid RAG retrieval stays behind `RetrievalService`.
- Reranking must be added as a port/adapter and must not be called directly from routers.
- Parent-child chunks must preserve compatibility with current chunks unless response changes are deliberate and tested.
- Public response changes must be intentional, documented, and covered by tests.
- Do not add frontend, streaming, history, or observability work as part of backend hardening changes.

## Style

- Keep line length at or below 100 characters.
- Use type hints.
- Keep services small and testable.
- Prefer composition over giant services.
- Make the smallest correct change and preserve existing behavior during refactors.
