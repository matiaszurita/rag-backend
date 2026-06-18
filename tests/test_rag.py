import io
from inspect import getsource
from uuid import UUID, uuid4

import pytest
import sqlalchemy as sa

from rag_backend.core.database import get_session_maker
from rag_backend.modules.documents.infrastructure.models import DocumentORM
from rag_backend.modules.rag.application.dtos import QueryRagCommand, RetrievalMetadataDTO
from rag_backend.modules.rag.application.services import QueryRagService
from rag_backend.modules.rag.domain.entities import RetrievalMode, RetrievalSource, SimilarChunk
from rag_backend.modules.rag.infrastructure.fakes import FakeEmbeddingProvider, FakeLLMProvider
from rag_backend.modules.rag.infrastructure.models import DocumentChunkORM
from rag_backend.modules.rag.interfaces.router import get_embedding_provider, get_llm_provider

INSUFFICIENT_CONTEXT_ANSWER = (
    "No encontré información suficiente en los documentos indexados para responder esta pregunta."
)


class AlwaysExistingDocuments:
    async def get_document_for_owner(self, **kwargs):  # noqa: ANN003
        return None

    async def workspace_exists_for_owner(self, **kwargs) -> bool:  # noqa: ANN003
        return True

    async def read_document_content(self, storage_path: str) -> bytes:
        return b""

    async def update_document_status(self, document_id, status) -> None:  # noqa: ANN001
        return None

    async def commit(self) -> None:
        return None


class StaticRetrieval:
    def __init__(self, results: list[SimilarChunk], metadata: RetrievalMetadataDTO) -> None:
        self.results = results
        self.metadata = metadata

    async def retrieve(self, **kwargs):  # noqa: ANN003
        return self.results, self.metadata


async def _create_user_workspace(client, email: str):
    await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "password123"},
    )
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "password123"},
    )
    token = login.json()["access_token"]
    workspace = await client.post(
        "/api/v1/workspaces",
        json={"name": f"Workspace {email}", "description": None},
        headers={"Authorization": f"Bearer {token}"},
    )
    return token, workspace.json()["id"]


async def _upload_document(client, token: str, workspace_id: str, filename: str, content: bytes):
    response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/documents",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": (filename, io.BytesIO(content), "text/plain")},
    )
    assert response.status_code == 201
    return response.json()


async def _chunk_count(document_id: str) -> int:
    session_maker = get_session_maker()
    async with session_maker() as session:
        result = await session.execute(
            sa.select(sa.func.count()).select_from(DocumentChunkORM).where(
                DocumentChunkORM.document_id == UUID(document_id)
            )
        )
        return result.scalar_one()


async def _chunk_embeddings(document_id: str) -> list[list[float]]:
    session_maker = get_session_maker()
    async with session_maker() as session:
        result = await session.execute(
            sa.select(DocumentChunkORM.embedding).where(
                DocumentChunkORM.document_id == UUID(document_id)
            )
        )
        return list(result.scalars().all())


@pytest.mark.asyncio
async def test_index_txt_document_creates_chunks_and_embeddings(client) -> None:
    token, workspace_id = await _create_user_workspace(client, "rag-txt@example.com")
    document = await _upload_document(
        client,
        token,
        workspace_id,
        "notes.txt",
        b"alpha beta knowledge\n\ngamma delta notes",
    )

    response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/documents/{document['id']}/index",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "indexed"
    assert response.json()["chunks_indexed"] == 2
    assert await _chunk_count(document["id"]) == 2
    assert all(embedding for embedding in await _chunk_embeddings(document["id"]))


@pytest.mark.asyncio
async def test_index_md_document(client) -> None:
    token, workspace_id = await _create_user_workspace(client, "rag-md@example.com")
    document = await _upload_document(
        client,
        token,
        workspace_id,
        "guide.md",
        b"# Guide\n\nsemantic markdown content",
    )

    response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/documents/{document['id']}/index",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "indexed"
    assert await _chunk_count(document["id"]) == 2


@pytest.mark.asyncio
async def test_semantic_search_returns_results_from_owned_workspace_only(client) -> None:
    token_one, workspace_one = await _create_user_workspace(client, "rag-search-one@example.com")
    token_two, workspace_two = await _create_user_workspace(client, "rag-search-two@example.com")
    document_one = await _upload_document(
        client,
        token_one,
        workspace_one,
        "alpha.txt",
        b"alpha project context",
    )
    document_two = await _upload_document(
        client,
        token_two,
        workspace_two,
        "alpha.md",
        b"alpha other workspace",
    )
    await client.post(
        f"/api/v1/workspaces/{workspace_one}/documents/{document_one['id']}/index",
        headers={"Authorization": f"Bearer {token_one}"},
    )
    await client.post(
        f"/api/v1/workspaces/{workspace_two}/documents/{document_two['id']}/index",
        headers={"Authorization": f"Bearer {token_two}"},
    )

    response = await client.post(
        f"/api/v1/workspaces/{workspace_one}/rag/search",
        headers={"Authorization": f"Bearer {token_one}"},
        json={"query": "alpha", "top_k": 5},
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert results
    assert {item["document_id"] for item in results} == {document_one["id"]}
    assert response.json()["retrieval_mode"] == "hybrid"
    assert response.json()["metadata"]["final_results"] == len(results)


@pytest.mark.asyncio
async def test_rag_query_returns_answer_and_sources(app, client) -> None:
    token, workspace_id = await _create_user_workspace(client, "rag-query@example.com")
    document = await _upload_document(
        client,
        token,
        workspace_id,
        "alpha.txt",
        b"alpha project context\n\nbeta deployment notes",
    )
    await client.post(
        f"/api/v1/workspaces/{workspace_id}/documents/{document['id']}/index",
        headers={"Authorization": f"Bearer {token}"},
    )

    response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/rag/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"question": "What does alpha say?", "top_k": 5},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["question"] == "What does alpha say?"
    assert payload["answer"] == "fake grounded answer"
    assert payload["metadata"]["context_chunks_used"] >= 1
    assert payload["metadata"]["top_k"] == 5
    assert payload["metadata"]["llm_model"] == "models/gemini-2.5-flash"
    assert payload["metadata"]["retrieval_mode"] == "hybrid"
    assert app.state.fake_llm_provider.calls
    source = payload["sources"][0]
    assert source["chunk_id"]
    assert source["document_id"] == document["id"]
    assert source["filename"] == "alpha.txt"
    assert isinstance(source["score"], float)
    assert "vector_score" in source
    assert "keyword_score" in source
    assert source["retrieval_source"] in {"vector", "keyword", "hybrid"}
    assert source["content_preview"]


@pytest.mark.asyncio
async def test_rag_query_returns_insufficient_context_without_llm_call(app, client) -> None:
    token, workspace_id = await _create_user_workspace(client, "rag-query-empty@example.com")

    response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/rag/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"question": "What is indexed?", "top_k": 5},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == INSUFFICIENT_CONTEXT_ANSWER
    assert payload["sources"] == []
    assert payload["metadata"]["context_chunks_used"] == 0
    assert payload["metadata"]["retrieval_mode"] == "hybrid"
    assert app.state.fake_llm_provider.calls == []


@pytest.mark.asyncio
async def test_rag_query_rejects_other_users_workspace_without_llm_call(app, client) -> None:
    token_one, _ = await _create_user_workspace(client, "rag-query-owner-one@example.com")
    _, workspace_two = await _create_user_workspace(client, "rag-query-owner-two@example.com")

    response = await client.post(
        f"/api/v1/workspaces/{workspace_two}/rag/query",
        headers={"Authorization": f"Bearer {token_one}"},
        json={"question": "What is private?", "top_k": 5},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "workspace_not_found"
    assert app.state.fake_llm_provider.calls == []


@pytest.mark.asyncio
async def test_rag_query_uses_fake_llm_provider(app, client) -> None:
    fake_llm = FakeLLMProvider(answer="custom fake answer")
    app.dependency_overrides[get_llm_provider] = lambda: fake_llm
    token, workspace_id = await _create_user_workspace(client, "rag-query-fake@example.com")
    document = await _upload_document(
        client,
        token,
        workspace_id,
        "fake.txt",
        b"fake provider context",
    )
    await client.post(
        f"/api/v1/workspaces/{workspace_id}/documents/{document['id']}/index",
        headers={"Authorization": f"Bearer {token}"},
    )

    response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/rag/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"question": "fake provider", "top_k": 1},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "custom fake answer"
    assert len(fake_llm.calls) == 1
    app.dependency_overrides[get_llm_provider] = lambda: app.state.fake_llm_provider


@pytest.mark.asyncio
async def test_search_still_returns_chunks_only_after_query_endpoint(client) -> None:
    token, workspace_id = await _create_user_workspace(client, "rag-search-regression@example.com")
    document = await _upload_document(
        client,
        token,
        workspace_id,
        "search.txt",
        b"search debug chunk",
    )
    await client.post(
        f"/api/v1/workspaces/{workspace_id}/documents/{document['id']}/index",
        headers={"Authorization": f"Bearer {token}"},
    )

    response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/rag/search",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "search", "top_k": 5},
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"query", "retrieval_mode", "results", "metadata"}
    assert payload["retrieval_mode"] == "hybrid"
    assert payload["metadata"]["fusion_algorithm"] == "weighted_rrf"
    result = payload["results"][0]
    assert "vector_score" in result
    assert "keyword_score" in result
    assert result["retrieval_source"] in {"vector", "keyword", "hybrid"}
    assert "answer" not in payload


@pytest.mark.asyncio
async def test_search_includes_reranking_metadata_when_requested(client) -> None:
    token, workspace_id = await _create_user_workspace(client, "rag-search-rerank@example.com")
    document = await _upload_document(
        client,
        token,
        workspace_id,
        "rerank.txt",
        b"reranking debug chunk",
    )
    await client.post(
        f"/api/v1/workspaces/{workspace_id}/documents/{document['id']}/index",
        headers={"Authorization": f"Bearer {token}"},
    )

    response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/rag/search",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "reranking", "top_k": 1, "reranking_enabled": True},
    )

    assert response.status_code == 200
    metadata = response.json()["metadata"]
    assert metadata["reranking_enabled"] is True
    assert metadata["reranking_provider"] == "noop"
    assert metadata["reranking_applied"] is True
    assert metadata["candidates_before_rerank"] >= 1
    assert metadata["final_results"] == 1


@pytest.mark.asyncio
async def test_rag_query_context_follows_reranked_order() -> None:
    first = SimilarChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content="first reranked context",
        score=0.4,
        metadata={"source": "first.txt"},
        rerank_score=0.9,
        original_rank=2,
        reranked_rank=1,
        retrieval_source=RetrievalSource.VECTOR,
    )
    second = SimilarChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content="second reranked context",
        score=0.8,
        metadata={"source": "second.txt"},
        rerank_score=0.5,
        original_rank=1,
        reranked_rank=2,
        retrieval_source=RetrievalSource.VECTOR,
    )
    retrieval = StaticRetrieval(
        [first, second],
        RetrievalMetadataDTO(
            retrieval_mode=RetrievalMode.VECTOR,
            vector_candidates=2,
            keyword_candidates=0,
            vector_results=2,
            keyword_results=0,
            deduplicated_results=2,
            final_results=2,
            reranking_enabled=True,
            reranking_applied=True,
            candidates_before_rerank=2,
        ),
    )
    llm = FakeLLMProvider(answer="ordered answer")
    service = QueryRagService(
        documents=AlwaysExistingDocuments(),
        retrieval=retrieval,
        llm=llm,
        max_context_chunks=2,
        min_relevance_score=0.0,
        llm_model="fake-model",
    )

    result = await service.query(
        QueryRagCommand(
            owner_id=uuid4(),
            workspace_id=uuid4(),
            question="order?",
            top_k=2,
            reranking_enabled=True,
        )
    )

    user_prompt = llm.calls[0]["user_prompt"]
    assert user_prompt.index("first reranked context") < user_prompt.index(
        "second reranked context"
    )
    assert result.sources[0].reranked_rank == 1
    assert result.metadata.reranking_enabled is True
    assert result.metadata.reranking_applied is True


@pytest.mark.asyncio
async def test_search_supports_vector_keyword_and_hybrid_modes(client) -> None:
    token, workspace_id = await _create_user_workspace(client, "rag-modes@example.com")
    document = await _upload_document(
        client,
        token,
        workspace_id,
        "modes.txt",
        b"JWT_SECRET_KEY exact setting\n\nsemantic deployment context",
    )
    await client.post(
        f"/api/v1/workspaces/{workspace_id}/documents/{document['id']}/index",
        headers={"Authorization": f"Bearer {token}"},
    )

    for mode in ["vector", "keyword", "hybrid"]:
        response = await client.post(
            f"/api/v1/workspaces/{workspace_id}/rag/search",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "JWT_SECRET_KEY", "top_k": 5, "retrieval_mode": mode},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["retrieval_mode"] == mode
        assert payload["metadata"]["retrieval_mode"] == mode
        assert payload["results"]


@pytest.mark.asyncio
async def test_rag_query_accepts_keyword_retrieval_mode(app, client) -> None:
    token, workspace_id = await _create_user_workspace(client, "rag-query-keyword@example.com")
    document = await _upload_document(
        client,
        token,
        workspace_id,
        "keyword.txt",
        b"JWT_SECRET_KEY configures signing secrets",
    )
    await client.post(
        f"/api/v1/workspaces/{workspace_id}/documents/{document['id']}/index",
        headers={"Authorization": f"Bearer {token}"},
    )

    response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/rag/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"question": "JWT_SECRET_KEY", "top_k": 3, "retrieval_mode": "keyword"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "fake grounded answer"
    assert payload["metadata"]["retrieval_mode"] == "keyword"
    assert payload["sources"][0]["retrieval_source"] == "keyword"
    assert app.state.fake_llm_provider.calls


def test_query_rag_service_does_not_depend_on_gemini_directly() -> None:
    source = getsource(QueryRagService)
    assert "Gemini" not in source
    assert "langchain" not in source.lower()


@pytest.mark.asyncio
async def test_rag_query_llm_failure_returns_controlled_error(app, client) -> None:
    fake_llm = FakeLLMProvider(fail=True)
    app.dependency_overrides[get_llm_provider] = lambda: fake_llm
    token, workspace_id = await _create_user_workspace(client, "rag-query-llm-fail@example.com")
    document = await _upload_document(client, token, workspace_id, "fail.txt", b"failure context")
    await client.post(
        f"/api/v1/workspaces/{workspace_id}/documents/{document['id']}/index",
        headers={"Authorization": f"Bearer {token}"},
    )

    response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/rag/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"question": "failure context", "top_k": 1},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["code"] == "rag_answer_failed"
    assert "secret-token" not in str(payload)
    assert "Contexto recuperado" not in str(payload)
    app.dependency_overrides[get_llm_provider] = lambda: app.state.fake_llm_provider


@pytest.mark.asyncio
async def test_rag_query_empty_question_returns_bad_request(client) -> None:
    token, workspace_id = await _create_user_workspace(
        client,
        "rag-query-empty-question@example.com",
    )

    response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/rag/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"question": "   ", "top_k": 5},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "empty_question"


@pytest.mark.asyncio
async def test_search_rejects_other_users_workspace(client) -> None:
    token_one, _ = await _create_user_workspace(client, "rag-owner-one@example.com")
    _, workspace_two = await _create_user_workspace(client, "rag-owner-two@example.com")

    response = await client.post(
        f"/api/v1/workspaces/{workspace_two}/rag/search",
        headers={"Authorization": f"Bearer {token_one}"},
        json={"query": "alpha", "top_k": 5},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "workspace_not_found"


@pytest.mark.asyncio
async def test_index_rejects_other_users_document(client) -> None:
    token_one, _ = await _create_user_workspace(client, "rag-index-one@example.com")
    token_two, workspace_two = await _create_user_workspace(client, "rag-index-two@example.com")
    document_two = await _upload_document(
        client,
        token_two,
        workspace_two,
        "secret.txt",
        b"private alpha",
    )

    response = await client.post(
        f"/api/v1/workspaces/{workspace_two}/documents/{document_two['id']}/index",
        headers={"Authorization": f"Bearer {token_one}"},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "document_not_found"


@pytest.mark.asyncio
async def test_index_empty_file_marks_index_failed(client) -> None:
    token, workspace_id = await _create_user_workspace(client, "rag-empty@example.com")
    document = await _upload_document(client, token, workspace_id, "empty.txt", b"   ")

    response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/documents/{document['id']}/index",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "empty_document"
    get_response = await client.get(
        f"/api/v1/workspaces/{workspace_id}/documents/{document['id']}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert get_response.json()["status"] == "index_failed"


@pytest.mark.asyncio
async def test_index_unsupported_stored_type_marks_index_failed(client) -> None:
    token, workspace_id = await _create_user_workspace(client, "rag-unsupported@example.com")
    document = await _upload_document(client, token, workspace_id, "payload.txt", b"boom")
    session_maker = get_session_maker()
    async with session_maker() as session:
        await session.execute(
            sa.update(DocumentORM)
            .where(DocumentORM.id == UUID(document["id"]))
            .values(original_filename="payload.exe")
        )
        await session.commit()

    response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/documents/{document['id']}/index",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_document_type"
    get_response = await client.get(
        f"/api/v1/workspaces/{workspace_id}/documents/{document['id']}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert get_response.json()["status"] == "index_failed"


@pytest.mark.asyncio
async def test_embedding_failure_marks_index_failed(app, client) -> None:
    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider(fail=True)
    token, workspace_id = await _create_user_workspace(client, "rag-fail@example.com")
    document = await _upload_document(client, token, workspace_id, "fail.txt", b"alpha")

    response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/documents/{document['id']}/index",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "document_indexing_failed"
    get_response = await client.get(
        f"/api/v1/workspaces/{workspace_id}/documents/{document['id']}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert get_response.json()["status"] == "index_failed"

    app.dependency_overrides[get_embedding_provider] = lambda: FakeEmbeddingProvider()
