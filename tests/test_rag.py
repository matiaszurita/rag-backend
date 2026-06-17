import io
from uuid import UUID

import pytest
import sqlalchemy as sa

from rag_backend.core.database import get_session_maker
from rag_backend.modules.documents.infrastructure.models import DocumentORM
from rag_backend.modules.rag.infrastructure.fakes import FakeEmbeddingProvider
from rag_backend.modules.rag.infrastructure.models import DocumentChunkORM
from rag_backend.modules.rag.interfaces.router import get_embedding_provider


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
