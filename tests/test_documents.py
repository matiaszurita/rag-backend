import io
from pathlib import Path

import pytest

from rag_backend.core.config import get_settings


async def _create_workspace(client):
    await client.post(
        "/api/v1/auth/register",
        json={"email": "docs@example.com", "password": "password123"},
    )
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": "docs@example.com", "password": "password123"},
    )
    token = login.json()["access_token"]
    workspace = await client.post(
        "/api/v1/workspaces",
        json={"name": "Docs Workspace", "description": None},
        headers={"Authorization": f"Bearer {token}"},
    )
    return token, workspace.json()["id"]


def _stored_files() -> list[Path]:
    storage_root = get_settings().local_storage_path
    if not storage_root.exists():
        return []
    return [path for path in storage_root.rglob("*") if path.is_file()]


@pytest.mark.asyncio
async def test_upload_document_metadata(client) -> None:
    token, workspace_id = await _create_workspace(client)

    response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/documents",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("guide.md", io.BytesIO(b"hello world"), "text/markdown")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["workspace_id"] == workspace_id
    assert body["original_filename"] == "guide.md"
    assert body["status"] == "uploaded"
    assert (get_settings().local_storage_path / body["storage_path"]).exists()


@pytest.mark.asyncio
async def test_upload_document_rejects_unsupported_extension(client) -> None:
    token, workspace_id = await _create_workspace(client)

    response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/documents",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("payload.exe", io.BytesIO(b"boom"), "application/octet-stream")},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_document_type"
    assert _stored_files() == []

    list_response = await client.get(
        f"/api/v1/workspaces/{workspace_id}/documents",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert list_response.status_code == 200
    assert list_response.json() == []


@pytest.mark.asyncio
async def test_upload_document_rejects_oversized_file(client) -> None:
    token, workspace_id = await _create_workspace(client)
    content = b"a" * (get_settings().max_upload_size_bytes + 1)

    response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/documents",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("large.pdf", io.BytesIO(content), "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "document_too_large"
    assert _stored_files() == []

    list_response = await client.get(
        f"/api/v1/workspaces/{workspace_id}/documents",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert list_response.status_code == 200
    assert list_response.json() == []
