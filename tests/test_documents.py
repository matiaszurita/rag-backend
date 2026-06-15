import io

import pytest


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


@pytest.mark.asyncio
async def test_upload_document_metadata(client) -> None:
    token, workspace_id = await _create_workspace(client)

    response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/documents",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("guide.txt", io.BytesIO(b"hello world"), "text/plain")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["workspace_id"] == workspace_id
    assert body["original_filename"] == "guide.txt"
    assert body["status"] == "uploaded"
