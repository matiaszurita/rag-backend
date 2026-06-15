import pytest


async def _create_user_and_token(client):
    await client.post(
        "/api/v1/auth/register",
        json={"email": "owner@example.com", "password": "password123"},
    )
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "owner@example.com", "password": "password123"},
    )
    return response.json()["access_token"]


@pytest.mark.asyncio
async def test_create_workspace(client) -> None:
    token = await _create_user_and_token(client)

    response = await client.post(
        "/api/v1/workspaces",
        json={"name": "Technical Notes", "description": "Workspace description"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 201
    assert response.json()["name"] == "Technical Notes"
