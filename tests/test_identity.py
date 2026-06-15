import pytest


@pytest.mark.asyncio
async def test_register_login_and_me(client) -> None:
    register_response = await client.post(
        "/api/v1/auth/register",
        json={"email": "user@example.com", "password": "password123"},
    )

    assert register_response.status_code == 201
    assert register_response.json()["email"] == "user@example.com"

    login_response = await client.post(
        "/api/v1/auth/login",
        json={"email": "user@example.com", "password": "password123"},
    )

    assert login_response.status_code == 200
    token = login_response.json()["access_token"]

    me_response = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert me_response.status_code == 200
    assert me_response.json()["email"] == "user@example.com"
