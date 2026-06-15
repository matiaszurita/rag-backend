from typing import Annotated
from uuid import UUID

import jwt
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from rag_backend.core.config import get_settings
from rag_backend.core.database import get_db_session
from rag_backend.core.errors import AuthenticationError
from rag_backend.core.security import decode_access_token
from rag_backend.modules.identity.domain.entities import User
from rag_backend.modules.identity.infrastructure.repositories import SqlAlchemyUserRepository

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> User:
    if credentials is None:
        raise AuthenticationError()

    settings = get_settings()
    try:
        payload = decode_access_token(
            credentials.credentials,
            secret_key=settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm,
        )
    except jwt.PyJWTError as exc:
        raise AuthenticationError("Invalid token", code="invalid_token") from exc

    subject = payload.get("sub")
    if subject is None:
        raise AuthenticationError("Invalid token subject", code="invalid_token")

    user = await SqlAlchemyUserRepository(session).get_by_id(UUID(subject))
    if user is None:
        raise AuthenticationError("Authenticated user was not found", code="user_not_found")

    return user
