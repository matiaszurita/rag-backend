from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from rag_backend.core.config import get_settings
from rag_backend.core.database import get_db_session
from rag_backend.modules.identity.application.dtos import LoginUserCommand, RegisterUserCommand
from rag_backend.modules.identity.application.services import IdentityService
from rag_backend.modules.identity.domain.entities import User
from rag_backend.modules.identity.infrastructure.repositories import SqlAlchemyUserRepository
from rag_backend.modules.identity.infrastructure.security import (
    JwtTokenService,
    PasswordHasherAdapter,
)
from rag_backend.modules.identity.interfaces.dependencies import get_current_user
from rag_backend.modules.identity.interfaces.schemas import (
    AccessTokenResponse,
    LoginRequest,
    RegisterRequest,
    UserResponse,
)

router = APIRouter(prefix="/auth", tags=["identity"])


def build_identity_service(session: AsyncSession) -> IdentityService:
    settings = get_settings()
    return IdentityService(
        users=SqlAlchemyUserRepository(session),
        password_hasher=PasswordHasherAdapter(),
        token_service=JwtTokenService(settings),
    )


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    payload: RegisterRequest,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> UserResponse:
    service = build_identity_service(session)
    user = await service.register(
        RegisterUserCommand(email=payload.email, password=payload.password)
    )
    return UserResponse.model_validate(user, from_attributes=True)


@router.post("/login", response_model=AccessTokenResponse)
async def login(
    payload: LoginRequest,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> AccessTokenResponse:
    service = build_identity_service(session)
    token = await service.login(LoginUserCommand(email=payload.email, password=payload.password))
    return AccessTokenResponse.model_validate(token, from_attributes=True)


@router.get("/me", response_model=UserResponse)
async def me(current_user: Annotated[User, Depends(get_current_user)]) -> UserResponse:
    return UserResponse.model_validate(current_user, from_attributes=True)
