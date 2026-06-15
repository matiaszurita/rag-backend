from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from rag_backend.core.database import get_db_session
from rag_backend.modules.identity.domain.entities import User
from rag_backend.modules.identity.interfaces.dependencies import get_current_user
from rag_backend.modules.workspaces.application.dtos import CreateWorkspaceCommand
from rag_backend.modules.workspaces.application.services import WorkspaceService
from rag_backend.modules.workspaces.infrastructure.repositories import SqlAlchemyWorkspaceRepository
from rag_backend.modules.workspaces.interfaces.schemas import (
    CreateWorkspaceRequest,
    WorkspaceResponse,
)

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


def build_workspace_service(session: AsyncSession) -> WorkspaceService:
    return WorkspaceService(SqlAlchemyWorkspaceRepository(session))


@router.post("", response_model=WorkspaceResponse, status_code=status.HTTP_201_CREATED)
async def create_workspace(
    payload: CreateWorkspaceRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> WorkspaceResponse:
    service = build_workspace_service(session)
    workspace = await service.create(
        CreateWorkspaceCommand(
            owner_id=current_user.id,
            name=payload.name,
            description=payload.description,
        )
    )
    return WorkspaceResponse.model_validate(workspace, from_attributes=True)


@router.get("", response_model=list[WorkspaceResponse])
async def list_workspaces(
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> list[WorkspaceResponse]:
    service = build_workspace_service(session)
    workspaces = await service.list_for_owner(current_user.id)
    return [
        WorkspaceResponse.model_validate(workspace, from_attributes=True)
        for workspace in workspaces
    ]


@router.get("/{workspace_id}", response_model=WorkspaceResponse)
async def get_workspace(
    workspace_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> WorkspaceResponse:
    service = build_workspace_service(session)
    workspace = await service.get_for_owner(workspace_id, current_user.id)
    return WorkspaceResponse.model_validate(workspace, from_attributes=True)
