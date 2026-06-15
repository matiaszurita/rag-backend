from uuid import UUID

from rag_backend.core.errors import NotFoundError
from rag_backend.modules.workspaces.application.dtos import CreateWorkspaceCommand, WorkspaceDTO
from rag_backend.modules.workspaces.application.ports import WorkspaceRepositoryPort
from rag_backend.modules.workspaces.domain.entities import Workspace


def _to_dto(workspace: Workspace) -> WorkspaceDTO:
    return WorkspaceDTO(
        id=workspace.id,
        owner_id=workspace.owner_id,
        name=workspace.name,
        description=workspace.description,
        created_at=workspace.created_at,
        updated_at=workspace.updated_at,
    )


class WorkspaceService:
    def __init__(self, workspaces: WorkspaceRepositoryPort) -> None:
        self.workspaces = workspaces

    async def create(self, command: CreateWorkspaceCommand) -> WorkspaceDTO:
        workspace = await self.workspaces.add(
            owner_id=command.owner_id,
            name=command.name,
            description=command.description,
        )
        await self.workspaces.commit()
        return _to_dto(workspace)

    async def list_for_owner(self, owner_id: UUID) -> list[WorkspaceDTO]:
        workspaces = await self.workspaces.list_by_owner(owner_id)
        return [_to_dto(workspace) for workspace in workspaces]

    async def get_for_owner(self, workspace_id: UUID, owner_id: UUID) -> WorkspaceDTO:
        workspace = await self.workspaces.get_by_id_for_owner(workspace_id, owner_id)
        if workspace is None:
            raise NotFoundError("Workspace not found", code="workspace_not_found")
        return _to_dto(workspace)
