from typing import Protocol
from uuid import UUID

from rag_backend.modules.workspaces.domain.entities import Workspace


class WorkspaceRepositoryPort(Protocol):
    async def add(self, *, owner_id: UUID, name: str, description: str | None) -> Workspace: ...

    async def list_by_owner(self, owner_id: UUID) -> list[Workspace]: ...

    async def get_by_id_for_owner(self, workspace_id: UUID, owner_id: UUID) -> Workspace | None: ...

    async def commit(self) -> None: ...
