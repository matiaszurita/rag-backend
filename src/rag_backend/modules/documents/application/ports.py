from typing import Protocol
from uuid import UUID

from rag_backend.modules.documents.domain.entities import Document, DocumentStatus
from rag_backend.modules.workspaces.domain.entities import Workspace


class DocumentRepositoryPort(Protocol):
    async def add(
        self,
        *,
        document_id: UUID,
        workspace_id: UUID,
        original_filename: str,
        storage_path: str,
        content_type: str | None,
        status: DocumentStatus,
    ) -> Document: ...

    async def list_for_workspace(self, workspace_id: UUID) -> list[Document]: ...

    async def get_for_workspace(self, document_id: UUID, workspace_id: UUID) -> Document | None: ...

    async def mark_deleted(self, document_id: UUID) -> None: ...

    async def commit(self) -> None: ...


class WorkspaceAccessPort(Protocol):
    async def get_by_id_for_owner(self, workspace_id: UUID, owner_id: UUID) -> Workspace | None: ...


class DocumentStoragePort(Protocol):
    async def save(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        filename: str,
        content: bytes,
    ) -> str: ...

    async def delete(self, storage_path: str) -> None: ...
