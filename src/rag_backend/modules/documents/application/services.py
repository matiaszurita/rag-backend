from uuid import UUID, uuid4

from rag_backend.core.errors import NotFoundError
from rag_backend.modules.documents.application.dtos import DocumentDTO, UploadDocumentCommand
from rag_backend.modules.documents.application.ports import (
    DocumentRepositoryPort,
    DocumentStoragePort,
    WorkspaceAccessPort,
)
from rag_backend.modules.documents.domain.entities import Document, DocumentStatus


def _to_dto(document: Document) -> DocumentDTO:
    return DocumentDTO(
        id=document.id,
        workspace_id=document.workspace_id,
        original_filename=document.original_filename,
        storage_path=document.storage_path,
        content_type=document.content_type,
        status=document.status,
        created_at=document.created_at,
        updated_at=document.updated_at,
        deleted_at=document.deleted_at,
    )


class DocumentService:
    def __init__(
        self,
        documents: DocumentRepositoryPort,
        workspaces: WorkspaceAccessPort,
        storage: DocumentStoragePort,
    ) -> None:
        self.documents = documents
        self.workspaces = workspaces
        self.storage = storage

    async def upload(self, command: UploadDocumentCommand) -> DocumentDTO:
        workspace = await self.workspaces.get_by_id_for_owner(
            command.workspace_id,
            command.owner_id,
        )
        if workspace is None:
            raise NotFoundError("Workspace not found", code="workspace_not_found")

        document_id = uuid4()
        storage_path = await self.storage.save(
            workspace_id=workspace.id,
            document_id=document_id,
            filename=command.original_filename,
            content=command.content,
        )
        document = await self.documents.add(
            document_id=document_id,
            workspace_id=workspace.id,
            original_filename=command.original_filename,
            storage_path=storage_path,
            content_type=command.content_type,
            status=DocumentStatus.UPLOADED,
        )
        await self.documents.commit()
        return _to_dto(document)

    async def list_for_workspace(self, *, owner_id: UUID, workspace_id: UUID) -> list[DocumentDTO]:
        workspace = await self.workspaces.get_by_id_for_owner(workspace_id, owner_id)
        if workspace is None:
            raise NotFoundError("Workspace not found", code="workspace_not_found")
        documents = await self.documents.list_for_workspace(workspace.id)
        return [_to_dto(document) for document in documents]

    async def get(self, *, owner_id: UUID, workspace_id: UUID, document_id: UUID) -> DocumentDTO:
        workspace = await self.workspaces.get_by_id_for_owner(workspace_id, owner_id)
        if workspace is None:
            raise NotFoundError("Workspace not found", code="workspace_not_found")
        document = await self.documents.get_for_workspace(document_id, workspace.id)
        if document is None:
            raise NotFoundError("Document not found", code="document_not_found")
        return _to_dto(document)

    async def delete(self, *, owner_id: UUID, workspace_id: UUID, document_id: UUID) -> None:
        workspace = await self.workspaces.get_by_id_for_owner(workspace_id, owner_id)
        if workspace is None:
            raise NotFoundError("Workspace not found", code="workspace_not_found")
        document = await self.documents.get_for_workspace(document_id, workspace.id)
        if document is None:
            raise NotFoundError("Document not found", code="document_not_found")
        await self.storage.delete(document.storage_path)
        await self.documents.mark_deleted(document.id)
        await self.documents.commit()
