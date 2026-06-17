import asyncio
from pathlib import Path
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from rag_backend.core.errors import BadRequestError
from rag_backend.modules.documents.domain.entities import Document, DocumentStatus
from rag_backend.modules.documents.infrastructure.models import DocumentORM
from rag_backend.modules.workspaces.infrastructure.models import WorkspaceORM


def _to_domain(model: DocumentORM) -> Document:
    return Document(
        id=model.id,
        workspace_id=model.workspace_id,
        original_filename=model.original_filename,
        storage_path=model.storage_path,
        content_type=model.content_type,
        status=model.status,
        created_at=model.created_at,
        updated_at=model.updated_at,
        deleted_at=model.deleted_at,
    )


class SqlAlchemyDocumentAccessAdapter:
    def __init__(self, *, session: AsyncSession, storage_root: Path) -> None:
        self.session = session
        self.storage_root = storage_root

    async def get_document_for_owner(
        self,
        *,
        owner_id: UUID,
        workspace_id: UUID,
        document_id: UUID,
    ) -> Document | None:
        result = await self.session.execute(
            sa.select(DocumentORM)
            .join(WorkspaceORM, WorkspaceORM.id == DocumentORM.workspace_id)
            .where(
                WorkspaceORM.id == workspace_id,
                WorkspaceORM.owner_id == owner_id,
                DocumentORM.id == document_id,
                DocumentORM.workspace_id == workspace_id,
                DocumentORM.deleted_at.is_(None),
            )
        )
        model = result.scalar_one_or_none()
        return _to_domain(model) if model else None

    async def workspace_exists_for_owner(self, *, owner_id: UUID, workspace_id: UUID) -> bool:
        result = await self.session.execute(
            sa.select(WorkspaceORM.id).where(
                WorkspaceORM.id == workspace_id,
                WorkspaceORM.owner_id == owner_id,
            )
        )
        return result.scalar_one_or_none() is not None

    async def read_document_content(self, storage_path: str) -> bytes:
        target = self.storage_root / storage_path

        def read_file() -> bytes:
            if not target.exists():
                raise FileNotFoundError(storage_path)
            return target.read_bytes()

        try:
            return await asyncio.to_thread(read_file)
        except FileNotFoundError as error:
            raise BadRequestError(
                "Stored document file was not found",
                code="document_file_missing",
            ) from error

    async def update_document_status(
        self,
        document_id: UUID,
        status: DocumentStatus,
    ) -> None:
        await self.session.execute(
            sa.update(DocumentORM)
            .where(DocumentORM.id == document_id, DocumentORM.deleted_at.is_(None))
            .values(status=status)
        )

    async def commit(self) -> None:
        await self.session.commit()
