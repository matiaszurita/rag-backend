from datetime import UTC, datetime
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from rag_backend.modules.documents.domain.entities import Document, DocumentStatus
from rag_backend.modules.documents.infrastructure.models import DocumentORM


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


class SqlAlchemyDocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(
        self,
        *,
        document_id: UUID,
        workspace_id: UUID,
        original_filename: str,
        storage_path: str,
        content_type: str | None,
        status: DocumentStatus,
    ) -> Document:
        model = DocumentORM(
            id=document_id,
            workspace_id=workspace_id,
            original_filename=original_filename,
            storage_path=storage_path,
            content_type=content_type,
            status=status,
        )
        self.session.add(model)
        await self.session.flush()
        await self.session.refresh(model)
        return _to_domain(model)

    async def list_for_workspace(self, workspace_id: UUID) -> list[Document]:
        result = await self.session.execute(
            sa.select(DocumentORM)
            .where(DocumentORM.workspace_id == workspace_id, DocumentORM.deleted_at.is_(None))
            .order_by(DocumentORM.created_at.desc())
        )
        return [_to_domain(model) for model in result.scalars().all()]

    async def get_for_workspace(self, document_id: UUID, workspace_id: UUID) -> Document | None:
        result = await self.session.execute(
            sa.select(DocumentORM).where(
                DocumentORM.id == document_id,
                DocumentORM.workspace_id == workspace_id,
                DocumentORM.deleted_at.is_(None),
            )
        )
        model = result.scalar_one_or_none()
        return _to_domain(model) if model else None

    async def mark_deleted(self, document_id: UUID) -> None:
        await self.session.execute(
            sa.update(DocumentORM)
            .where(DocumentORM.id == document_id)
            .values(deleted_at=datetime.now(UTC))
        )

    async def commit(self) -> None:
        await self.session.commit()
