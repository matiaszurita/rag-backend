from datetime import datetime
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from rag_backend.core.database import Base, TimestampMixin, UUIDPrimaryKeyMixin
from rag_backend.modules.documents.domain.entities import DocumentStatus


class DocumentORM(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "documents"

    workspace_id: Mapped[UUID] = mapped_column(
        sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    original_filename: Mapped[str] = mapped_column(sa.String(512), nullable=False)
    storage_path: Mapped[str] = mapped_column(sa.String(1024), nullable=False)
    content_type: Mapped[str | None] = mapped_column(sa.String(255), nullable=True)
    status: Mapped[DocumentStatus] = mapped_column(
        sa.Enum(DocumentStatus, name="document_status", native_enum=False),
        nullable=False,
        default=DocumentStatus.UPLOADED,
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=True,
    )
