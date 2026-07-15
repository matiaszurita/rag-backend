from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

from rag_backend.core.database import Base, TimestampMixin, UUIDPrimaryKeyMixin
from rag_backend.modules.rag.domain.entities import ConversationMessageRole


class EmbeddingType(TypeDecorator):
    impl = sa.JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):  # type: ignore[no-untyped-def]
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector

            return dialect.type_descriptor(Vector())
        return dialect.type_descriptor(sa.JSON())


class MetadataType(TypeDecorator):
    impl = sa.JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):  # type: ignore[no-untyped-def]
        if dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import JSONB

            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(sa.JSON())


class DocumentChunkORM(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_document_chunks_document_index"),
    )

    workspace_id: Mapped[UUID] = mapped_column(
        sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[UUID] = mapped_column(
        sa.ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    content: Mapped[str] = mapped_column(sa.Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(EmbeddingType(), nullable=False)
    chunk_metadata: Mapped[dict[str, object] | None] = mapped_column(
        "metadata",
        MetadataType(),
        nullable=True,
    )


class ConversationORM(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "conversations"

    workspace_id: Mapped[UUID] = mapped_column(
        sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str | None] = mapped_column(sa.String(255), nullable=True)


class ConversationMessageORM(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "conversation_messages"
    __table_args__ = (
        sa.UniqueConstraint(
            "conversation_id",
            "message_index",
            name="uq_conversation_messages_conversation_index",
        ),
    )

    conversation_id: Mapped[UUID] = mapped_column(
        sa.ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    message_index: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    role: Mapped[ConversationMessageRole] = mapped_column(
        sa.Enum(ConversationMessageRole, name="conversation_message_role", native_enum=False),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(sa.Text, nullable=False)
    sources: Mapped[list[dict[str, object]] | None] = mapped_column(MetadataType(), nullable=True)
    message_metadata: Mapped[dict[str, object] | None] = mapped_column(
        "metadata",
        MetadataType(),
        nullable=True,
    )
