"""add rag vector search

Revision ID: 20260616_000001
Revises: 20260614_000001
Create Date: 2026-06-16 00:00:01
"""

import sqlalchemy as sa

from alembic import op

revision = "20260616_000001"
down_revision = "20260614_000001"
branch_labels = None
depends_on = None


def _embedding_column_type():  # type: ignore[no-untyped-def]
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        from pgvector.sqlalchemy import Vector

        return Vector()
    return sa.JSON()


def _metadata_column_type():  # type: ignore[no-untyped-def]
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import JSONB

        return JSONB()
    return sa.JSON()


def upgrade() -> None:
    bind = op.get_bind()

    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.alter_column(
        "documents",
        "status",
        existing_type=sa.String(length=10),
        type_=sa.String(length=20),
        existing_nullable=False,
    )

    op.execute("UPDATE documents SET status = 'indexed' WHERE status = 'ready'")
    op.execute("UPDATE documents SET status = 'indexing' WHERE status = 'processing'")
    op.execute("UPDATE documents SET status = 'index_failed' WHERE status = 'failed'")

    op.create_table(
        "document_chunks",
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("embedding", _embedding_column_type(), nullable=False),
        sa.Column("metadata", _metadata_column_type(), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_document_chunks_document_index"),
    )
    op.create_index(
        op.f("ix_document_chunks_document_id"),
        "document_chunks",
        ["document_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_document_chunks_workspace_id"),
        "document_chunks",
        ["workspace_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_document_chunks_workspace_id"), table_name="document_chunks")
    op.drop_index(op.f("ix_document_chunks_document_id"), table_name="document_chunks")
    op.drop_table("document_chunks")

    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("UPDATE documents SET status = 'ready' WHERE status = 'indexed'")
        op.execute("UPDATE documents SET status = 'processing' WHERE status = 'indexing'")
        op.execute("UPDATE documents SET status = 'failed' WHERE status = 'index_failed'")
