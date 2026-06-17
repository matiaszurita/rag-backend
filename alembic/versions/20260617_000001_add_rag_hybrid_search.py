"""add rag hybrid search

Revision ID: 20260617_000001
Revises: 20260616_000001
Create Date: 2026-06-17 00:00:01
"""

from alembic import op

revision = "20260617_000001"
down_revision = "20260616_000001"
branch_labels = None
depends_on = None

INDEX_NAME = "ix_document_chunks_content_fts"


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            f"""
            CREATE INDEX {INDEX_NAME}
            ON document_chunks
            USING gin (to_tsvector('simple', content))
            """
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")
