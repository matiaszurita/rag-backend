"""add rag conversations phase 6

Revision ID: 20260630_000001
Revises: 20260617_000001
Create Date: 2026-06-30 00:00:01
"""

import sqlalchemy as sa

from alembic import op

revision = "20260630_000001"
down_revision = "20260617_000001"
branch_labels = None
depends_on = None


def _metadata_column_type():  # type: ignore[no-untyped-def]
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import JSONB

        return JSONB()
    return sa.JSON()


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=True),
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
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_conversations_workspace_id"), "conversations", ["workspace_id"])
    op.create_index(
        "ix_conversations_workspace_id_updated_at",
        "conversations",
        ["workspace_id", "updated_at"],
    )

    op.create_table(
        "conversation_messages",
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("message_index", sa.Integer(), nullable=False),
        sa.Column(
            "role",
            sa.Enum("user", "assistant", name="conversation_message_role", native_enum=False),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("sources", _metadata_column_type(), nullable=True),
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
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "conversation_id",
            "message_index",
            name="uq_conversation_messages_conversation_index",
        ),
    )
    op.create_index(
        op.f("ix_conversation_messages_conversation_id"),
        "conversation_messages",
        ["conversation_id"],
    )
    op.create_index(
        "ix_conversation_messages_conversation_id_message_index",
        "conversation_messages",
        ["conversation_id", "message_index"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_conversation_messages_conversation_id_message_index",
        table_name="conversation_messages",
    )
    op.drop_index(
        op.f("ix_conversation_messages_conversation_id"),
        table_name="conversation_messages",
    )
    op.drop_table("conversation_messages")
    op.drop_index("ix_conversations_workspace_id_updated_at", table_name="conversations")
    op.drop_index(op.f("ix_conversations_workspace_id"), table_name="conversations")
    op.drop_table("conversations")
