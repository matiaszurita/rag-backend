from collections.abc import AsyncGenerator
from datetime import datetime
from functools import lru_cache
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from rag_backend.core.config import get_settings


class Base(DeclarativeBase):
    pass


class UUIDPrimaryKeyMixin:
    id: Mapped[UUID] = mapped_column(sa.Uuid, primary_key=True, default=uuid4)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
        onupdate=sa.func.now(),
        nullable=False,
    )


def import_model_modules() -> None:
    import rag_backend.modules.documents.infrastructure.models  # noqa: F401
    import rag_backend.modules.identity.infrastructure.models  # noqa: F401
    import rag_backend.modules.rag.infrastructure.models  # noqa: F401
    import rag_backend.modules.workspaces.infrastructure.models  # noqa: F401


@lru_cache
def get_engine() -> AsyncEngine:
    settings = get_settings()
    return create_async_engine(settings.database_url, future=True)


@lru_cache
def get_session_maker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    session_maker = get_session_maker()
    async with session_maker() as session:
        yield session


def reset_database_state() -> None:
    get_session_maker.cache_clear()
    get_engine.cache_clear()
