from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from rag_backend.modules.identity.domain.entities import User
from rag_backend.modules.identity.infrastructure.models import UserORM


def _to_domain(model: UserORM) -> User:
    return User(
        id=model.id,
        email=model.email,
        password_hash=model.password_hash,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


class SqlAlchemyUserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_email(self, email: str) -> User | None:
        result = await self.session.execute(sa.select(UserORM).where(UserORM.email == email))
        model = result.scalar_one_or_none()
        return _to_domain(model) if model else None

    async def get_by_id(self, user_id: UUID | str) -> User | None:
        result = await self.session.execute(sa.select(UserORM).where(UserORM.id == user_id))
        model = result.scalar_one_or_none()
        return _to_domain(model) if model else None

    async def add(self, *, email: str, password_hash: str) -> User:
        model = UserORM(email=email, password_hash=password_hash)
        self.session.add(model)
        await self.session.flush()
        await self.session.refresh(model)
        return _to_domain(model)

    async def commit(self) -> None:
        await self.session.commit()
