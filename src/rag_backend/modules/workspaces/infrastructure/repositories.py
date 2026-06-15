from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from rag_backend.modules.workspaces.domain.entities import Workspace
from rag_backend.modules.workspaces.infrastructure.models import WorkspaceORM


def _to_domain(model: WorkspaceORM) -> Workspace:
    return Workspace(
        id=model.id,
        owner_id=model.owner_id,
        name=model.name,
        description=model.description,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


class SqlAlchemyWorkspaceRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, *, owner_id: UUID, name: str, description: str | None) -> Workspace:
        model = WorkspaceORM(owner_id=owner_id, name=name, description=description)
        self.session.add(model)
        await self.session.flush()
        await self.session.refresh(model)
        return _to_domain(model)

    async def list_by_owner(self, owner_id: UUID) -> list[Workspace]:
        result = await self.session.execute(
            sa.select(WorkspaceORM)
            .where(WorkspaceORM.owner_id == owner_id)
            .order_by(WorkspaceORM.created_at.desc())
        )
        return [_to_domain(model) for model in result.scalars().all()]

    async def get_by_id_for_owner(self, workspace_id: UUID, owner_id: UUID) -> Workspace | None:
        result = await self.session.execute(
            sa.select(WorkspaceORM).where(
                WorkspaceORM.id == workspace_id,
                WorkspaceORM.owner_id == owner_id,
            )
        )
        model = result.scalar_one_or_none()
        return _to_domain(model) if model else None

    async def commit(self) -> None:
        await self.session.commit()
