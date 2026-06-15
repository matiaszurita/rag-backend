from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(slots=True)
class CreateWorkspaceCommand:
    owner_id: UUID
    name: str
    description: str | None


@dataclass(slots=True)
class WorkspaceDTO:
    id: UUID
    owner_id: UUID
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime
