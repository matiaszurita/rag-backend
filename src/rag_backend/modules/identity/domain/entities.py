from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(slots=True)
class User:
    id: UUID
    email: str
    password_hash: str
    created_at: datetime
    updated_at: datetime
