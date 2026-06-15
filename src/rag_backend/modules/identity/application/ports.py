from typing import Protocol
from uuid import UUID

from rag_backend.modules.identity.domain.entities import User


class UserRepositoryPort(Protocol):
    async def get_by_email(self, email: str) -> User | None: ...

    async def get_by_id(self, user_id: UUID) -> User | None: ...

    async def add(self, *, email: str, password_hash: str) -> User: ...

    async def commit(self) -> None: ...


class PasswordHasherPort(Protocol):
    def hash(self, password: str) -> str: ...

    def verify(self, password: str, password_hash: str) -> bool: ...


class TokenServicePort(Protocol):
    def create_access_token(self, subject: str) -> str: ...
