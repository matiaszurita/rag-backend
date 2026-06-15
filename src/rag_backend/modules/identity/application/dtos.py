from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(slots=True)
class RegisterUserCommand:
    email: str
    password: str


@dataclass(slots=True)
class LoginUserCommand:
    email: str
    password: str


@dataclass(slots=True)
class UserDTO:
    id: UUID
    email: str
    created_at: datetime
    updated_at: datetime


@dataclass(slots=True)
class AuthTokenDTO:
    access_token: str
    token_type: str = "bearer"
