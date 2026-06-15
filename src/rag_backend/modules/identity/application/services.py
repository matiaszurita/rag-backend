from rag_backend.core.errors import AuthenticationError, ConflictError
from rag_backend.modules.identity.application.dtos import (
    AuthTokenDTO,
    LoginUserCommand,
    RegisterUserCommand,
    UserDTO,
)
from rag_backend.modules.identity.application.ports import (
    PasswordHasherPort,
    TokenServicePort,
    UserRepositoryPort,
)
from rag_backend.modules.identity.domain.entities import User


def _to_dto(user: User) -> UserDTO:
    return UserDTO(
        id=user.id,
        email=user.email,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


class IdentityService:
    def __init__(
        self,
        users: UserRepositoryPort,
        password_hasher: PasswordHasherPort,
        token_service: TokenServicePort,
    ) -> None:
        self.users = users
        self.password_hasher = password_hasher
        self.token_service = token_service

    async def register(self, command: RegisterUserCommand) -> UserDTO:
        existing_user = await self.users.get_by_email(command.email)
        if existing_user is not None:
            raise ConflictError("A user with this email already exists", code="user_already_exists")

        user = await self.users.add(
            email=command.email,
            password_hash=self.password_hasher.hash(command.password),
        )
        await self.users.commit()
        return _to_dto(user)

    async def login(self, command: LoginUserCommand) -> AuthTokenDTO:
        user = await self.users.get_by_email(command.email)
        if user is None or not self.password_hasher.verify(command.password, user.password_hash):
            raise AuthenticationError("Invalid credentials", code="invalid_credentials")

        return AuthTokenDTO(access_token=self.token_service.create_access_token(str(user.id)))

    async def me(self, user_id: str) -> UserDTO:
        user = await self.users.get_by_id(user_id)
        if user is None:
            raise AuthenticationError("Authenticated user was not found", code="user_not_found")
        return _to_dto(user)
