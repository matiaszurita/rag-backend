from rag_backend.core.config import Settings
from rag_backend.core.security import create_access_token, hash_password, verify_password


class PasswordHasherAdapter:
    def hash(self, password: str) -> str:
        return hash_password(password)

    def verify(self, password: str, password_hash: str) -> bool:
        return verify_password(password, password_hash)


class JwtTokenService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def create_access_token(self, subject: str) -> str:
        return create_access_token(
            subject,
            secret_key=self.settings.jwt_secret_key,
            algorithm=self.settings.jwt_algorithm,
            expires_in_minutes=self.settings.jwt_access_token_expire_minutes,
        )
