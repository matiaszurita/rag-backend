import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from rag_backend.core.database import Base, TimestampMixin, UUIDPrimaryKeyMixin


class UserORM(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(sa.String(320), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(sa.String(255), nullable=False)
