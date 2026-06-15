from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient

from rag_backend.api.app import create_app
from rag_backend.core.config import get_settings
from rag_backend.core.database import (
    Base,
    get_engine,
    get_session_maker,
    import_model_modules,
    reset_database_state,
)


@pytest.fixture(autouse=True)
def test_environment(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    root = tmp_path / "storage"
    root.mkdir(parents=True, exist_ok=True)
    db_file = root / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_file}")
    monkeypatch.setenv("ALEMBIC_DATABASE_URL", f"sqlite+aiosqlite:///{db_file}")
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(root / "documents"))
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "10")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-with-sufficient-length")
    monkeypatch.setenv("ENVIRONMENT", "test")
    get_settings.cache_clear()
    reset_database_state()


@pytest.fixture
async def app() -> AsyncGenerator:
    import_model_modules()
    engine = get_engine()
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    application = create_app()
    yield application

    session_maker = get_session_maker()
    async with session_maker() as session:
        await session.close()

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
    await engine.dispose()
    get_settings.cache_clear()
    reset_database_state()


@pytest.fixture
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as async_client:
        yield async_client
