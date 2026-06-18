import pytest

from rag_backend.core.config import Settings, get_settings


def test_settings_reject_insecure_secret_outside_development(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("JWT_SECRET_KEY", "change-me")
    get_settings.cache_clear()

    with pytest.raises(
        ValueError,
        match="JWT_SECRET_KEY cannot be 'change-me' outside development",
    ):
        Settings()


def test_settings_allow_change_me_in_development(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("JWT_SECRET_KEY", "change-me")
    get_settings.cache_clear()

    settings = Settings()

    assert settings.jwt_secret_key == "change-me"


def test_settings_parse_cors_allowed_origins_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS",
        '["http://localhost:5173", "http://127.0.0.1:5173"]',
    )

    settings = Settings()

    assert settings.cors_allowed_origins == [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


def test_settings_default_reranking_is_disabled_noop() -> None:
    settings = Settings()

    assert settings.rag_reranking_enabled is False
    assert settings.rag_reranking_provider == "noop"
    assert settings.rag_reranking_candidates == 20


def test_settings_reject_invalid_reranking_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_RERANKING_PROVIDER", "gemini")

    with pytest.raises(ValueError, match="RAG_RERANKING_PROVIDER must be noop"):
        Settings()


def test_settings_reject_invalid_reranking_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_RERANKING_CANDIDATES", "0")

    with pytest.raises(ValueError, match="RAG_RERANKING_CANDIDATES must be at least 1"):
        Settings()
