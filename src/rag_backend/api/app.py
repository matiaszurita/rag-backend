from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_backend.api.router import api_router
from rag_backend.core.config import get_settings
from rag_backend.core.errors import register_exception_handlers
from rag_backend.core.logging import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title=settings.app_name)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/health", tags=["health"])
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok", "service": settings.app_name}

    register_exception_handlers(app)
    return app
