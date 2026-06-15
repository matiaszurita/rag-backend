from fastapi import APIRouter

from rag_backend.modules.documents.interfaces.router import router as documents_router
from rag_backend.modules.identity.interfaces.router import router as identity_router
from rag_backend.modules.workspaces.interfaces.router import router as workspaces_router

api_router = APIRouter()
api_router.include_router(identity_router)
api_router.include_router(workspaces_router)
api_router.include_router(documents_router)
