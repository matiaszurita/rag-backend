from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from rag_backend.core.config import get_settings
from rag_backend.core.database import get_db_session
from rag_backend.modules.documents.application.dtos import UploadDocumentCommand
from rag_backend.modules.documents.application.services import DocumentService, DocumentUploadPolicy
from rag_backend.modules.documents.infrastructure.repositories import SqlAlchemyDocumentRepository
from rag_backend.modules.documents.infrastructure.storage import LocalDocumentStorage
from rag_backend.modules.documents.interfaces.schemas import DocumentResponse
from rag_backend.modules.identity.domain.entities import User
from rag_backend.modules.identity.interfaces.dependencies import get_current_user
from rag_backend.modules.workspaces.infrastructure.repositories import SqlAlchemyWorkspaceRepository

router = APIRouter(tags=["documents"])


def build_document_service(session: AsyncSession) -> DocumentService:
    settings = get_settings()
    return DocumentService(
        documents=SqlAlchemyDocumentRepository(session),
        workspaces=SqlAlchemyWorkspaceRepository(session),
        storage=LocalDocumentStorage(settings.local_storage_path),
        upload_policy=DocumentUploadPolicy(
            allowed_extensions=frozenset({".md", ".txt", ".pdf"}),
            max_size_bytes=settings.max_upload_size_bytes,
        ),
    )


@router.post(
    "/workspaces/{workspace_id}/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    workspace_id: UUID,
    file: Annotated[UploadFile, File(...)],
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> DocumentResponse:
    service = build_document_service(session)
    document = await service.upload(
        UploadDocumentCommand(
            owner_id=current_user.id,
            workspace_id=workspace_id,
            original_filename=file.filename or "document.bin",
            content_type=file.content_type,
            content=await file.read(),
        )
    )
    return DocumentResponse.model_validate(document, from_attributes=True)


@router.get("/workspaces/{workspace_id}/documents", response_model=list[DocumentResponse])
async def list_documents(
    workspace_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> list[DocumentResponse]:
    service = build_document_service(session)
    documents = await service.list_for_workspace(
        owner_id=current_user.id,
        workspace_id=workspace_id,
    )
    return [
        DocumentResponse.model_validate(document, from_attributes=True)
        for document in documents
    ]


@router.get("/workspaces/{workspace_id}/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    workspace_id: UUID,
    document_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> DocumentResponse:
    service = build_document_service(session)
    document = await service.get(
        owner_id=current_user.id,
        workspace_id=workspace_id,
        document_id=document_id,
    )
    return DocumentResponse.model_validate(document, from_attributes=True)


@router.delete(
    "/workspaces/{workspace_id}/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_document(
    workspace_id: UUID,
    document_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> None:
    service = build_document_service(session)
    await service.delete(
        owner_id=current_user.id,
        workspace_id=workspace_id,
        document_id=document_id,
    )
