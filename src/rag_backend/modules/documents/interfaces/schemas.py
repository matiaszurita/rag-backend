from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from rag_backend.modules.documents.domain.entities import DocumentStatus


class DocumentResponse(BaseModel):
    id: UUID
    workspace_id: UUID
    original_filename: str
    storage_path: str
    content_type: str | None
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None
