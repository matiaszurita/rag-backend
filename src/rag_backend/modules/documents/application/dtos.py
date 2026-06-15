from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from rag_backend.modules.documents.domain.entities import DocumentStatus


@dataclass(slots=True)
class UploadDocumentCommand:
    owner_id: UUID
    workspace_id: UUID
    original_filename: str
    content_type: str | None
    content: bytes


@dataclass(slots=True)
class DocumentDTO:
    id: UUID
    workspace_id: UUID
    original_filename: str
    storage_path: str
    content_type: str | None
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None
