from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID


class DocumentStatus(StrEnum):
    UPLOADED = "uploaded"
    INDEXING = "indexing"
    INDEXED = "indexed"
    INDEX_FAILED = "index_failed"


@dataclass(slots=True)
class Document:
    id: UUID
    workspace_id: UUID
    original_filename: str
    storage_path: str
    content_type: str | None
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None
