from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID


@dataclass(slots=True)
class DocumentChunk:
    id: UUID
    workspace_id: UUID
    document_id: UUID
    content: str
    chunk_index: int
    embedding: list[float]
    metadata: dict[str, object] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(slots=True)
class SimilarChunk:
    chunk_id: UUID
    document_id: UUID
    content: str
    score: float
    metadata: dict[str, object]
