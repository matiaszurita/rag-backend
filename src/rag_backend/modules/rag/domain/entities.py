from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from uuid import UUID


class RetrievalMode(StrEnum):
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class RetrievalSource(StrEnum):
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


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
    vector_score: float | None = None
    keyword_score: float | None = None
    retrieval_source: RetrievalSource = RetrievalSource.VECTOR
