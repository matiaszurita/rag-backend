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


class ConversationMessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


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
    rerank_score: float | None = None
    original_rank: int | None = None
    reranked_rank: int | None = None
    retrieval_source: RetrievalSource = RetrievalSource.VECTOR


@dataclass(slots=True)
class Conversation:
    id: UUID
    workspace_id: UUID
    title: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(slots=True)
class ConversationMessage:
    id: UUID
    conversation_id: UUID
    message_index: int
    role: ConversationMessageRole
    content: str
    sources: list[dict[str, object]] | None = None
    metadata: dict[str, object] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
