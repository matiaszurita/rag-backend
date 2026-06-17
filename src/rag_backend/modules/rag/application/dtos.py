from dataclasses import dataclass, field
from uuid import UUID


@dataclass(slots=True)
class IndexDocumentCommand:
    owner_id: UUID
    workspace_id: UUID
    document_id: UUID


@dataclass(slots=True)
class IndexDocumentResult:
    document_id: UUID
    chunks_indexed: int
    status: str


@dataclass(slots=True)
class SearchSimilarChunksCommand:
    owner_id: UUID
    workspace_id: UUID
    query: str
    top_k: int | None


@dataclass(slots=True)
class SimilarChunkDTO:
    chunk_id: UUID
    document_id: UUID
    content: str
    score: float
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class SearchSimilarChunksResult:
    query: str
    results: list[SimilarChunkDTO]
