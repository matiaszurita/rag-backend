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


@dataclass(slots=True)
class QueryRagCommand:
    owner_id: UUID
    workspace_id: UUID
    question: str
    top_k: int | None


@dataclass(slots=True)
class RagSourceDTO:
    chunk_id: UUID
    document_id: UUID
    filename: str
    score: float
    content_preview: str


@dataclass(slots=True)
class QueryRagMetadataDTO:
    context_chunks_used: int
    top_k: int
    llm_model: str
    context_char_count: int


@dataclass(slots=True)
class QueryRagResult:
    question: str
    answer: str
    sources: list[RagSourceDTO]
    metadata: QueryRagMetadataDTO
