from dataclasses import dataclass, field
from uuid import UUID

from rag_backend.modules.rag.domain.entities import RetrievalMode, RetrievalSource


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
    retrieval_mode: RetrievalMode | None = None
    reranking_enabled: bool | None = None


@dataclass(slots=True)
class SimilarChunkDTO:
    chunk_id: UUID
    document_id: UUID
    content: str
    score: float
    vector_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None
    original_rank: int | None = None
    reranked_rank: int | None = None
    retrieval_source: RetrievalSource = RetrievalSource.VECTOR
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalMetadataDTO:
    retrieval_mode: RetrievalMode
    vector_candidates: int
    keyword_candidates: int
    vector_results: int
    keyword_results: int
    deduplicated_results: int
    final_results: int
    fusion_algorithm: str | None = None
    reranking_enabled: bool = False
    reranking_provider: str = "noop"
    reranking_applied: bool = False
    reranking_candidates: int = 0
    candidates_before_rerank: int = 0


@dataclass(slots=True)
class SearchSimilarChunksResult:
    query: str
    retrieval_mode: RetrievalMode
    results: list[SimilarChunkDTO]
    metadata: RetrievalMetadataDTO


@dataclass(slots=True)
class QueryRagCommand:
    owner_id: UUID
    workspace_id: UUID
    question: str
    top_k: int | None
    retrieval_mode: RetrievalMode | None = None
    reranking_enabled: bool | None = None


@dataclass(slots=True)
class RagSourceDTO:
    chunk_id: UUID
    document_id: UUID
    filename: str
    score: float
    vector_score: float | None
    keyword_score: float | None
    retrieval_source: RetrievalSource
    content_preview: str
    rerank_score: float | None = None
    original_rank: int | None = None
    reranked_rank: int | None = None


@dataclass(slots=True)
class QueryRagMetadataDTO:
    context_chunks_used: int
    top_k: int
    llm_model: str
    context_char_count: int
    retrieval_mode: RetrievalMode
    fusion_algorithm: str | None = None
    reranking_enabled: bool = False
    reranking_provider: str = "noop"
    reranking_applied: bool = False


@dataclass(slots=True)
class QueryRagResult:
    question: str
    answer: str
    sources: list[RagSourceDTO]
    metadata: QueryRagMetadataDTO
