from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class IndexDocumentResponse(BaseModel):
    document_id: UUID
    chunks_indexed: int
    status: str


class SearchRequest(BaseModel):
    query: str
    top_k: int | None = Field(default=None, ge=1)
    retrieval_mode: Literal["vector", "keyword", "hybrid"] | None = None
    reranking_enabled: bool | None = None


class SearchResultItem(BaseModel):
    chunk_id: UUID
    document_id: UUID
    content: str
    score: float
    vector_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None
    original_rank: int | None = None
    reranked_rank: int | None = None
    retrieval_source: Literal["vector", "keyword", "hybrid"]
    metadata: dict[str, object]


class RetrievalMetadata(BaseModel):
    retrieval_mode: Literal["vector", "keyword", "hybrid"]
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


class SearchResponse(BaseModel):
    query: str
    retrieval_mode: Literal["vector", "keyword", "hybrid"]
    results: list[SearchResultItem]
    metadata: RetrievalMetadata


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = Field(default=None, ge=1)
    retrieval_mode: Literal["vector", "keyword", "hybrid"] | None = None
    reranking_enabled: bool | None = None


class RagSource(BaseModel):
    chunk_id: UUID
    document_id: UUID
    filename: str
    score: float
    vector_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None
    original_rank: int | None = None
    reranked_rank: int | None = None
    retrieval_source: Literal["vector", "keyword", "hybrid"]
    content_preview: str


class QueryMetadata(BaseModel):
    context_chunks_used: int
    top_k: int
    llm_model: str
    context_char_count: int
    retrieval_mode: Literal["vector", "keyword", "hybrid"]
    fusion_algorithm: str | None = None
    reranking_enabled: bool = False
    reranking_provider: str = "noop"
    reranking_applied: bool = False


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[RagSource]
    metadata: QueryMetadata
