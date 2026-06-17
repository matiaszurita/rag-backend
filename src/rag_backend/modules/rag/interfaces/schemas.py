from uuid import UUID

from pydantic import BaseModel, Field


class IndexDocumentResponse(BaseModel):
    document_id: UUID
    chunks_indexed: int
    status: str


class SearchRequest(BaseModel):
    query: str
    top_k: int | None = Field(default=None, ge=1)


class SearchResultItem(BaseModel):
    chunk_id: UUID
    document_id: UUID
    content: str
    score: float
    metadata: dict[str, object]


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = Field(default=None, ge=1)


class RagSource(BaseModel):
    chunk_id: UUID
    document_id: UUID
    filename: str
    score: float
    content_preview: str


class QueryMetadata(BaseModel):
    context_chunks_used: int
    top_k: int
    llm_model: str
    context_char_count: int


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[RagSource]
    metadata: QueryMetadata
