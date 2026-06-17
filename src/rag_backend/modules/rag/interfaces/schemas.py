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
