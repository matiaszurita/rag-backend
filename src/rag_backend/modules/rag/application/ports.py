from typing import Protocol
from uuid import UUID

from rag_backend.modules.documents.domain.entities import Document, DocumentStatus
from rag_backend.modules.rag.domain.entities import DocumentChunk, SimilarChunk


class TextExtractorPort(Protocol):
    async def extract(
        self,
        *,
        filename: str,
        content_type: str | None,
        content: bytes,
    ) -> str: ...


class TextSplitterPort(Protocol):
    def split(self, text: str) -> list[str]: ...


class EmbeddingProviderPort(Protocol):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class LLMProviderPort(Protocol):
    async def generate_answer(self, *, system_prompt: str, user_prompt: str) -> str: ...


class ChunkRepositoryPort(Protocol):
    async def replace_for_document(
        self,
        document_id: UUID,
        chunks: list[DocumentChunk],
    ) -> None: ...

    async def vector_search(
        self,
        *,
        workspace_id: UUID,
        query_embedding: list[float],
        limit: int,
    ) -> list[SimilarChunk]: ...

    async def keyword_search(
        self,
        *,
        workspace_id: UUID,
        query: str,
        limit: int,
    ) -> list[SimilarChunk]: ...

    async def commit(self) -> None: ...


class DocumentAccessPort(Protocol):
    async def get_document_for_owner(
        self,
        *,
        owner_id: UUID,
        workspace_id: UUID,
        document_id: UUID,
    ) -> Document | None: ...

    async def workspace_exists_for_owner(self, *, owner_id: UUID, workspace_id: UUID) -> bool: ...

    async def read_document_content(self, storage_path: str) -> bytes: ...

    async def update_document_status(
        self,
        document_id: UUID,
        status: DocumentStatus,
    ) -> None: ...

    async def commit(self) -> None: ...
