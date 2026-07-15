from typing import Protocol
from uuid import UUID

from rag_backend.modules.documents.domain.entities import Document, DocumentStatus
from rag_backend.modules.rag.domain.entities import (
    Conversation,
    ConversationMessage,
    DocumentChunk,
    SimilarChunk,
)


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


class RerankerPort(Protocol):
    async def rerank(
        self,
        *,
        query: str,
        candidates: list[SimilarChunk],
        top_k: int,
    ) -> list[SimilarChunk]: ...


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


class ConversationRepositoryPort(Protocol):
    async def workspace_exists_for_owner(self, *, owner_id: UUID, workspace_id: UUID) -> bool: ...

    async def create_conversation(
        self,
        *,
        workspace_id: UUID,
        title: str | None,
    ) -> Conversation: ...

    async def list_conversations_for_owner(
        self,
        *,
        owner_id: UUID,
        workspace_id: UUID,
    ) -> list[Conversation]: ...

    async def get_conversation_for_owner(
        self,
        *,
        owner_id: UUID,
        workspace_id: UUID,
        conversation_id: UUID,
    ) -> Conversation | None: ...

    async def list_messages(self, *, conversation_id: UUID) -> list[ConversationMessage]: ...

    async def list_recent_messages(
        self,
        *,
        conversation_id: UUID,
        limit: int,
    ) -> list[ConversationMessage]: ...

    async def append_turn(
        self,
        *,
        conversation_id: UUID,
        user_content: str,
        assistant_content: str,
        assistant_sources: list[dict[str, object]],
        assistant_metadata: dict[str, object],
        title: str | None = None,
    ) -> tuple[ConversationMessage, ConversationMessage]: ...

    async def commit(self) -> None: ...
