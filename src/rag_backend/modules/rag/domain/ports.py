from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(slots=True)
class ChatResponse:
    content: str


class ChatModelPort(Protocol):
    async def complete(self, messages: list[ChatMessage]) -> ChatResponse: ...


class EmbeddingModelPort(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorStorePort(Protocol):
    async def upsert(self, vectors: list[list[float]], metadata: list[dict[str, str]]) -> None: ...

    async def similarity_search(
        self,
        query_vector: list[float],
        limit: int,
    ) -> list[dict[str, str]]: ...


class TextExtractorPort(Protocol):
    async def extract(self, content: bytes, content_type: str | None) -> str: ...
