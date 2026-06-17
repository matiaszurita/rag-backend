import math
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from rag_backend.modules.rag.domain.entities import DocumentChunk, RetrievalSource, SimilarChunk
from rag_backend.modules.rag.infrastructure.models import DocumentChunkORM


def _metadata(model: DocumentChunkORM) -> dict[str, object]:
    return model.chunk_metadata or {}


def _cosine_score(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    score = dot / (left_norm * right_norm)
    return max(0.0, min(1.0, score))


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.12g}" for value in values) + "]"


def _keyword_score(content: str, query: str) -> float:
    content_lower = content.lower()
    content_tokens = {token.lower() for token in content.replace("_", " ").split()}
    query_tokens = [token.lower() for token in query.replace("_", " ").split() if token.strip()]
    if not content_tokens or not query_tokens:
        return 0.0
    matches = sum(
        1
        for token in query_tokens
        if token in content_tokens or token in content_lower
    )
    return matches / len(query_tokens)


class SqlAlchemyChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def replace_for_document(self, document_id: UUID, chunks: list[DocumentChunk]) -> None:
        await self.session.execute(
            sa.delete(DocumentChunkORM).where(DocumentChunkORM.document_id == document_id)
        )
        self.session.add_all(
            DocumentChunkORM(
                id=chunk.id,
                workspace_id=chunk.workspace_id,
                document_id=chunk.document_id,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                embedding=chunk.embedding,
                chunk_metadata=chunk.metadata,
            )
            for chunk in chunks
        )

    async def vector_search(
        self,
        *,
        workspace_id: UUID,
        query_embedding: list[float],
        limit: int,
    ) -> list[SimilarChunk]:
        bind = self.session.get_bind()
        if bind.dialect.name == "postgresql":
            result = await self.session.execute(
                sa.text(
                    """
                    SELECT id, document_id, content, metadata,
                           1 - (embedding <=> CAST(:embedding AS vector)) AS score
                    FROM document_chunks
                    WHERE workspace_id = :workspace_id
                    ORDER BY embedding <=> CAST(:embedding AS vector)
                    LIMIT :limit
                    """
                ),
                {
                    "workspace_id": workspace_id,
                    "embedding": _vector_literal(query_embedding),
                    "limit": limit,
                },
            )
            return [
                SimilarChunk(
                    chunk_id=row["id"],
                    document_id=row["document_id"],
                    content=row["content"],
                    score=max(0.0, min(1.0, float(row["score"]))),
                    vector_score=max(0.0, min(1.0, float(row["score"]))),
                    retrieval_source=RetrievalSource.VECTOR,
                    metadata=row["metadata"] or {},
                )
                for row in result.mappings().all()
            ]

        result = await self.session.execute(
            sa.select(DocumentChunkORM).where(DocumentChunkORM.workspace_id == workspace_id)
        )
        ranked = sorted(
            (
                (
                    _cosine_score(model.embedding, query_embedding),
                    model,
                )
                for model in result.scalars().all()
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        return [
            SimilarChunk(
                chunk_id=model.id,
                document_id=model.document_id,
                content=model.content,
                score=score,
                vector_score=score,
                retrieval_source=RetrievalSource.VECTOR,
                metadata=_metadata(model),
            )
            for score, model in ranked[:limit]
        ]

    async def keyword_search(
        self,
        *,
        workspace_id: UUID,
        query: str,
        limit: int,
    ) -> list[SimilarChunk]:
        bind = self.session.get_bind()
        if bind.dialect.name == "postgresql":
            result = await self.session.execute(
                sa.text(
                    """
                    SELECT id, document_id, content, metadata,
                           ts_rank_cd(
                               to_tsvector('simple', content),
                               plainto_tsquery('simple', :query)
                           ) AS score
                    FROM document_chunks
                    WHERE workspace_id = :workspace_id
                      AND to_tsvector('simple', content) @@ plainto_tsquery('simple', :query)
                    ORDER BY score DESC
                    LIMIT :limit
                    """
                ),
                {
                    "workspace_id": workspace_id,
                    "query": query,
                    "limit": limit,
                },
            )
            return [
                SimilarChunk(
                    chunk_id=row["id"],
                    document_id=row["document_id"],
                    content=row["content"],
                    score=max(0.0, float(row["score"])),
                    keyword_score=max(0.0, float(row["score"])),
                    retrieval_source=RetrievalSource.KEYWORD,
                    metadata=row["metadata"] or {},
                )
                for row in result.mappings().all()
            ]

        result = await self.session.execute(
            sa.select(DocumentChunkORM).where(DocumentChunkORM.workspace_id == workspace_id)
        )
        ranked = sorted(
            (
                (_keyword_score(model.content, query), model)
                for model in result.scalars().all()
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        return [
            SimilarChunk(
                chunk_id=model.id,
                document_id=model.document_id,
                content=model.content,
                score=score,
                keyword_score=score,
                retrieval_source=RetrievalSource.KEYWORD,
                metadata=_metadata(model),
            )
            for score, model in ranked[:limit]
            if score > 0
        ]

    async def commit(self) -> None:
        await self.session.commit()
