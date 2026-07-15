import math
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from rag_backend.modules.rag.domain.entities import (
    Conversation,
    ConversationMessage,
    ConversationMessageRole,
    DocumentChunk,
    RetrievalSource,
    SimilarChunk,
)
from rag_backend.modules.rag.infrastructure.models import (
    ConversationMessageORM,
    ConversationORM,
    DocumentChunkORM,
)
from rag_backend.modules.workspaces.infrastructure.models import WorkspaceORM


def _metadata(model: DocumentChunkORM) -> dict[str, object]:
    return model.chunk_metadata or {}


def _to_conversation(model: ConversationORM) -> Conversation:
    return Conversation(
        id=model.id,
        workspace_id=model.workspace_id,
        title=model.title,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def _to_message(model: ConversationMessageORM) -> ConversationMessage:
    return ConversationMessage(
        id=model.id,
        conversation_id=model.conversation_id,
        message_index=model.message_index,
        role=model.role,
        content=model.content,
        sources=model.sources,
        metadata=model.message_metadata,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


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


class SqlAlchemyConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def workspace_exists_for_owner(self, *, owner_id: UUID, workspace_id: UUID) -> bool:
        result = await self.session.execute(
            sa.select(WorkspaceORM.id).where(
                WorkspaceORM.id == workspace_id,
                WorkspaceORM.owner_id == owner_id,
            )
        )
        return result.scalar_one_or_none() is not None

    async def create_conversation(
        self,
        *,
        workspace_id: UUID,
        title: str | None,
    ) -> Conversation:
        model = ConversationORM(workspace_id=workspace_id, title=title)
        self.session.add(model)
        await self.session.flush()
        await self.session.refresh(model)
        return _to_conversation(model)

    async def list_conversations_for_owner(
        self,
        *,
        owner_id: UUID,
        workspace_id: UUID,
    ) -> list[Conversation]:
        result = await self.session.execute(
            sa.select(ConversationORM)
            .join(WorkspaceORM, WorkspaceORM.id == ConversationORM.workspace_id)
            .where(
                WorkspaceORM.owner_id == owner_id,
                ConversationORM.workspace_id == workspace_id,
            )
            .order_by(ConversationORM.updated_at.desc(), ConversationORM.created_at.desc())
        )
        return [_to_conversation(model) for model in result.scalars().all()]

    async def get_conversation_for_owner(
        self,
        *,
        owner_id: UUID,
        workspace_id: UUID,
        conversation_id: UUID,
    ) -> Conversation | None:
        result = await self.session.execute(
            sa.select(ConversationORM)
            .join(WorkspaceORM, WorkspaceORM.id == ConversationORM.workspace_id)
            .where(
                WorkspaceORM.owner_id == owner_id,
                ConversationORM.workspace_id == workspace_id,
                ConversationORM.id == conversation_id,
            )
        )
        model = result.scalar_one_or_none()
        return _to_conversation(model) if model else None

    async def list_messages(self, *, conversation_id: UUID) -> list[ConversationMessage]:
        result = await self.session.execute(
            sa.select(ConversationMessageORM)
            .where(ConversationMessageORM.conversation_id == conversation_id)
            .order_by(ConversationMessageORM.message_index.asc())
        )
        return [_to_message(model) for model in result.scalars().all()]

    async def list_recent_messages(
        self,
        *,
        conversation_id: UUID,
        limit: int,
    ) -> list[ConversationMessage]:
        result = await self.session.execute(
            sa.select(ConversationMessageORM)
            .where(ConversationMessageORM.conversation_id == conversation_id)
            .order_by(ConversationMessageORM.message_index.desc())
            .limit(limit)
        )
        models = list(result.scalars().all())
        models.reverse()
        return [_to_message(model) for model in models]

    async def append_turn(
        self,
        *,
        conversation_id: UUID,
        user_content: str,
        assistant_content: str,
        assistant_sources: list[dict[str, object]],
        assistant_metadata: dict[str, object],
        title: str | None = None,
    ) -> tuple[ConversationMessage, ConversationMessage]:
        result = await self.session.execute(
            sa.select(sa.func.max(ConversationMessageORM.message_index)).where(
                ConversationMessageORM.conversation_id == conversation_id
            )
        )
        current_max = result.scalar_one()
        user_index = (current_max or 0) + 1
        assistant_index = user_index + 1

        if title is not None:
            await self.session.execute(
                sa.update(ConversationORM)
                .where(
                    ConversationORM.id == conversation_id,
                    ConversationORM.title.is_(None),
                )
                .values(title=title)
            )

        await self.session.execute(
            sa.update(ConversationORM)
            .where(ConversationORM.id == conversation_id)
            .values(updated_at=sa.func.now())
        )

        user_model = ConversationMessageORM(
            id=uuid4(),
            conversation_id=conversation_id,
            message_index=user_index,
            role=ConversationMessageRole.USER,
            content=user_content,
            sources=None,
            message_metadata=None,
        )
        assistant_model = ConversationMessageORM(
            id=uuid4(),
            conversation_id=conversation_id,
            message_index=assistant_index,
            role=ConversationMessageRole.ASSISTANT,
            content=assistant_content,
            sources=assistant_sources,
            message_metadata=assistant_metadata,
        )
        self.session.add_all([user_model, assistant_model])
        await self.session.flush()
        await self.session.refresh(user_model)
        await self.session.refresh(assistant_model)
        return _to_message(user_model), _to_message(assistant_model)

    async def commit(self) -> None:
        await self.session.commit()
