from typing import Any

from rag_backend.core.errors import BadRequestError
from rag_backend.modules.rag.application.dtos import RetrievalMetadataDTO
from rag_backend.modules.rag.application.ports import ChunkRepositoryPort, EmbeddingProviderPort
from rag_backend.modules.rag.domain.entities import RetrievalMode, RetrievalSource, SimilarChunk

FUSION_ALGORITHM = "weighted_rrf"
RRF_K = 60


class RetrievalService:
    def __init__(
        self,
        *,
        embeddings: EmbeddingProviderPort,
        chunks: ChunkRepositoryPort,
        default_mode: RetrievalMode,
        vector_weight: float,
        keyword_weight: float,
        vector_candidates: int,
        keyword_candidates: int,
    ) -> None:
        self.embeddings = embeddings
        self.chunks = chunks
        self.default_mode = default_mode
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.vector_candidates = vector_candidates
        self.keyword_candidates = keyword_candidates

    async def retrieve(
        self,
        *,
        workspace_id: Any,
        query: str,
        top_k: int,
        retrieval_mode: RetrievalMode | None,
    ) -> tuple[list[SimilarChunk], RetrievalMetadataDTO]:
        mode = retrieval_mode or self.default_mode
        if mode == RetrievalMode.VECTOR:
            return await self._retrieve_vector(
                workspace_id=workspace_id,
                query=query,
                top_k=top_k,
            )
        if mode == RetrievalMode.KEYWORD:
            return await self._retrieve_keyword(
                workspace_id=workspace_id,
                query=query,
                top_k=top_k,
            )
        if mode == RetrievalMode.HYBRID:
            return await self._retrieve_hybrid(
                workspace_id=workspace_id,
                query=query,
                top_k=top_k,
            )
        raise BadRequestError("Unsupported retrieval mode", code="invalid_retrieval_mode")

    async def _retrieve_vector(
        self,
        *,
        workspace_id: Any,
        query: str,
        top_k: int,
    ) -> tuple[list[SimilarChunk], RetrievalMetadataDTO]:
        query_embedding = await self.embeddings.embed_query(query)
        results = await self.chunks.vector_search(
            workspace_id=workspace_id,
            query_embedding=query_embedding,
            limit=top_k,
        )
        return results, RetrievalMetadataDTO(
            retrieval_mode=RetrievalMode.VECTOR,
            vector_candidates=top_k,
            keyword_candidates=0,
            vector_results=len(results),
            keyword_results=0,
            deduplicated_results=len(results),
            final_results=len(results),
        )

    async def _retrieve_keyword(
        self,
        *,
        workspace_id: Any,
        query: str,
        top_k: int,
    ) -> tuple[list[SimilarChunk], RetrievalMetadataDTO]:
        results = await self.chunks.keyword_search(
            workspace_id=workspace_id,
            query=query,
            limit=top_k,
        )
        return results, RetrievalMetadataDTO(
            retrieval_mode=RetrievalMode.KEYWORD,
            vector_candidates=0,
            keyword_candidates=top_k,
            vector_results=0,
            keyword_results=len(results),
            deduplicated_results=len(results),
            final_results=len(results),
        )

    async def _retrieve_hybrid(
        self,
        *,
        workspace_id: Any,
        query: str,
        top_k: int,
    ) -> tuple[list[SimilarChunk], RetrievalMetadataDTO]:
        query_embedding = await self.embeddings.embed_query(query)
        vector_limit = max(top_k, self.vector_candidates)
        keyword_limit = max(top_k, self.keyword_candidates)
        vector_results = await self.chunks.vector_search(
            workspace_id=workspace_id,
            query_embedding=query_embedding,
            limit=vector_limit,
        )
        keyword_results = await self.chunks.keyword_search(
            workspace_id=workspace_id,
            query=query,
            limit=keyword_limit,
        )
        all_results = [*vector_results, *keyword_results]
        deduplicated_results = len({chunk.chunk_id for chunk in all_results})
        fused = self._fuse(vector_results, keyword_results)[:top_k]
        return fused, RetrievalMetadataDTO(
            retrieval_mode=RetrievalMode.HYBRID,
            vector_candidates=vector_limit,
            keyword_candidates=keyword_limit,
            vector_results=len(vector_results),
            keyword_results=len(keyword_results),
            deduplicated_results=deduplicated_results,
            final_results=len(fused),
            fusion_algorithm=FUSION_ALGORITHM,
        )

    def _fuse(
        self,
        vector_results: list[SimilarChunk],
        keyword_results: list[SimilarChunk],
    ) -> list[SimilarChunk]:
        by_id: dict[object, SimilarChunk] = {}
        fused_scores: dict[object, float] = {}

        for rank, chunk in enumerate(vector_results, start=1):
            by_id[chunk.chunk_id] = SimilarChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.content,
                score=0.0,
                vector_score=chunk.vector_score if chunk.vector_score is not None else chunk.score,
                keyword_score=None,
                retrieval_source=RetrievalSource.VECTOR,
                metadata=chunk.metadata,
            )
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + (
                self.vector_weight / (RRF_K + rank)
            )

        for rank, chunk in enumerate(keyword_results, start=1):
            existing = by_id.get(chunk.chunk_id)
            if existing is None:
                by_id[chunk.chunk_id] = SimilarChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    score=0.0,
                    vector_score=None,
                    keyword_score=(
                        chunk.keyword_score if chunk.keyword_score is not None else chunk.score
                    ),
                    retrieval_source=RetrievalSource.KEYWORD,
                    metadata=chunk.metadata,
                )
            else:
                existing.keyword_score = (
                    chunk.keyword_score if chunk.keyword_score is not None else chunk.score
                )
                existing.retrieval_source = RetrievalSource.HYBRID
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + (
                self.keyword_weight / (RRF_K + rank)
            )

        max_score = max(fused_scores.values(), default=1.0)
        for chunk_id, chunk in by_id.items():
            chunk.score = fused_scores[chunk_id] / max_score if max_score else 0.0
            if chunk.retrieval_source != RetrievalSource.HYBRID:
                chunk.retrieval_source = (
                    RetrievalSource.VECTOR
                    if chunk.vector_score is not None
                    else RetrievalSource.KEYWORD
                )
        return sorted(by_id.values(), key=lambda chunk: chunk.score, reverse=True)
