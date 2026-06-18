from typing import Any

from rag_backend.core.errors import BadRequestError
from rag_backend.modules.rag.application.dtos import RetrievalMetadataDTO
from rag_backend.modules.rag.application.ports import (
    ChunkRepositoryPort,
    EmbeddingProviderPort,
    RerankerPort,
)
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
        reranker: RerankerPort,
        reranking_enabled: bool,
        reranking_provider: str,
        reranking_candidates: int,
    ) -> None:
        self.embeddings = embeddings
        self.chunks = chunks
        self.reranker = reranker
        self.default_mode = default_mode
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.vector_candidates = vector_candidates
        self.keyword_candidates = keyword_candidates
        self.reranking_enabled = reranking_enabled
        self.reranking_provider = reranking_provider
        self.reranking_candidates = reranking_candidates

    async def retrieve(
        self,
        *,
        workspace_id: Any,
        query: str,
        top_k: int,
        retrieval_mode: RetrievalMode | None,
        reranking_enabled: bool | None = None,
    ) -> tuple[list[SimilarChunk], RetrievalMetadataDTO]:
        mode = retrieval_mode or self.default_mode
        use_reranking = self._use_reranking(reranking_enabled)
        if mode == RetrievalMode.VECTOR:
            return await self._retrieve_vector(
                workspace_id=workspace_id,
                query=query,
                top_k=top_k,
                reranking_enabled=use_reranking,
            )
        if mode == RetrievalMode.KEYWORD:
            return await self._retrieve_keyword(
                workspace_id=workspace_id,
                query=query,
                top_k=top_k,
                reranking_enabled=use_reranking,
            )
        if mode == RetrievalMode.HYBRID:
            return await self._retrieve_hybrid(
                workspace_id=workspace_id,
                query=query,
                top_k=top_k,
                reranking_enabled=use_reranking,
            )
        raise BadRequestError("Unsupported retrieval mode", code="invalid_retrieval_mode")

    def _use_reranking(self, override: bool | None) -> bool:
        return self.reranking_enabled if override is None else override

    async def _retrieve_vector(
        self,
        *,
        workspace_id: Any,
        query: str,
        top_k: int,
        reranking_enabled: bool,
    ) -> tuple[list[SimilarChunk], RetrievalMetadataDTO]:
        query_embedding = await self.embeddings.embed_query(query)
        limit = self._candidate_limit(top_k, reranking_enabled)
        results = await self.chunks.vector_search(
            workspace_id=workspace_id,
            query_embedding=query_embedding,
            limit=limit,
        )
        final_results, reranking_applied, candidates_before_rerank = await self._rerank_if_enabled(
            query=query,
            candidates=results,
            top_k=top_k,
            reranking_enabled=reranking_enabled,
        )
        return final_results, RetrievalMetadataDTO(
            retrieval_mode=RetrievalMode.VECTOR,
            vector_candidates=limit,
            keyword_candidates=0,
            vector_results=len(results),
            keyword_results=0,
            deduplicated_results=len(results),
            final_results=len(final_results),
            reranking_enabled=reranking_enabled,
            reranking_provider=self.reranking_provider,
            reranking_applied=reranking_applied,
            reranking_candidates=self.reranking_candidates if reranking_enabled else 0,
            candidates_before_rerank=candidates_before_rerank,
        )

    async def _retrieve_keyword(
        self,
        *,
        workspace_id: Any,
        query: str,
        top_k: int,
        reranking_enabled: bool,
    ) -> tuple[list[SimilarChunk], RetrievalMetadataDTO]:
        limit = self._candidate_limit(top_k, reranking_enabled)
        results = await self.chunks.keyword_search(
            workspace_id=workspace_id,
            query=query,
            limit=limit,
        )
        final_results, reranking_applied, candidates_before_rerank = await self._rerank_if_enabled(
            query=query,
            candidates=results,
            top_k=top_k,
            reranking_enabled=reranking_enabled,
        )
        return final_results, RetrievalMetadataDTO(
            retrieval_mode=RetrievalMode.KEYWORD,
            vector_candidates=0,
            keyword_candidates=limit,
            vector_results=0,
            keyword_results=len(results),
            deduplicated_results=len(results),
            final_results=len(final_results),
            reranking_enabled=reranking_enabled,
            reranking_provider=self.reranking_provider,
            reranking_applied=reranking_applied,
            reranking_candidates=self.reranking_candidates if reranking_enabled else 0,
            candidates_before_rerank=candidates_before_rerank,
        )

    async def _retrieve_hybrid(
        self,
        *,
        workspace_id: Any,
        query: str,
        top_k: int,
        reranking_enabled: bool,
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
        fused = self._fuse(vector_results, keyword_results)
        candidates = (
            fused[: self.reranking_candidates]
            if reranking_enabled
            else fused[:top_k]
        )
        final_results, reranking_applied, candidates_before_rerank = await self._rerank_if_enabled(
            query=query,
            candidates=candidates,
            top_k=top_k,
            reranking_enabled=reranking_enabled,
        )
        return final_results, RetrievalMetadataDTO(
            retrieval_mode=RetrievalMode.HYBRID,
            vector_candidates=vector_limit,
            keyword_candidates=keyword_limit,
            vector_results=len(vector_results),
            keyword_results=len(keyword_results),
            deduplicated_results=deduplicated_results,
            final_results=len(final_results),
            fusion_algorithm=FUSION_ALGORITHM,
            reranking_enabled=reranking_enabled,
            reranking_provider=self.reranking_provider,
            reranking_applied=reranking_applied,
            reranking_candidates=self.reranking_candidates if reranking_enabled else 0,
            candidates_before_rerank=candidates_before_rerank,
        )

    def _candidate_limit(self, top_k: int, reranking_enabled: bool) -> int:
        if not reranking_enabled:
            return top_k
        return max(top_k, self.reranking_candidates)

    async def _rerank_if_enabled(
        self,
        *,
        query: str,
        candidates: list[SimilarChunk],
        top_k: int,
        reranking_enabled: bool,
    ) -> tuple[list[SimilarChunk], bool, int]:
        if not reranking_enabled:
            return candidates[:top_k], False, 0
        if not candidates:
            return [], False, 0
        reranked = await self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )
        return reranked[:top_k], True, len(candidates)

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
