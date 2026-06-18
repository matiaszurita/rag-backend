from uuid import uuid4

import pytest

from rag_backend.modules.rag.application.services import RetrievalService
from rag_backend.modules.rag.domain.entities import RetrievalMode, RetrievalSource, SimilarChunk
from rag_backend.modules.rag.infrastructure.fakes import NoOpReranker, ScoreBasedFakeReranker


class TrackingEmbeddingProvider:
    def __init__(self) -> None:
        self.query_calls: list[str] = []

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [1.0]


class FakeChunkRepository:
    def __init__(
        self,
        *,
        vector_results: list[SimilarChunk] | None = None,
        keyword_results: list[SimilarChunk] | None = None,
    ) -> None:
        self.vector_results = vector_results or []
        self.keyword_results = keyword_results or []
        self.vector_calls = 0
        self.keyword_calls = 0
        self.vector_limits: list[int] = []
        self.keyword_limits: list[int] = []

    async def replace_for_document(self, document_id, chunks) -> None:  # noqa: ANN001
        return None

    async def vector_search(self, *, workspace_id, query_embedding, limit):  # noqa: ANN001
        self.vector_calls += 1
        self.vector_limits.append(limit)
        return self.vector_results[:limit]

    async def keyword_search(self, *, workspace_id, query, limit):  # noqa: ANN001
        self.keyword_calls += 1
        self.keyword_limits.append(limit)
        return self.keyword_results[:limit]

    async def commit(self) -> None:
        return None


def _chunk(
    *,
    chunk_id=None,  # noqa: ANN001
    score: float,
    source: RetrievalSource,
) -> SimilarChunk:
    return SimilarChunk(
        chunk_id=chunk_id or uuid4(),
        document_id=uuid4(),
        content=f"{source.value} content",
        score=score,
        vector_score=score if source == RetrievalSource.VECTOR else None,
        keyword_score=score if source == RetrievalSource.KEYWORD else None,
        retrieval_source=source,
        metadata={"source": "test.txt"},
    )


def _service(
    embeddings: TrackingEmbeddingProvider,
    chunks: FakeChunkRepository,
    reranker=None,  # noqa: ANN001
    reranking_enabled: bool = False,
) -> RetrievalService:
    return RetrievalService(
        embeddings=embeddings,
        chunks=chunks,
        reranker=reranker or NoOpReranker(),
        default_mode=RetrievalMode.HYBRID,
        vector_weight=0.65,
        keyword_weight=0.35,
        vector_candidates=20,
        keyword_candidates=20,
        reranking_enabled=reranking_enabled,
        reranking_provider="noop",
        reranking_candidates=20,
    )


@pytest.mark.asyncio
async def test_vector_mode_embeds_and_calls_vector_search_only() -> None:
    embeddings = TrackingEmbeddingProvider()
    chunks = FakeChunkRepository(vector_results=[_chunk(score=0.8, source=RetrievalSource.VECTOR)])

    results, metadata = await _service(embeddings, chunks).retrieve(
        workspace_id=uuid4(),
        query="alpha",
        top_k=5,
        retrieval_mode=RetrievalMode.VECTOR,
    )

    assert embeddings.query_calls == ["alpha"]
    assert chunks.vector_calls == 1
    assert chunks.keyword_calls == 0
    assert results[0].vector_score == 0.8
    assert results[0].retrieval_source == RetrievalSource.VECTOR
    assert metadata.retrieval_mode == RetrievalMode.VECTOR


@pytest.mark.asyncio
async def test_keyword_mode_does_not_embed_and_calls_keyword_search_only() -> None:
    embeddings = TrackingEmbeddingProvider()
    chunks = FakeChunkRepository(
        keyword_results=[_chunk(score=0.4, source=RetrievalSource.KEYWORD)]
    )

    results, metadata = await _service(embeddings, chunks).retrieve(
        workspace_id=uuid4(),
        query="JWT_SECRET_KEY",
        top_k=5,
        retrieval_mode=RetrievalMode.KEYWORD,
    )

    assert embeddings.query_calls == []
    assert chunks.vector_calls == 0
    assert chunks.keyword_calls == 1
    assert results[0].keyword_score == 0.4
    assert results[0].retrieval_source == RetrievalSource.KEYWORD
    assert metadata.retrieval_mode == RetrievalMode.KEYWORD


@pytest.mark.asyncio
async def test_hybrid_mode_fuses_and_deduplicates_results() -> None:
    shared_chunk_id = uuid4()
    vector_only = _chunk(score=0.9, source=RetrievalSource.VECTOR)
    shared_vector = _chunk(
        chunk_id=shared_chunk_id,
        score=0.7,
        source=RetrievalSource.VECTOR,
    )
    shared_keyword = _chunk(
        chunk_id=shared_chunk_id,
        score=0.5,
        source=RetrievalSource.KEYWORD,
    )
    keyword_only = _chunk(score=0.6, source=RetrievalSource.KEYWORD)
    embeddings = TrackingEmbeddingProvider()
    chunks = FakeChunkRepository(
        vector_results=[vector_only, shared_vector],
        keyword_results=[shared_keyword, keyword_only],
    )

    results, metadata = await _service(embeddings, chunks).retrieve(
        workspace_id=uuid4(),
        query="alpha",
        top_k=5,
        retrieval_mode=RetrievalMode.HYBRID,
    )

    assert embeddings.query_calls == ["alpha"]
    assert chunks.vector_calls == 1
    assert chunks.keyword_calls == 1
    assert len(results) == 3
    shared = next(result for result in results if result.chunk_id == shared_chunk_id)
    assert shared.retrieval_source == RetrievalSource.HYBRID
    assert shared.vector_score == 0.7
    assert shared.keyword_score == 0.5
    assert metadata.deduplicated_results == 3
    assert metadata.fusion_algorithm == "weighted_rrf"
    assert results[0].score >= results[-1].score


@pytest.mark.asyncio
async def test_reranking_disabled_preserves_current_order() -> None:
    first = _chunk(score=0.9, source=RetrievalSource.VECTOR)
    second = _chunk(score=0.8, source=RetrievalSource.VECTOR)
    embeddings = TrackingEmbeddingProvider()
    chunks = FakeChunkRepository(vector_results=[first, second])

    results, metadata = await _service(embeddings, chunks).retrieve(
        workspace_id=uuid4(),
        query="alpha",
        top_k=2,
        retrieval_mode=RetrievalMode.VECTOR,
    )

    assert [result.chunk_id for result in results] == [first.chunk_id, second.chunk_id]
    assert metadata.reranking_enabled is False
    assert metadata.reranking_applied is False
    assert metadata.candidates_before_rerank == 0


@pytest.mark.asyncio
async def test_reranking_enabled_reorders_with_fake_reranker() -> None:
    first = _chunk(score=0.9, source=RetrievalSource.VECTOR)
    second = _chunk(score=0.8, source=RetrievalSource.VECTOR)
    reranker = ScoreBasedFakeReranker({first.chunk_id: 0.1, second.chunk_id: 0.95})
    embeddings = TrackingEmbeddingProvider()
    chunks = FakeChunkRepository(vector_results=[first, second])

    results, metadata = await _service(
        embeddings,
        chunks,
        reranker=reranker,
    ).retrieve(
        workspace_id=uuid4(),
        query="alpha",
        top_k=2,
        retrieval_mode=RetrievalMode.VECTOR,
        reranking_enabled=True,
    )

    assert [result.chunk_id for result in results] == [second.chunk_id, first.chunk_id]
    assert results[0].rerank_score == 0.95
    assert results[0].original_rank == 2
    assert results[0].reranked_rank == 1
    assert metadata.reranking_enabled is True
    assert metadata.reranking_applied is True
    assert metadata.candidates_before_rerank == 2
    assert reranker.calls


@pytest.mark.asyncio
async def test_vector_reranking_receives_expanded_candidates() -> None:
    chunks = FakeChunkRepository(
        vector_results=[
            _chunk(score=1.0 - index / 100, source=RetrievalSource.VECTOR)
            for index in range(25)
        ]
    )
    embeddings = TrackingEmbeddingProvider()

    results, metadata = await _service(embeddings, chunks).retrieve(
        workspace_id=uuid4(),
        query="alpha",
        top_k=5,
        retrieval_mode=RetrievalMode.VECTOR,
        reranking_enabled=True,
    )

    assert chunks.vector_limits == [20]
    assert len(results) == 5
    assert metadata.vector_candidates == 20
    assert metadata.candidates_before_rerank == 20


@pytest.mark.asyncio
async def test_keyword_reranking_does_not_embed_query() -> None:
    chunks = FakeChunkRepository(
        keyword_results=[
            _chunk(score=1.0 - index / 100, source=RetrievalSource.KEYWORD)
            for index in range(25)
        ]
    )
    embeddings = TrackingEmbeddingProvider()

    results, metadata = await _service(embeddings, chunks).retrieve(
        workspace_id=uuid4(),
        query="JWT_SECRET_KEY",
        top_k=5,
        retrieval_mode=RetrievalMode.KEYWORD,
        reranking_enabled=True,
    )

    assert embeddings.query_calls == []
    assert chunks.keyword_limits == [20]
    assert len(results) == 5
    assert metadata.keyword_candidates == 20
    assert metadata.candidates_before_rerank == 20


@pytest.mark.asyncio
async def test_hybrid_reranking_happens_after_fusion_and_deduplication() -> None:
    shared_chunk_id = uuid4()
    vector_only = _chunk(score=0.9, source=RetrievalSource.VECTOR)
    shared_vector = _chunk(chunk_id=shared_chunk_id, score=0.8, source=RetrievalSource.VECTOR)
    shared_keyword = _chunk(chunk_id=shared_chunk_id, score=0.7, source=RetrievalSource.KEYWORD)
    keyword_only = _chunk(score=0.6, source=RetrievalSource.KEYWORD)
    reranker = ScoreBasedFakeReranker({keyword_only.chunk_id: 0.99})
    embeddings = TrackingEmbeddingProvider()
    chunks = FakeChunkRepository(
        vector_results=[vector_only, shared_vector],
        keyword_results=[shared_keyword, keyword_only],
    )

    results, metadata = await _service(
        embeddings,
        chunks,
        reranker=reranker,
    ).retrieve(
        workspace_id=uuid4(),
        query="alpha",
        top_k=2,
        retrieval_mode=RetrievalMode.HYBRID,
        reranking_enabled=True,
    )

    assert results[0].chunk_id == keyword_only.chunk_id
    assert metadata.deduplicated_results == 3
    assert metadata.candidates_before_rerank == 3
    assert len(reranker.calls[0]["candidates"]) == 3
