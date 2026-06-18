import hashlib
from uuid import UUID

from rag_backend.modules.rag.domain.entities import SimilarChunk


class FakeEmbeddingProvider:
    def __init__(self, *, dimensions: int = 8, fail: bool = False) -> None:
        self.dimensions = dimensions
        self.fail = fail

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("fake embedding failure")
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        if self.fail:
            raise RuntimeError("fake embedding failure")
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        tokens = [token.lower() for token in text.split()]
        vector = [0.0 for _ in range(self.dimensions)]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[digest[0] % self.dimensions] += 1.0
        if any(vector):
            return vector
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [byte / 255 for byte in digest[: self.dimensions]]


class FakeLLMProvider:
    def __init__(self, *, answer: str = "fake grounded answer", fail: bool = False) -> None:
        self.answer = answer
        self.fail = fail
        self.calls: list[dict[str, str]] = []

    async def generate_answer(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        if self.fail:
            raise RuntimeError("fake llm failure with secret-token")
        return self.answer


class NoOpReranker:
    async def rerank(
        self,
        *,
        query: str,
        candidates: list[SimilarChunk],
        top_k: int,
    ) -> list[SimilarChunk]:
        return candidates[:top_k]


class ScoreBasedFakeReranker:
    def __init__(self, scores: dict[UUID, float]) -> None:
        self.scores = scores
        self.calls: list[dict[str, object]] = []

    async def rerank(
        self,
        *,
        query: str,
        candidates: list[SimilarChunk],
        top_k: int,
    ) -> list[SimilarChunk]:
        self.calls.append({"query": query, "candidates": candidates, "top_k": top_k})
        ranked = sorted(
            enumerate(candidates, start=1),
            key=lambda item: self.scores.get(item[1].chunk_id, item[1].score),
            reverse=True,
        )
        results = [chunk for _, chunk in ranked[:top_k]]
        for reranked_rank, chunk in enumerate(results, start=1):
            chunk.original_rank = next(
                original_rank
                for original_rank, candidate in enumerate(candidates, start=1)
                if candidate.chunk_id == chunk.chunk_id
            )
            chunk.reranked_rank = reranked_rank
            chunk.rerank_score = self.scores.get(chunk.chunk_id, chunk.score)
        return results
