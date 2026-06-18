from uuid import uuid4

from rag_backend.modules.rag.application.services import RagPromptBuilder
from rag_backend.modules.rag.domain.entities import RetrievalSource, SimilarChunk


def test_rag_prompt_builder_builds_context_prompt() -> None:
    chunk = SimilarChunk(
        chunk_id=uuid4(),
        document_id=uuid4(),
        content="JWT_SECRET_KEY configures signing secrets",
        score=0.875,
        metadata={"source": "security.md"},
        retrieval_source=RetrievalSource.KEYWORD,
        keyword_score=0.875,
    )

    prompt = RagPromptBuilder().build_user_prompt(
        question="What configures signing secrets?",
        chunks=[chunk],
    )

    assert "Contexto recuperado:" in prompt
    assert "[Fuente 1]" in prompt
    assert f"chunk_id: {chunk.chunk_id}" in prompt
    assert f"document_id: {chunk.document_id}" in prompt
    assert "filename: security.md" in prompt
    assert "score: 0.8750" in prompt
    assert "retrieval_source: keyword" in prompt
    assert "JWT_SECRET_KEY configures signing secrets" in prompt
    assert "Pregunta:\n\nWhat configures signing secrets?" in prompt
    assert prompt.endswith("Respuesta:")
