from rag_backend.modules.rag.domain.entities import SimilarChunk

RAG_SYSTEM_PROMPT = """Eres un asistente de recuperación de conocimiento técnico.
Responde únicamente usando el contexto proporcionado.
Si el contexto no contiene información suficiente para responder, dilo claramente.
No inventes información.
Responde de forma clara, breve y técnica.
Cuando sea útil, menciona que la respuesta se basa en los documentos recuperados."""


class RagPromptBuilder:
    def build_user_prompt(self, *, question: str, chunks: list[SimilarChunk]) -> str:
        context_blocks = []
        for index, chunk in enumerate(chunks, start=1):
            filename = filename_from_metadata(chunk.metadata)
            context_blocks.append(
                "\n".join(
                    [
                        f"[Fuente {index}]",
                        f"chunk_id: {chunk.chunk_id}",
                        f"document_id: {chunk.document_id}",
                        f"filename: {filename}",
                        f"score: {chunk.score:.4f}",
                        f"retrieval_source: {chunk.retrieval_source.value}",
                        "content:",
                        chunk.content,
                    ]
                )
            )
        return "\n\n".join(
            [
                "Contexto recuperado:",
                "\n\n".join(context_blocks),
                "Pregunta:",
                question,
                "Respuesta:",
            ]
        )


def filename_from_metadata(metadata: dict[str, object]) -> str:
    source = metadata.get("source")
    if isinstance(source, str) and source.strip():
        return source
    return "unknown"
