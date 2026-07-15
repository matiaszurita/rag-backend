from rag_backend.modules.rag.domain.entities import ConversationMessage, SimilarChunk


class ChatPromptBuilder:
    def build_contextualized_query(
        self,
        *,
        question: str,
        history_messages: list[ConversationMessage],
    ) -> str:
        if not history_messages:
            return question

        history_lines = [
            f"{message.role.value}: {' '.join(message.content.split())}"
            for message in history_messages
        ]
        return "\n".join(
            [
                "Historial reciente:",
                *history_lines,
                "Pregunta actual:",
                question,
            ]
        )

    def build_user_prompt(
        self,
        *,
        question: str,
        history_messages: list[ConversationMessage],
        chunks: list[SimilarChunk],
    ) -> str:
        history_section = self._history_section(history_messages)
        context_blocks = []
        for index, chunk in enumerate(chunks, start=1):
            source = chunk.metadata.get("source")
            filename = source if isinstance(source, str) else "unknown"
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
                history_section,
                "Contexto recuperado:",
                "\n\n".join(context_blocks),
                "Pregunta actual:",
                question,
                "Respuesta:",
            ]
        )

    def _history_section(self, history_messages: list[ConversationMessage]) -> str:
        if not history_messages:
            return "Historial reciente:\n(sin historial previo)"

        lines = [
            f"[{message.role.value}] {' '.join(message.content.split())}"
            for message in history_messages
        ]
        return "\n".join(["Historial reciente:", *lines])
