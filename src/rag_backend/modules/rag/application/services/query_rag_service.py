from rag_backend.core.errors import AppError, BadRequestError, NotFoundError
from rag_backend.modules.rag.application.dtos import (
    QueryRagCommand,
    QueryRagMetadataDTO,
    QueryRagResult,
    RagSourceDTO,
)
from rag_backend.modules.rag.application.ports import DocumentAccessPort, LLMProviderPort
from rag_backend.modules.rag.application.services.prompt_builder import (
    RAG_SYSTEM_PROMPT,
    RagPromptBuilder,
    filename_from_metadata,
)
from rag_backend.modules.rag.application.services.retrieval_service import RetrievalService
from rag_backend.modules.rag.domain.entities import RetrievalMode, SimilarChunk

INSUFFICIENT_CONTEXT_ANSWER = (
    "No encontré información suficiente en los documentos indexados para responder esta pregunta."
)


class QueryRagService:
    def __init__(
        self,
        *,
        documents: DocumentAccessPort,
        retrieval: RetrievalService,
        llm: LLMProviderPort,
        max_context_chunks: int,
        min_relevance_score: float,
        llm_model: str,
        prompt_builder: RagPromptBuilder | None = None,
    ) -> None:
        self.documents = documents
        self.retrieval = retrieval
        self.llm = llm
        self.max_context_chunks = max_context_chunks
        self.min_relevance_score = min_relevance_score
        self.llm_model = llm_model
        self.prompt_builder = prompt_builder or RagPromptBuilder()

    async def query(self, command: QueryRagCommand) -> QueryRagResult:
        question = command.question.strip()
        if not question:
            raise BadRequestError("Question cannot be empty", code="empty_question")

        if self.max_context_chunks < 1:
            raise BadRequestError(
                "RAG answer context chunk limit must be at least 1",
                code="invalid_rag_answer_context_limit",
            )

        if command.top_k is not None and command.top_k < 1:
            raise BadRequestError("top_k must be at least 1", code="invalid_top_k")

        workspace_exists = await self.documents.workspace_exists_for_owner(
            owner_id=command.owner_id,
            workspace_id=command.workspace_id,
        )
        if not workspace_exists:
            raise NotFoundError("Workspace not found", code="workspace_not_found")

        top_k = min(command.top_k or self.max_context_chunks, self.max_context_chunks)
        results, retrieval_metadata = await self.retrieval.retrieve(
            workspace_id=command.workspace_id,
            query=question,
            top_k=top_k,
            retrieval_mode=command.retrieval_mode,
        )
        if retrieval_metadata.retrieval_mode == RetrievalMode.VECTOR:
            relevant_chunks = [
                result for result in results if result.score >= self.min_relevance_score
            ][:top_k]
        else:
            relevant_chunks = results[:top_k]

        if not relevant_chunks:
            return QueryRagResult(
                question=question,
                answer=INSUFFICIENT_CONTEXT_ANSWER,
                sources=[],
                metadata=QueryRagMetadataDTO(
                    context_chunks_used=0,
                    top_k=top_k,
                    llm_model=self.llm_model,
                    context_char_count=0,
                    retrieval_mode=retrieval_metadata.retrieval_mode,
                    fusion_algorithm=retrieval_metadata.fusion_algorithm,
                ),
            )

        user_prompt = self.prompt_builder.build_user_prompt(
            question=question,
            chunks=relevant_chunks,
        )
        try:
            answer = await self.llm.generate_answer(
                system_prompt=RAG_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except AppError:
            raise
        except Exception as error:
            raise BadRequestError(
                "RAG answer generation failed",
                code="rag_answer_failed",
            ) from error

        return QueryRagResult(
            question=question,
            answer=answer,
            sources=[self._source_from_chunk(result) for result in relevant_chunks],
            metadata=QueryRagMetadataDTO(
                context_chunks_used=len(relevant_chunks),
                top_k=top_k,
                llm_model=self.llm_model,
                context_char_count=sum(len(result.content) for result in relevant_chunks),
                retrieval_mode=retrieval_metadata.retrieval_mode,
                fusion_algorithm=retrieval_metadata.fusion_algorithm,
            ),
        )

    def _source_from_chunk(self, chunk: SimilarChunk) -> RagSourceDTO:
        return RagSourceDTO(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            filename=filename_from_metadata(chunk.metadata),
            score=chunk.score,
            vector_score=chunk.vector_score,
            keyword_score=chunk.keyword_score,
            retrieval_source=chunk.retrieval_source,
            content_preview=self._preview(chunk.content),
        )

    def _preview(self, content: str) -> str:
        normalized = " ".join(content.split())
        return normalized[:240]
