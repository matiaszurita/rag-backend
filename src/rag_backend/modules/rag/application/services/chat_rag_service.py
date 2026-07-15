from dataclasses import asdict

from rag_backend.core.errors import AppError, BadRequestError, NotFoundError
from rag_backend.modules.rag.application.dtos import (
    ConversationMessageDTO,
    QueryRagMetadataDTO,
    RagSourceDTO,
    SendConversationMessageCommand,
    SendConversationMessageResult,
)
from rag_backend.modules.rag.application.ports import ConversationRepositoryPort, LLMProviderPort
from rag_backend.modules.rag.application.services.chat_prompt_builder import ChatPromptBuilder
from rag_backend.modules.rag.application.services.conversation_service import _to_message_dto
from rag_backend.modules.rag.application.services.prompt_builder import (
    RAG_SYSTEM_PROMPT,
    filename_from_metadata,
)
from rag_backend.modules.rag.application.services.query_rag_service import (
    INSUFFICIENT_CONTEXT_ANSWER,
)
from rag_backend.modules.rag.application.services.retrieval_service import RetrievalService
from rag_backend.modules.rag.domain.entities import ConversationMessage, RetrievalMode, SimilarChunk


class ChatRagService:
    def __init__(
        self,
        *,
        conversations: ConversationRepositoryPort,
        retrieval: RetrievalService,
        llm: LLMProviderPort,
        max_context_chunks: int,
        min_relevance_score: float,
        llm_model: str,
        history_messages: int,
        title_max_length: int,
        prompt_builder: ChatPromptBuilder | None = None,
    ) -> None:
        self.conversations = conversations
        self.retrieval = retrieval
        self.llm = llm
        self.max_context_chunks = max_context_chunks
        self.min_relevance_score = min_relevance_score
        self.llm_model = llm_model
        self.history_messages = history_messages
        self.title_max_length = title_max_length
        self.prompt_builder = prompt_builder or ChatPromptBuilder()

    async def send_message(
        self,
        command: SendConversationMessageCommand,
    ) -> SendConversationMessageResult:
        message = command.message.strip()
        if not message:
            raise BadRequestError("Message cannot be empty", code="empty_message")

        if self.max_context_chunks < 1:
            raise BadRequestError(
                "RAG answer context chunk limit must be at least 1",
                code="invalid_rag_answer_context_limit",
            )

        if command.top_k is not None and command.top_k < 1:
            raise BadRequestError("top_k must be at least 1", code="invalid_top_k")

        if not await self.conversations.workspace_exists_for_owner(
            owner_id=command.owner_id,
            workspace_id=command.workspace_id,
        ):
            raise NotFoundError("Workspace not found", code="workspace_not_found")

        conversation = await self.conversations.get_conversation_for_owner(
            owner_id=command.owner_id,
            workspace_id=command.workspace_id,
            conversation_id=command.conversation_id,
        )
        if conversation is None:
            raise NotFoundError("Conversation not found", code="conversation_not_found")

        top_k = min(command.top_k or self.max_context_chunks, self.max_context_chunks)
        history = await self.conversations.list_recent_messages(
            conversation_id=command.conversation_id,
            limit=self.history_messages,
        )
        retrieval_query = self.prompt_builder.build_contextualized_query(
            question=message,
            history_messages=history,
        )
        results, retrieval_metadata = await self.retrieval.retrieve(
            workspace_id=command.workspace_id,
            query=retrieval_query,
            top_k=top_k,
            retrieval_mode=command.retrieval_mode,
            reranking_enabled=command.reranking_enabled,
        )
        if retrieval_metadata.retrieval_mode == RetrievalMode.VECTOR:
            relevant_chunks = [
                result for result in results if result.score >= self.min_relevance_score
            ][:top_k]
        else:
            relevant_chunks = results[:top_k]

        metadata = QueryRagMetadataDTO(
            context_chunks_used=len(relevant_chunks),
            top_k=top_k,
            llm_model=self.llm_model,
            context_char_count=sum(len(result.content) for result in relevant_chunks),
            retrieval_mode=retrieval_metadata.retrieval_mode,
            fusion_algorithm=retrieval_metadata.fusion_algorithm,
            reranking_enabled=retrieval_metadata.reranking_enabled,
            reranking_provider=retrieval_metadata.reranking_provider,
            reranking_applied=retrieval_metadata.reranking_applied,
        )
        stored_metadata = {
            **asdict(metadata),
            "retrieval_mode": metadata.retrieval_mode.value,
            "history_messages_used": len(history),
        }

        if not relevant_chunks:
            user_message, assistant_message = await self.conversations.append_turn(
                conversation_id=conversation.id,
                user_content=message,
                assistant_content=INSUFFICIENT_CONTEXT_ANSWER,
                assistant_sources=[],
                assistant_metadata=stored_metadata,
                title=self._derived_title(conversation.title, history, message),
            )
            await self.conversations.commit()
            return SendConversationMessageResult(
                conversation_id=conversation.id,
                user_message=_to_message_dto(user_message),
                assistant_message=_to_message_dto(assistant_message),
            )

        user_prompt = self.prompt_builder.build_user_prompt(
            question=message,
            history_messages=history,
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

        sources = [self._source_from_chunk(chunk) for chunk in relevant_chunks]
        user_message, assistant_message = await self.conversations.append_turn(
            conversation_id=conversation.id,
            user_content=message,
            assistant_content=answer,
            assistant_sources=[self._serialize_source(source) for source in sources],
            assistant_metadata=stored_metadata,
            title=self._derived_title(conversation.title, history, message),
        )
        await self.conversations.commit()
        return SendConversationMessageResult(
            conversation_id=conversation.id,
            user_message=_to_message_dto(user_message),
            assistant_message=self._assistant_message_dto(assistant_message, sources),
        )

    def _derived_title(
        self,
        title: str | None,
        history: list[ConversationMessage],
        message: str,
    ) -> str | None:
        if title is not None or history:
            return None
        return " ".join(message.split())[: self.title_max_length]

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
            rerank_score=chunk.rerank_score,
            original_rank=chunk.original_rank,
            reranked_rank=chunk.reranked_rank,
        )

    def _assistant_message_dto(
        self,
        message: ConversationMessage,
        sources: list[RagSourceDTO],
    ) -> ConversationMessageDTO:
        dto = _to_message_dto(message)
        dto.sources = [self._serialize_source(source) for source in sources]
        return dto

    def _serialize_source(self, source: RagSourceDTO) -> dict[str, object]:
        return {
            "chunk_id": str(source.chunk_id),
            "document_id": str(source.document_id),
            "filename": source.filename,
            "score": source.score,
            "vector_score": source.vector_score,
            "keyword_score": source.keyword_score,
            "rerank_score": source.rerank_score,
            "original_rank": source.original_rank,
            "reranked_rank": source.reranked_rank,
            "retrieval_source": source.retrieval_source.value,
            "content_preview": source.content_preview,
        }

    def _preview(self, content: str) -> str:
        normalized = " ".join(content.split())
        return normalized[:240]
