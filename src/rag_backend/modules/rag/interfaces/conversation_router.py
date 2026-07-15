from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from rag_backend.core.config import get_settings
from rag_backend.core.database import get_db_session
from rag_backend.modules.identity.domain.entities import User
from rag_backend.modules.identity.interfaces.dependencies import get_current_user
from rag_backend.modules.rag.application.dtos import (
    CreateConversationCommand,
    SendConversationMessageCommand,
)
from rag_backend.modules.rag.application.ports import (
    EmbeddingProviderPort,
    LLMProviderPort,
    RerankerPort,
)
from rag_backend.modules.rag.application.services import ChatRagService, ConversationService
from rag_backend.modules.rag.infrastructure.repositories import (
    SqlAlchemyChunkRepository,
    SqlAlchemyConversationRepository,
)
from rag_backend.modules.rag.interfaces.router import (
    _retrieval_mode,
    _retrieval_service,
    get_embedding_provider,
    get_llm_provider,
    get_reranker,
)
from rag_backend.modules.rag.interfaces.schemas import (
    ConversationDetailResponse,
    ConversationMessageResponse,
    ConversationResponse,
    CreateConversationRequest,
    RagSource,
    SendConversationMessageRequest,
    SendConversationMessageResponse,
)

router = APIRouter(tags=["rag"])


def _conversation_service(session: AsyncSession) -> ConversationService:
    settings = get_settings()
    return ConversationService(
        SqlAlchemyConversationRepository(session),
        title_max_length=settings.rag_chat_title_max_length,
    )


def _chat_service(
    *,
    session: AsyncSession,
    embeddings: EmbeddingProviderPort,
    llm: LLMProviderPort,
    reranker: RerankerPort,
) -> ChatRagService:
    settings = get_settings()
    return ChatRagService(
        conversations=SqlAlchemyConversationRepository(session),
        retrieval=_retrieval_service(
            embeddings=embeddings,
            chunks=SqlAlchemyChunkRepository(session),
            reranker=reranker,
        ),
        llm=llm,
        max_context_chunks=settings.rag_answer_max_context_chunks,
        min_relevance_score=settings.rag_min_relevance_score,
        llm_model=settings.gemini_llm_model,
        history_messages=settings.rag_chat_history_messages,
        title_max_length=settings.rag_chat_title_max_length,
    )


def _message_response(payload) -> ConversationMessageResponse:  # noqa: ANN001
    sources = None
    if payload.sources is not None:
        sources = [RagSource.model_validate(source) for source in payload.sources]
    return ConversationMessageResponse(
        id=payload.id,
        conversation_id=payload.conversation_id,
        message_index=payload.message_index,
        role=payload.role.value,
        content=payload.content,
        sources=sources,
        metadata=payload.metadata,
        created_at=payload.created_at,
        updated_at=payload.updated_at,
    )


@router.post(
    "/workspaces/{workspace_id}/conversations",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    workspace_id: UUID,
    payload: CreateConversationRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> ConversationResponse:
    conversation = await _conversation_service(session).create(
        CreateConversationCommand(
            owner_id=current_user.id,
            workspace_id=workspace_id,
            title=payload.title,
        )
    )
    return ConversationResponse.model_validate(conversation, from_attributes=True)


@router.get(
    "/workspaces/{workspace_id}/conversations",
    response_model=list[ConversationResponse],
)
async def list_conversations(
    workspace_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> list[ConversationResponse]:
    conversations = await _conversation_service(session).list_for_owner(
        owner_id=current_user.id,
        workspace_id=workspace_id,
    )
    return [
        ConversationResponse.model_validate(item, from_attributes=True)
        for item in conversations
    ]


@router.get(
    "/workspaces/{workspace_id}/conversations/{conversation_id}",
    response_model=ConversationDetailResponse,
)
async def get_conversation(
    workspace_id: UUID,
    conversation_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> ConversationDetailResponse:
    conversation = await _conversation_service(session).get_for_owner(
        owner_id=current_user.id,
        workspace_id=workspace_id,
        conversation_id=conversation_id,
    )
    return ConversationDetailResponse(
        id=conversation.id,
        workspace_id=conversation.workspace_id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[_message_response(message) for message in conversation.messages],
    )


@router.post(
    "/workspaces/{workspace_id}/conversations/{conversation_id}/messages",
    response_model=SendConversationMessageResponse,
)
async def send_message(
    workspace_id: UUID,
    conversation_id: UUID,
    request: SendConversationMessageRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    embeddings=Depends(get_embedding_provider),  # noqa: ANN001,B008
    llm=Depends(get_llm_provider),  # noqa: ANN001,B008
    reranker=Depends(get_reranker),  # noqa: ANN001,B008
) -> SendConversationMessageResponse:
    result = await _chat_service(
        session=session,
        embeddings=embeddings,
        llm=llm,
        reranker=reranker,
    ).send_message(
        SendConversationMessageCommand(
            owner_id=current_user.id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            message=request.message,
            top_k=request.top_k,
            retrieval_mode=_retrieval_mode(request.retrieval_mode),
            reranking_enabled=request.reranking_enabled,
        )
    )
    return SendConversationMessageResponse(
        conversation_id=result.conversation_id,
        user_message=_message_response(result.user_message),
        assistant_message=_message_response(result.assistant_message),
    )
