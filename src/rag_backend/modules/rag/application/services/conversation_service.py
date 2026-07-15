from rag_backend.core.errors import NotFoundError
from rag_backend.modules.rag.application.dtos import (
    ConversationDTO,
    ConversationMessageDTO,
    CreateConversationCommand,
)
from rag_backend.modules.rag.application.ports import ConversationRepositoryPort
from rag_backend.modules.rag.domain.entities import Conversation, ConversationMessage


def _to_message_dto(message: ConversationMessage) -> ConversationMessageDTO:
    return ConversationMessageDTO(
        id=message.id,
        conversation_id=message.conversation_id,
        message_index=message.message_index,
        role=message.role,
        content=message.content,
        sources=message.sources,
        metadata=message.metadata,
        created_at=message.created_at,
        updated_at=message.updated_at,
    )


def _to_conversation_dto(
    conversation: Conversation,
    messages: list[ConversationMessage] | None = None,
) -> ConversationDTO:
    return ConversationDTO(
        id=conversation.id,
        workspace_id=conversation.workspace_id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[_to_message_dto(message) for message in messages or []],
    )


class ConversationService:
    def __init__(
        self,
        conversations: ConversationRepositoryPort,
        *,
        title_max_length: int,
    ) -> None:
        self.conversations = conversations
        self.title_max_length = title_max_length

    async def create(self, command: CreateConversationCommand) -> ConversationDTO:
        if not await self.conversations.workspace_exists_for_owner(
            owner_id=command.owner_id,
            workspace_id=command.workspace_id,
        ):
            raise NotFoundError("Workspace not found", code="workspace_not_found")

        conversation = await self.conversations.create_conversation(
            workspace_id=command.workspace_id,
            title=self._normalize_title(command.title),
        )
        await self.conversations.commit()
        return _to_conversation_dto(conversation)

    async def list_for_owner(self, *, owner_id, workspace_id) -> list[ConversationDTO]:  # noqa: ANN001
        if not await self.conversations.workspace_exists_for_owner(
            owner_id=owner_id,
            workspace_id=workspace_id,
        ):
            raise NotFoundError("Workspace not found", code="workspace_not_found")

        conversations = await self.conversations.list_conversations_for_owner(
            owner_id=owner_id,
            workspace_id=workspace_id,
        )
        return [_to_conversation_dto(conversation) for conversation in conversations]

    async def get_for_owner(self, *, owner_id, workspace_id, conversation_id) -> ConversationDTO:  # noqa: ANN001
        if not await self.conversations.workspace_exists_for_owner(
            owner_id=owner_id,
            workspace_id=workspace_id,
        ):
            raise NotFoundError("Workspace not found", code="workspace_not_found")

        conversation = await self.conversations.get_conversation_for_owner(
            owner_id=owner_id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
        )
        if conversation is None:
            raise NotFoundError("Conversation not found", code="conversation_not_found")

        messages = await self.conversations.list_messages(conversation_id=conversation.id)
        return _to_conversation_dto(conversation, messages)

    def _normalize_title(self, title: str | None) -> str | None:
        if title is None:
            return None
        normalized = " ".join(title.split())
        if not normalized:
            return None
        return normalized[: self.title_max_length]
