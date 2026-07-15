import io
from datetime import datetime
from uuid import uuid4

import pytest

from rag_backend.core.errors import NotFoundError
from rag_backend.modules.rag.application.dtos import (
    CreateConversationCommand,
    RetrievalMetadataDTO,
    SendConversationMessageCommand,
)
from rag_backend.modules.rag.application.services import (
    ChatPromptBuilder,
    ChatRagService,
    ConversationService,
)
from rag_backend.modules.rag.domain.entities import (
    Conversation,
    ConversationMessage,
    ConversationMessageRole,
    RetrievalMode,
    RetrievalSource,
    SimilarChunk,
)
from rag_backend.modules.rag.infrastructure.fakes import FakeLLMProvider


class FakeConversationRepository:
    def __init__(
        self,
        *,
        workspace_exists: bool = True,
        conversation: Conversation | None = None,
    ) -> None:
        self.workspace_exists = workspace_exists
        self.conversation = conversation
        self.conversations: list[Conversation] = []
        self.messages: list[ConversationMessage] = []
        self.recent_limit: int | None = None
        self.commits = 0

    async def workspace_exists_for_owner(self, *, owner_id, workspace_id) -> bool:  # noqa: ANN001
        return self.workspace_exists

    async def create_conversation(self, *, workspace_id, title):  # noqa: ANN001
        conversation = Conversation(
            id=uuid4(),
            workspace_id=workspace_id,
            title=title,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        self.conversations.append(conversation)
        return conversation

    async def list_conversations_for_owner(self, *, owner_id, workspace_id):  # noqa: ANN001
        return self.conversations

    async def get_conversation_for_owner(self, *, owner_id, workspace_id, conversation_id):  # noqa: ANN001
        return self.conversation

    async def list_messages(self, *, conversation_id):  # noqa: ANN001
        return self.messages

    async def list_recent_messages(self, *, conversation_id, limit):  # noqa: ANN001
        self.recent_limit = limit
        return self.messages[-limit:]

    async def append_turn(
        self,
        *,
        conversation_id,
        user_content,
        assistant_content,
        assistant_sources,
        assistant_metadata,
        title=None,
    ):
        user_message = ConversationMessage(
            id=uuid4(),
            conversation_id=conversation_id,
            message_index=len(self.messages) + 1,
            role=ConversationMessageRole.USER,
            content=user_content,
        )
        assistant_message = ConversationMessage(
            id=uuid4(),
            conversation_id=conversation_id,
            message_index=len(self.messages) + 2,
            role=ConversationMessageRole.ASSISTANT,
            content=assistant_content,
            sources=assistant_sources,
            metadata=assistant_metadata,
        )
        self.messages.extend([user_message, assistant_message])
        return user_message, assistant_message

    async def commit(self) -> None:
        self.commits += 1


class StaticRetrieval:
    def __init__(self, results: list[SimilarChunk], metadata: RetrievalMetadataDTO) -> None:
        self.results = results
        self.metadata = metadata
        self.calls: list[dict[str, object]] = []

    async def retrieve(self, **kwargs):  # noqa: ANN003
        self.calls.append(kwargs)
        return self.results, self.metadata


async def _create_user_workspace(client, email: str):
    await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "password123"},
    )
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "password123"},
    )
    token = login.json()["access_token"]
    workspace = await client.post(
        "/api/v1/workspaces",
        json={"name": f"Workspace {email}", "description": None},
        headers={"Authorization": f"Bearer {token}"},
    )
    return token, workspace.json()["id"]


async def _upload_document(client, token: str, workspace_id: str, filename: str, content: bytes):
    response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/documents",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": (filename, io.BytesIO(content), "text/plain")},
    )
    assert response.status_code == 201
    return response.json()


def test_chat_prompt_builder_includes_history_and_current_question() -> None:
    builder = ChatPromptBuilder()
    history = [
        ConversationMessage(
            id=uuid4(),
            conversation_id=uuid4(),
            message_index=1,
            role=ConversationMessageRole.USER,
            content="What stores vectors?",
        ),
        ConversationMessage(
            id=uuid4(),
            conversation_id=uuid4(),
            message_index=2,
            role=ConversationMessageRole.ASSISTANT,
            content="PostgreSQL with pgvector.",
        ),
    ]

    query = builder.build_contextualized_query(
        question="Why was that chosen?",
        history_messages=history,
    )

    assert "Historial reciente:" in query
    assert "assistant: PostgreSQL with pgvector." in query
    assert query.endswith("Why was that chosen?")


@pytest.mark.asyncio
async def test_conversation_service_rejects_other_users_workspace() -> None:
    service = ConversationService(
        FakeConversationRepository(workspace_exists=False),
        title_max_length=80,
    )

    with pytest.raises(NotFoundError) as error:
        await service.create(
            CreateConversationCommand(
                owner_id=uuid4(),
                workspace_id=uuid4(),
                title="Secret",
            )
        )

    assert error.value.code == "workspace_not_found"


@pytest.mark.asyncio
async def test_chat_rag_service_uses_bounded_recent_history() -> None:
    conversation = Conversation(
        id=uuid4(),
        workspace_id=uuid4(),
        title=None,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    repository = FakeConversationRepository(workspace_exists=True, conversation=conversation)
    repository.messages = [
        ConversationMessage(
            id=uuid4(),
            conversation_id=conversation.id,
            message_index=1,
            role=ConversationMessageRole.USER,
            content="What stores vectors?",
        ),
        ConversationMessage(
            id=uuid4(),
            conversation_id=conversation.id,
            message_index=2,
            role=ConversationMessageRole.ASSISTANT,
            content="PostgreSQL with pgvector.",
        ),
    ]
    retrieval = StaticRetrieval(
        [
            SimilarChunk(
                chunk_id=uuid4(),
                document_id=uuid4(),
                content="PostgreSQL with pgvector stores vectors.",
                score=0.8,
                metadata={"source": "rag.md"},
                retrieval_source=RetrievalSource.HYBRID,
            )
        ],
        RetrievalMetadataDTO(
            retrieval_mode=RetrievalMode.HYBRID,
            vector_candidates=1,
            keyword_candidates=1,
            vector_results=1,
            keyword_results=1,
            deduplicated_results=1,
            final_results=1,
        ),
    )
    llm = FakeLLMProvider(answer="Because it keeps vectors in PostgreSQL.")
    service = ChatRagService(
        conversations=repository,
        retrieval=retrieval,
        llm=llm,
        max_context_chunks=5,
        min_relevance_score=0.0,
        llm_model="fake-model",
        history_messages=2,
        title_max_length=80,
    )

    await service.send_message(
        SendConversationMessageCommand(
            owner_id=uuid4(),
            workspace_id=conversation.workspace_id,
            conversation_id=conversation.id,
            message="Why was that chosen?",
            top_k=1,
            retrieval_mode=RetrievalMode.HYBRID,
        )
    )

    assert repository.recent_limit == 2
    assert "PostgreSQL with pgvector." in retrieval.calls[0]["query"]
    assert llm.calls


@pytest.mark.asyncio
async def test_conversation_lifecycle_and_message_persistence(client) -> None:
    token, workspace_id = await _create_user_workspace(client, "rag-chat@example.com")
    document = await _upload_document(
        client,
        token,
        workspace_id,
        "vectors.txt",
        b"ContextVault uses PostgreSQL with pgvector to store document vectors.",
    )
    await client.post(
        f"/api/v1/workspaces/{workspace_id}/documents/{document['id']}/index",
        headers={"Authorization": f"Bearer {token}"},
    )

    create_response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/conversations",
        headers={"Authorization": f"Bearer {token}"},
        json={"title": "RAG Chat"},
    )

    assert create_response.status_code == 201
    conversation_id = create_response.json()["id"]

    list_response = await client.get(
        f"/api/v1/workspaces/{workspace_id}/conversations",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert list_response.status_code == 200
    assert [item["id"] for item in list_response.json()] == [conversation_id]

    message_response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/conversations/{conversation_id}/messages",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "message": "What technology stores vectors?",
            "top_k": 5,
            "retrieval_mode": "hybrid",
            "reranking_enabled": True,
        },
    )
    assert message_response.status_code == 200
    payload = message_response.json()
    assert payload["conversation_id"] == conversation_id
    assert payload["user_message"]["role"] == "user"
    assert payload["assistant_message"]["role"] == "assistant"
    assert payload["assistant_message"]["sources"]
    assert payload["assistant_message"]["metadata"]["retrieval_mode"] == "hybrid"
    assert payload["assistant_message"]["metadata"]["history_messages_used"] == 0

    detail_response = await client.get(
        f"/api/v1/workspaces/{workspace_id}/conversations/{conversation_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    assert len(detail_payload["messages"]) == 2
    assert [message["role"] for message in detail_payload["messages"]] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_conversation_endpoints_enforce_workspace_isolation(client) -> None:
    token_one, workspace_one = await _create_user_workspace(
        client,
        "rag-chat-owner-one@example.com",
    )
    _, workspace_two = await _create_user_workspace(client, "rag-chat-owner-two@example.com")

    create_response = await client.post(
        f"/api/v1/workspaces/{workspace_one}/conversations",
        headers={"Authorization": f"Bearer {token_one}"},
        json={"title": "Private"},
    )
    conversation_id = create_response.json()["id"]

    wrong_workspace = await client.get(
        f"/api/v1/workspaces/{workspace_two}/conversations/{conversation_id}",
        headers={"Authorization": f"Bearer {token_one}"},
    )
    assert wrong_workspace.status_code == 404
    assert wrong_workspace.json()["error"]["code"] == "workspace_not_found"

    missing_response = await client.get(
        f"/api/v1/workspaces/{workspace_one}/conversations/{uuid4()}",
        headers={"Authorization": f"Bearer {token_one}"},
    )
    assert missing_response.status_code == 404
    assert missing_response.json()["error"]["code"] == "conversation_not_found"


@pytest.mark.asyncio
async def test_rag_query_and_search_contracts_remain_compatible_with_conversations(client) -> None:
    token, workspace_id = await _create_user_workspace(client, "rag-chat-regression@example.com")
    document = await _upload_document(
        client,
        token,
        workspace_id,
        "regression.txt",
        b"pgvector stores embeddings in PostgreSQL.",
    )
    await client.post(
        f"/api/v1/workspaces/{workspace_id}/documents/{document['id']}/index",
        headers={"Authorization": f"Bearer {token}"},
    )

    query_response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/rag/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"question": "What stores embeddings?", "top_k": 5},
    )
    search_response = await client.post(
        f"/api/v1/workspaces/{workspace_id}/rag/search",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "pgvector", "top_k": 5},
    )

    assert query_response.status_code == 200
    assert set(query_response.json()) == {"question", "answer", "sources", "metadata"}
    assert search_response.status_code == 200
    assert set(search_response.json()) == {"query", "retrieval_mode", "results", "metadata"}
