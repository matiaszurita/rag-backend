from rag_backend.core.errors import BadRequestError, NotFoundError
from rag_backend.modules.rag.application.dtos import (
    SearchSimilarChunksCommand,
    SearchSimilarChunksResult,
    SimilarChunkDTO,
)
from rag_backend.modules.rag.application.ports import DocumentAccessPort
from rag_backend.modules.rag.application.services.retrieval_service import RetrievalService


class SearchSimilarChunksService:
    def __init__(
        self,
        *,
        documents: DocumentAccessPort,
        retrieval: RetrievalService,
        default_top_k: int,
    ) -> None:
        self.documents = documents
        self.retrieval = retrieval
        self.default_top_k = default_top_k

    async def search(self, command: SearchSimilarChunksCommand) -> SearchSimilarChunksResult:
        query = command.query.strip()
        if not query:
            raise BadRequestError("Search query cannot be empty", code="empty_query")

        workspace_exists = await self.documents.workspace_exists_for_owner(
            owner_id=command.owner_id,
            workspace_id=command.workspace_id,
        )
        if not workspace_exists:
            raise NotFoundError("Workspace not found", code="workspace_not_found")

        top_k = command.top_k or self.default_top_k
        if top_k < 1:
            raise BadRequestError("top_k must be at least 1", code="invalid_top_k")

        results, metadata = await self.retrieval.retrieve(
            workspace_id=command.workspace_id,
            query=query,
            top_k=top_k,
            retrieval_mode=command.retrieval_mode,
        )
        return SearchSimilarChunksResult(
            query=query,
            retrieval_mode=metadata.retrieval_mode,
            results=[
                SimilarChunkDTO(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    content=result.content,
                    score=result.score,
                    vector_score=result.vector_score,
                    keyword_score=result.keyword_score,
                    retrieval_source=result.retrieval_source,
                    metadata=result.metadata,
                )
                for result in results
            ],
            metadata=metadata,
        )
