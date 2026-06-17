from uuid import uuid4

from rag_backend.core.errors import AppError, BadRequestError, NotFoundError
from rag_backend.modules.documents.domain.entities import DocumentStatus
from rag_backend.modules.rag.application.dtos import (
    IndexDocumentCommand,
    IndexDocumentResult,
    SearchSimilarChunksCommand,
    SearchSimilarChunksResult,
    SimilarChunkDTO,
)
from rag_backend.modules.rag.application.ports import (
    ChunkRepositoryPort,
    DocumentAccessPort,
    EmbeddingProviderPort,
    TextExtractorPort,
    TextSplitterPort,
)
from rag_backend.modules.rag.domain.entities import DocumentChunk


class IndexDocumentService:
    def __init__(
        self,
        *,
        documents: DocumentAccessPort,
        extractor: TextExtractorPort,
        splitter: TextSplitterPort,
        embeddings: EmbeddingProviderPort,
        chunks: ChunkRepositoryPort,
    ) -> None:
        self.documents = documents
        self.extractor = extractor
        self.splitter = splitter
        self.embeddings = embeddings
        self.chunks = chunks

    async def index(self, command: IndexDocumentCommand) -> IndexDocumentResult:
        document = await self.documents.get_document_for_owner(
            owner_id=command.owner_id,
            workspace_id=command.workspace_id,
            document_id=command.document_id,
        )
        if document is None:
            raise NotFoundError("Document not found", code="document_not_found")

        await self.documents.update_document_status(document.id, DocumentStatus.INDEXING)
        await self.documents.commit()

        try:
            content = await self.documents.read_document_content(document.storage_path)
            text = await self.extractor.extract(
                filename=document.original_filename,
                content_type=document.content_type,
                content=content,
            )
            if not text.strip():
                raise BadRequestError("Document has no extractable text", code="empty_document")

            split_texts = [chunk for chunk in self.splitter.split(text) if chunk.strip()]
            if not split_texts:
                raise BadRequestError("Document has no extractable text", code="empty_document")

            vectors = await self.embeddings.embed_documents(split_texts)
            if len(vectors) != len(split_texts):
                raise BadRequestError(
                    "Embedding provider returned invalid results",
                    code="embedding_failed",
                )

            chunks = [
                DocumentChunk(
                    id=uuid4(),
                    workspace_id=document.workspace_id,
                    document_id=document.id,
                    content=chunk_text,
                    chunk_index=index,
                    embedding=vectors[index],
                    metadata={"source": document.original_filename},
                )
                for index, chunk_text in enumerate(split_texts)
            ]

            await self.chunks.replace_for_document(document.id, chunks)
            await self.documents.update_document_status(document.id, DocumentStatus.INDEXED)
            await self.chunks.commit()
            return IndexDocumentResult(
                document_id=document.id,
                chunks_indexed=len(chunks),
                status=DocumentStatus.INDEXED.value,
            )
        except AppError:
            await self.documents.update_document_status(document.id, DocumentStatus.INDEX_FAILED)
            await self.documents.commit()
            raise
        except Exception as error:
            await self.documents.update_document_status(document.id, DocumentStatus.INDEX_FAILED)
            await self.documents.commit()
            raise BadRequestError(
                "Document indexing failed",
                code="document_indexing_failed",
            ) from error


class SearchSimilarChunksService:
    def __init__(
        self,
        *,
        documents: DocumentAccessPort,
        embeddings: EmbeddingProviderPort,
        chunks: ChunkRepositoryPort,
        default_top_k: int,
    ) -> None:
        self.documents = documents
        self.embeddings = embeddings
        self.chunks = chunks
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

        query_embedding = await self.embeddings.embed_query(query)
        results = await self.chunks.similarity_search(
            workspace_id=command.workspace_id,
            query_embedding=query_embedding,
            limit=top_k,
        )
        return SearchSimilarChunksResult(
            query=query,
            results=[
                SimilarChunkDTO(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    content=result.content,
                    score=result.score,
                    metadata=result.metadata,
                )
                for result in results
            ],
        )
