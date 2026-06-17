from uuid import uuid4

from rag_backend.core.errors import AppError, BadRequestError, NotFoundError
from rag_backend.modules.documents.domain.entities import DocumentStatus
from rag_backend.modules.rag.application.dtos import (
    IndexDocumentCommand,
    IndexDocumentResult,
    QueryRagCommand,
    QueryRagMetadataDTO,
    QueryRagResult,
    RagSourceDTO,
    SearchSimilarChunksCommand,
    SearchSimilarChunksResult,
    SimilarChunkDTO,
)
from rag_backend.modules.rag.application.ports import (
    ChunkRepositoryPort,
    DocumentAccessPort,
    EmbeddingProviderPort,
    LLMProviderPort,
    TextExtractorPort,
    TextSplitterPort,
)
from rag_backend.modules.rag.domain.entities import DocumentChunk

INSUFFICIENT_CONTEXT_ANSWER = (
    "No encontré información suficiente en los documentos indexados para responder esta pregunta."
)

RAG_SYSTEM_PROMPT = """Eres un asistente de recuperación de conocimiento técnico.
Responde únicamente usando el contexto proporcionado.
Si el contexto no contiene información suficiente para responder, dilo claramente.
No inventes información.
Responde de forma clara, breve y técnica.
Cuando sea útil, menciona que la respuesta se basa en los documentos recuperados."""


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


class QueryRagService:
    def __init__(
        self,
        *,
        documents: DocumentAccessPort,
        embeddings: EmbeddingProviderPort,
        chunks: ChunkRepositoryPort,
        llm: LLMProviderPort,
        max_context_chunks: int,
        min_relevance_score: float,
        llm_model: str,
    ) -> None:
        self.documents = documents
        self.embeddings = embeddings
        self.chunks = chunks
        self.llm = llm
        self.max_context_chunks = max_context_chunks
        self.min_relevance_score = min_relevance_score
        self.llm_model = llm_model

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
        query_embedding = await self.embeddings.embed_query(question)
        results = await self.chunks.similarity_search(
            workspace_id=command.workspace_id,
            query_embedding=query_embedding,
            limit=top_k,
        )
        relevant_chunks = [
            result for result in results if result.score >= self.min_relevance_score
        ][:top_k]

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
                ),
            )

        user_prompt = self._build_user_prompt(question, relevant_chunks)
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
            ),
        )

    def _build_user_prompt(self, question: str, chunks: list) -> str:
        context_blocks = []
        for index, chunk in enumerate(chunks, start=1):
            filename = self._filename(chunk.metadata)
            context_blocks.append(
                "\n".join(
                    [
                        f"[Fuente {index}]",
                        f"chunk_id: {chunk.chunk_id}",
                        f"document_id: {chunk.document_id}",
                        f"filename: {filename}",
                        f"score: {chunk.score:.4f}",
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

    def _source_from_chunk(self, chunk) -> RagSourceDTO:
        return RagSourceDTO(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            filename=self._filename(chunk.metadata),
            score=chunk.score,
            content_preview=self._preview(chunk.content),
        )

    def _filename(self, metadata: dict[str, object]) -> str:
        source = metadata.get("source")
        if isinstance(source, str) and source.strip():
            return source
        return "unknown"

    def _preview(self, content: str) -> str:
        normalized = " ".join(content.split())
        return normalized[:240]
