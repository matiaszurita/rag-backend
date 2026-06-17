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
    RetrievalMetadataDTO,
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
from rag_backend.modules.rag.domain.entities import (
    DocumentChunk,
    RetrievalMode,
    RetrievalSource,
    SimilarChunk,
)

INSUFFICIENT_CONTEXT_ANSWER = (
    "No encontré información suficiente en los documentos indexados para responder esta pregunta."
)

RAG_SYSTEM_PROMPT = """Eres un asistente de recuperación de conocimiento técnico.
Responde únicamente usando el contexto proporcionado.
Si el contexto no contiene información suficiente para responder, dilo claramente.
No inventes información.
Responde de forma clara, breve y técnica.
Cuando sea útil, menciona que la respuesta se basa en los documentos recuperados."""

FUSION_ALGORITHM = "weighted_rrf"
RRF_K = 60


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


class RetrievalService:
    def __init__(
        self,
        *,
        embeddings: EmbeddingProviderPort,
        chunks: ChunkRepositoryPort,
        default_mode: RetrievalMode,
        vector_weight: float,
        keyword_weight: float,
        vector_candidates: int,
        keyword_candidates: int,
    ) -> None:
        self.embeddings = embeddings
        self.chunks = chunks
        self.default_mode = default_mode
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.vector_candidates = vector_candidates
        self.keyword_candidates = keyword_candidates

    async def retrieve(
        self,
        *,
        workspace_id,
        query: str,
        top_k: int,
        retrieval_mode: RetrievalMode | None,
    ) -> tuple[list[SimilarChunk], RetrievalMetadataDTO]:
        mode = retrieval_mode or self.default_mode
        if mode == RetrievalMode.VECTOR:
            return await self._retrieve_vector(
                workspace_id=workspace_id,
                query=query,
                top_k=top_k,
            )
        if mode == RetrievalMode.KEYWORD:
            return await self._retrieve_keyword(
                workspace_id=workspace_id,
                query=query,
                top_k=top_k,
            )
        if mode == RetrievalMode.HYBRID:
            return await self._retrieve_hybrid(
                workspace_id=workspace_id,
                query=query,
                top_k=top_k,
            )
        raise BadRequestError("Unsupported retrieval mode", code="invalid_retrieval_mode")

    async def _retrieve_vector(self, *, workspace_id, query: str, top_k: int):
        query_embedding = await self.embeddings.embed_query(query)
        results = await self.chunks.vector_search(
            workspace_id=workspace_id,
            query_embedding=query_embedding,
            limit=top_k,
        )
        return results, RetrievalMetadataDTO(
            retrieval_mode=RetrievalMode.VECTOR,
            vector_candidates=top_k,
            keyword_candidates=0,
            vector_results=len(results),
            keyword_results=0,
            deduplicated_results=len(results),
            final_results=len(results),
        )

    async def _retrieve_keyword(self, *, workspace_id, query: str, top_k: int):
        results = await self.chunks.keyword_search(
            workspace_id=workspace_id,
            query=query,
            limit=top_k,
        )
        return results, RetrievalMetadataDTO(
            retrieval_mode=RetrievalMode.KEYWORD,
            vector_candidates=0,
            keyword_candidates=top_k,
            vector_results=0,
            keyword_results=len(results),
            deduplicated_results=len(results),
            final_results=len(results),
        )

    async def _retrieve_hybrid(self, *, workspace_id, query: str, top_k: int):
        query_embedding = await self.embeddings.embed_query(query)
        vector_limit = max(top_k, self.vector_candidates)
        keyword_limit = max(top_k, self.keyword_candidates)
        vector_results = await self.chunks.vector_search(
            workspace_id=workspace_id,
            query_embedding=query_embedding,
            limit=vector_limit,
        )
        keyword_results = await self.chunks.keyword_search(
            workspace_id=workspace_id,
            query=query,
            limit=keyword_limit,
        )
        all_results = [*vector_results, *keyword_results]
        deduplicated_results = len({chunk.chunk_id for chunk in all_results})
        fused = self._fuse(vector_results, keyword_results)[:top_k]
        return fused, RetrievalMetadataDTO(
            retrieval_mode=RetrievalMode.HYBRID,
            vector_candidates=vector_limit,
            keyword_candidates=keyword_limit,
            vector_results=len(vector_results),
            keyword_results=len(keyword_results),
            deduplicated_results=deduplicated_results,
            final_results=len(fused),
            fusion_algorithm=FUSION_ALGORITHM,
        )

    def _fuse(
        self,
        vector_results: list[SimilarChunk],
        keyword_results: list[SimilarChunk],
    ) -> list[SimilarChunk]:
        by_id: dict[object, SimilarChunk] = {}
        fused_scores: dict[object, float] = {}

        for rank, chunk in enumerate(vector_results, start=1):
            by_id[chunk.chunk_id] = SimilarChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.content,
                score=0.0,
                vector_score=(
                    chunk.vector_score
                    if chunk.vector_score is not None
                    else chunk.score
                ),
                keyword_score=None,
                retrieval_source=RetrievalSource.VECTOR,
                metadata=chunk.metadata,
            )
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + (
                self.vector_weight / (RRF_K + rank)
            )

        for rank, chunk in enumerate(keyword_results, start=1):
            existing = by_id.get(chunk.chunk_id)
            if existing is None:
                by_id[chunk.chunk_id] = SimilarChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    score=0.0,
                    vector_score=None,
                    keyword_score=(
                        chunk.keyword_score
                        if chunk.keyword_score is not None
                        else chunk.score
                    ),
                    retrieval_source=RetrievalSource.KEYWORD,
                    metadata=chunk.metadata,
                )
            else:
                existing.keyword_score = (
                    chunk.keyword_score
                    if chunk.keyword_score is not None
                    else chunk.score
                )
                existing.retrieval_source = RetrievalSource.HYBRID
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + (
                self.keyword_weight / (RRF_K + rank)
            )

        max_score = max(fused_scores.values(), default=1.0)
        for chunk_id, chunk in by_id.items():
            chunk.score = fused_scores[chunk_id] / max_score if max_score else 0.0
            if chunk.retrieval_source != RetrievalSource.HYBRID:
                chunk.retrieval_source = (
                    RetrievalSource.VECTOR
                    if chunk.vector_score is not None
                    else RetrievalSource.KEYWORD
                )
        return sorted(by_id.values(), key=lambda chunk: chunk.score, reverse=True)


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
    ) -> None:
        self.documents = documents
        self.retrieval = retrieval
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
                retrieval_mode=retrieval_metadata.retrieval_mode,
                fusion_algorithm=retrieval_metadata.fusion_algorithm,
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

    def _source_from_chunk(self, chunk) -> RagSourceDTO:
        return RagSourceDTO(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            filename=self._filename(chunk.metadata),
            score=chunk.score,
            vector_score=chunk.vector_score,
            keyword_score=chunk.keyword_score,
            retrieval_source=chunk.retrieval_source,
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
