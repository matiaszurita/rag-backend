from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from rag_backend.core.config import get_settings
from rag_backend.core.database import get_db_session
from rag_backend.modules.identity.domain.entities import User
from rag_backend.modules.identity.interfaces.dependencies import get_current_user
from rag_backend.modules.rag.application.dtos import (
    IndexDocumentCommand,
    QueryRagCommand,
    SearchSimilarChunksCommand,
)
from rag_backend.modules.rag.application.ports import (
    EmbeddingProviderPort,
    LLMProviderPort,
    RerankerPort,
    TextSplitterPort,
)
from rag_backend.modules.rag.application.services import (
    IndexDocumentService,
    QueryRagService,
    RetrievalService,
    SearchSimilarChunksService,
)
from rag_backend.modules.rag.domain.entities import RetrievalMode
from rag_backend.modules.rag.infrastructure.documents import SqlAlchemyDocumentAccessAdapter
from rag_backend.modules.rag.infrastructure.embeddings import GeminiEmbeddingProviderAdapter
from rag_backend.modules.rag.infrastructure.extractors import CompositeTextExtractor
from rag_backend.modules.rag.infrastructure.fakes import NoOpReranker
from rag_backend.modules.rag.infrastructure.llm_providers import GeminiLLMProviderAdapter
from rag_backend.modules.rag.infrastructure.repositories import SqlAlchemyChunkRepository
from rag_backend.modules.rag.infrastructure.splitters import LangChainTextSplitterAdapter
from rag_backend.modules.rag.interfaces.schemas import (
    IndexDocumentResponse,
    QueryMetadata,
    QueryRequest,
    QueryResponse,
    RagSource,
    RetrievalMetadata,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)

router = APIRouter(tags=["rag"])


def get_text_splitter() -> TextSplitterPort:
    settings = get_settings()
    return LangChainTextSplitterAdapter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )


def get_embedding_provider() -> EmbeddingProviderPort:
    settings = get_settings()
    return GeminiEmbeddingProviderAdapter(
        model=settings.gemini_embedding_model,
        api_key=settings.gemini_api_key,
    )


def get_llm_provider() -> LLMProviderPort:
    settings = get_settings()
    return GeminiLLMProviderAdapter(
        model=settings.gemini_llm_model,
        api_key=settings.gemini_api_key,
    )


def get_reranker() -> RerankerPort:
    settings = get_settings()
    if settings.rag_reranking_provider == "noop":
        return NoOpReranker()
    raise ValueError("Unsupported RAG reranking provider")


def _retrieval_mode(value: str | None) -> RetrievalMode | None:
    if value is None:
        return None
    return RetrievalMode(value)


def _retrieval_service(
    *,
    embeddings: EmbeddingProviderPort,
    chunks: SqlAlchemyChunkRepository,
    reranker: RerankerPort,
) -> RetrievalService:
    settings = get_settings()
    return RetrievalService(
        embeddings=embeddings,
        chunks=chunks,
        reranker=reranker,
        default_mode=RetrievalMode(settings.rag_retrieval_mode),
        vector_weight=settings.rag_vector_weight,
        keyword_weight=settings.rag_keyword_weight,
        vector_candidates=settings.rag_vector_candidates,
        keyword_candidates=settings.rag_keyword_candidates,
        reranking_enabled=settings.rag_reranking_enabled,
        reranking_provider=settings.rag_reranking_provider,
        reranking_candidates=settings.rag_reranking_candidates,
    )


@router.post(
    "/workspaces/{workspace_id}/documents/{document_id}/index",
    response_model=IndexDocumentResponse,
)
async def index_document(
    workspace_id: UUID,
    document_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    splitter: Annotated[TextSplitterPort, Depends(get_text_splitter)],
    embeddings: Annotated[EmbeddingProviderPort, Depends(get_embedding_provider)],
) -> IndexDocumentResponse:
    settings = get_settings()
    service = IndexDocumentService(
        documents=SqlAlchemyDocumentAccessAdapter(
            session=session,
            storage_root=settings.local_storage_path,
        ),
        extractor=CompositeTextExtractor(),
        splitter=splitter,
        embeddings=embeddings,
        chunks=SqlAlchemyChunkRepository(session),
    )
    result = await service.index(
        IndexDocumentCommand(
            owner_id=current_user.id,
            workspace_id=workspace_id,
            document_id=document_id,
        )
    )
    return IndexDocumentResponse(
        document_id=result.document_id,
        chunks_indexed=result.chunks_indexed,
        status=result.status,
    )


@router.post("/workspaces/{workspace_id}/rag/search", response_model=SearchResponse)
async def search_similar_chunks(
    workspace_id: UUID,
    request: SearchRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    embeddings: Annotated[EmbeddingProviderPort, Depends(get_embedding_provider)],
    reranker: Annotated[RerankerPort, Depends(get_reranker)],
) -> SearchResponse:
    settings = get_settings()
    chunks = SqlAlchemyChunkRepository(session)
    service = SearchSimilarChunksService(
        documents=SqlAlchemyDocumentAccessAdapter(
            session=session,
            storage_root=settings.local_storage_path,
        ),
        retrieval=_retrieval_service(
            embeddings=embeddings,
            chunks=chunks,
            reranker=reranker,
        ),
        default_top_k=settings.rag_search_top_k,
    )
    result = await service.search(
        SearchSimilarChunksCommand(
            owner_id=current_user.id,
            workspace_id=workspace_id,
            query=request.query,
            top_k=request.top_k,
            retrieval_mode=_retrieval_mode(request.retrieval_mode),
            reranking_enabled=request.reranking_enabled,
        )
    )
    return SearchResponse(
        query=result.query,
        retrieval_mode=result.retrieval_mode.value,
        results=[
            SearchResultItem(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                content=item.content,
                score=item.score,
                vector_score=item.vector_score,
                keyword_score=item.keyword_score,
                rerank_score=item.rerank_score,
                original_rank=item.original_rank,
                reranked_rank=item.reranked_rank,
                retrieval_source=item.retrieval_source.value,
                metadata=item.metadata,
            )
            for item in result.results
        ],
        metadata=RetrievalMetadata(
            retrieval_mode=result.metadata.retrieval_mode.value,
            vector_candidates=result.metadata.vector_candidates,
            keyword_candidates=result.metadata.keyword_candidates,
            vector_results=result.metadata.vector_results,
            keyword_results=result.metadata.keyword_results,
            deduplicated_results=result.metadata.deduplicated_results,
            final_results=result.metadata.final_results,
            fusion_algorithm=result.metadata.fusion_algorithm,
            reranking_enabled=result.metadata.reranking_enabled,
            reranking_provider=result.metadata.reranking_provider,
            reranking_applied=result.metadata.reranking_applied,
            reranking_candidates=result.metadata.reranking_candidates,
            candidates_before_rerank=result.metadata.candidates_before_rerank,
        ),
    )


@router.post("/workspaces/{workspace_id}/rag/query", response_model=QueryResponse)
async def query_rag(
    workspace_id: UUID,
    request: QueryRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    embeddings: Annotated[EmbeddingProviderPort, Depends(get_embedding_provider)],
    llm: Annotated[LLMProviderPort, Depends(get_llm_provider)],
    reranker: Annotated[RerankerPort, Depends(get_reranker)],
) -> QueryResponse:
    settings = get_settings()
    chunks = SqlAlchemyChunkRepository(session)
    service = QueryRagService(
        documents=SqlAlchemyDocumentAccessAdapter(
            session=session,
            storage_root=settings.local_storage_path,
        ),
        retrieval=_retrieval_service(
            embeddings=embeddings,
            chunks=chunks,
            reranker=reranker,
        ),
        llm=llm,
        max_context_chunks=settings.rag_answer_max_context_chunks,
        min_relevance_score=settings.rag_min_relevance_score,
        llm_model=settings.gemini_llm_model,
    )
    result = await service.query(
        QueryRagCommand(
            owner_id=current_user.id,
            workspace_id=workspace_id,
            question=request.question,
            top_k=request.top_k,
            retrieval_mode=_retrieval_mode(request.retrieval_mode),
            reranking_enabled=request.reranking_enabled,
        )
    )
    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sources=[
            RagSource(
                chunk_id=source.chunk_id,
                document_id=source.document_id,
                filename=source.filename,
                score=source.score,
                vector_score=source.vector_score,
                keyword_score=source.keyword_score,
                rerank_score=source.rerank_score,
                original_rank=source.original_rank,
                reranked_rank=source.reranked_rank,
                retrieval_source=source.retrieval_source.value,
                content_preview=source.content_preview,
            )
            for source in result.sources
        ],
        metadata=QueryMetadata(
            context_chunks_used=result.metadata.context_chunks_used,
            top_k=result.metadata.top_k,
            llm_model=result.metadata.llm_model,
            context_char_count=result.metadata.context_char_count,
            retrieval_mode=result.metadata.retrieval_mode.value,
            fusion_algorithm=result.metadata.fusion_algorithm,
            reranking_enabled=result.metadata.reranking_enabled,
            reranking_provider=result.metadata.reranking_provider,
            reranking_applied=result.metadata.reranking_applied,
        ),
    )
