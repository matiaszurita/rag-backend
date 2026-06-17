import asyncio


class GeminiEmbeddingProviderAdapter:
    def __init__(self, *, model: str, api_key: str | None = None) -> None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        kwargs: dict[str, str] = {"model": model}
        if api_key:
            kwargs["api_key"] = api_key
        self.embeddings = GoogleGenerativeAIEmbeddings(**kwargs)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embeddings.embed_documents, texts)

    async def embed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embeddings.embed_query, text)
