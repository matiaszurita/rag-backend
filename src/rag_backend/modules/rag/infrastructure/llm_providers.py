import asyncio


class GeminiLLMProviderAdapter:
    def __init__(self, *, model: str, api_key: str | None = None) -> None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs: dict[str, str] = {"model": model}
        if api_key:
            kwargs["api_key"] = api_key
        self.model = ChatGoogleGenerativeAI(**kwargs)

    async def generate_answer(self, *, system_prompt: str, user_prompt: str) -> str:
        response = await asyncio.to_thread(
            self.model.invoke,
            [("system", system_prompt), ("human", user_prompt)],
        )
        content = response.content
        if isinstance(content, str):
            return content
        return str(content)
