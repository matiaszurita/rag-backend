class LangChainTextSplitterAdapter:
    def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, text: str) -> list[str]:
        return self.splitter.split_text(text)
