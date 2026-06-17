import asyncio
from io import BytesIO
from pathlib import Path

from rag_backend.core.errors import BadRequestError


class SimpleTextExtractor:
    async def extract(
        self,
        *,
        filename: str,
        content_type: str | None,
        content: bytes,
    ) -> str:
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise BadRequestError(
                "Document text encoding is not supported",
                code="unsupported_document_encoding",
                details={"filename": filename},
            ) from error


class PdfTextExtractor:
    async def extract(
        self,
        *,
        filename: str,
        content_type: str | None,
        content: bytes,
    ) -> str:
        def read_pdf() -> str:
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)

        try:
            return await asyncio.to_thread(read_pdf)
        except Exception as error:
            raise BadRequestError(
                "PDF text extraction failed",
                code="text_extraction_failed",
                details={"filename": filename},
            ) from error


class CompositeTextExtractor:
    def __init__(self) -> None:
        self.simple = SimpleTextExtractor()
        self.pdf = PdfTextExtractor()

    async def extract(
        self,
        *,
        filename: str,
        content_type: str | None,
        content: bytes,
    ) -> str:
        extension = Path(filename).suffix.lower()
        if extension in {".txt", ".md"}:
            return await self.simple.extract(
                filename=filename,
                content_type=content_type,
                content=content,
            )
        if extension == ".pdf":
            return await self.pdf.extract(
                filename=filename,
                content_type=content_type,
                content=content,
            )
        raise BadRequestError(
            "Unsupported document file type",
            code="unsupported_document_type",
            details={"filename": filename},
        )
