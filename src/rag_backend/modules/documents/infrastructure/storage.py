import asyncio
from pathlib import Path
from uuid import UUID


def _safe_filename(filename: str) -> str:
    return filename.replace("/", "_").replace("\\", "_")


class LocalDocumentStorage:
    def __init__(self, root_path: Path) -> None:
        self.root_path = root_path

    async def save(
        self,
        *,
        workspace_id: UUID,
        document_id: UUID,
        filename: str,
        content: bytes,
    ) -> str:
        safe_name = _safe_filename(filename)
        relative_path = Path(str(workspace_id)) / f"{document_id}-{safe_name}"
        destination = self.root_path / relative_path

        def write_file() -> None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)

        await asyncio.to_thread(write_file)
        return str(relative_path)

    async def delete(self, storage_path: str) -> None:
        target = self.root_path / storage_path

        def remove_file() -> None:
            if target.exists():
                target.unlink()

        await asyncio.to_thread(remove_file)
