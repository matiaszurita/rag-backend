from dataclasses import dataclass
from uuid import UUID


@dataclass(slots=True)
class KnowledgeChunk:
    document_id: UUID
    text: str
    index: int
