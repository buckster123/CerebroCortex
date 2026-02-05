"""Storage backend protocol for CerebroCortex vector store."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class VectorStore(ABC):
    """Protocol for vector store backends (ChromaDB, etc.)."""

    @abstractmethod
    def initialize(self) -> None: ...

    @abstractmethod
    def add(self, collection: str, ids: list[str], documents: list[str],
            metadatas: list[dict[str, Any]]) -> list[str]: ...

    @abstractmethod
    def search(self, collection: str, query: str, n_results: int = 10,
               where: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]: ...

    @abstractmethod
    def get(self, collection: str, ids: list[str]) -> list[dict[str, Any]]: ...

    @abstractmethod
    def update(self, collection: str, ids: list[str], documents: list[str],
               metadatas: list[dict[str, Any]]) -> bool: ...

    @abstractmethod
    def delete(self, collection: str, ids: list[str]) -> bool: ...

    @abstractmethod
    def count(self, collection: str) -> int: ...
