"""Unit tests for the vector store manager."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from storage.vector_store import VectorStoreManager


class DeterministicEmbeddings(Embeddings):
    """Deterministic embeddings for offline tests."""

    def __init__(self, *, dim: int = 8) -> None:
        self._dim = dim

    def _vec(self, text: str) -> list[float]:
        base = abs(hash(text)) % 10_000
        return [float((base + i * 997) % 1_000) / 1_000.0 for i in range(self._dim)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class TestVectorStoreManager(unittest.TestCase):
    def test_add_and_search_chroma(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            manager = VectorStoreManager(
                provider="faiss",
                persist_directory=tmp,
                collection_name="test_collection",
                embedding=DeterministicEmbeddings(),
            )

            docs = [
                Document(page_content="Cats are great pets.", metadata={"source": "a.txt", "chunk_index": 0}),
                Document(page_content="Dogs are loyal animals.", metadata={"source": "b.txt", "chunk_index": 0}),
            ]
            manager.add_documents(docs)
            hits = manager.similarity_search("cats", k=1)

            self.assertEqual(len(hits), 1)
            self.assertIsInstance(hits[0], Document)
            self.assertIn("source", hits[0].metadata)
            manager.close()

    def test_persist_and_load(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            emb = DeterministicEmbeddings()
            manager1 = VectorStoreManager(
                provider="faiss",
                persist_directory=tmp,
                collection_name="test_collection",
                embedding=emb,
            )
            manager1.add_documents([Document(page_content="Hello world.", metadata={"source": "a.txt"})])
            manager1.persist()
            manager1.close()

            manager2 = VectorStoreManager(
                provider="faiss",
                persist_directory=tmp,
                collection_name="test_collection",
                embedding=emb,
            )
            hits = manager2.similarity_search("hello", k=1)
            self.assertEqual(len(hits), 1)
            manager2.close()

            manager3 = VectorStoreManager(
                provider="faiss",
                persist_directory=Path(tmp) / "other",
                collection_name="test_collection",
                embedding=emb,
            )
            manager3.load(tmp)
            hits2 = manager3.similarity_search("hello", k=1)
            self.assertEqual(len(hits2), 1)
            manager3.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
