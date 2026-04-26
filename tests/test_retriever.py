"""Unit tests for retrievers and RAG chain."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from orchestration.rag_chain import RAGChain
from retriever.hybrid_retriever import HybridRetriever, HybridRetrieverConfig
from retriever.rag_retriever import RAGRetriever, RAGRetrieverConfig
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


class FakeChatModel(BaseChatModel):
    """Minimal fake chat model that always returns a fixed answer."""

    def __init__(self, answer: str = "mock-answer") -> None:
        super().__init__()
        self._answer = answer

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        generation = ChatGeneration(message=AIMessage(content=self._answer))
        return ChatResult(generations=[generation])


class TestRetrieversAndChain(unittest.TestCase):
    def _build_store(self) -> VectorStoreManager:
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        manager = VectorStoreManager(
            provider="faiss",
            persist_directory=tmp.name,
            collection_name="test_collection",
            embedding=DeterministicEmbeddings(),
        )
        manager.add_documents(
            [
                Document(page_content="Cats are great pets.", metadata={"source": "a.txt", "chunk_index": 0}),
                Document(page_content="Dogs are loyal animals.", metadata={"source": "b.txt", "chunk_index": 0}),
                Document(page_content="Cats like to sleep.", metadata={"source": "a.txt", "chunk_index": 1}),
            ]
        )
        self.addCleanup(manager.close)
        return manager

    def test_rag_retriever_similarity(self) -> None:
        store = self._build_store()
        retriever = RAGRetriever(store, config=RAGRetrieverConfig(search_type="similarity", k=2))
        docs = retriever.invoke("cats")
        self.assertGreaterEqual(len(docs), 1)

    def test_hybrid_retriever_augments_and_deduplicates(self) -> None:
        store = self._build_store()
        vector = RAGRetriever(store, config=RAGRetrieverConfig(search_type="similarity", k=2))

        hybrid = HybridRetriever(
            vector,
            kg_query_entities=lambda _q: ["cats"],
            config=HybridRetrieverConfig(vector_k=2, kg_entity_k=2, final_k=3),
        )
        docs = hybrid.invoke("cats")
        self.assertGreaterEqual(len(docs), 1)
        keys = {(d.metadata.get("source"), d.metadata.get("chunk_index")) for d in docs}
        self.assertEqual(len(keys), len(docs))

    def test_rag_chain_runs_with_mock_llm(self) -> None:
        store = self._build_store()
        retriever = RAGRetriever(store, config=RAGRetrieverConfig(search_type="similarity", k=2))
        llm = FakeChatModel(answer="ok")

        chain = RAGChain(retriever=retriever, llm=llm)
        result = chain.run("What are cats?")

        self.assertIn("answer", result)
        self.assertEqual(result["answer"], "ok")
        self.assertIn("source_documents", result)
        self.assertIsInstance(result["source_documents"], list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
