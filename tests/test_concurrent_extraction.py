from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

extractor_pkg = types.ModuleType("extractor")
extractor_pkg.__path__ = [str(PROJECT_ROOT / "extractor")]
ingestor_pkg = types.ModuleType("extractor.ingestor")
ingestor_pkg.__path__ = [str(PROJECT_ROOT / "extractor" / "ingestor")]
sys.modules.setdefault("extractor", extractor_pkg)
sys.modules.setdefault("extractor.ingestor", ingestor_pkg)

connection_stub = types.ModuleType("extractor.ingestor.connection")


class _DummyConfig:
    pass


def _dummy_build_llm(_config):
    return None


connection_stub.IngestorConnectionConfig = _DummyConfig
connection_stub.build_llm = _dummy_build_llm
sys.modules.setdefault("extractor.ingestor.connection", connection_stub)

prompt_stub = types.ModuleType("extractor.ingestor.extraction_prompt")


def _dummy_build_prompt(_definition):
    return None


prompt_stub.build_extraction_prompt = _dummy_build_prompt
sys.modules.setdefault("extractor.ingestor.extraction_prompt", prompt_stub)

kg_stub = types.ModuleType("extractor.ingestor.kg_extraction_definition")


class _DummyKGDefinition:
    pass


kg_stub.KGExtractionDefinition = _DummyKGDefinition
kg_stub.DEFAULT_KG_EXTRACTION_DEFINITION = _DummyKGDefinition()
sys.modules.setdefault("extractor.ingestor.kg_extraction_definition", kg_stub)

MODULE_PATH = PROJECT_ROOT / "extractor" / "ingestor" / "langchain_extractor.py"
SPEC = importlib.util.spec_from_file_location(
    "extractor.ingestor.langchain_extractor", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"无法加载模块: {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

Evidence = MODULE.Evidence
ExtractedTriple = MODULE.ExtractedTriple
LangChainKGExtractor = MODULE.LangChainKGExtractor


class _ThreadPoolSpy:
    called = False
    max_workers = None

    def __init__(self, max_workers: int | None = None, *args, **kwargs):
        _ThreadPoolSpy.called = True
        _ThreadPoolSpy.max_workers = max_workers
        self._executor = RealThreadPoolExecutor(
            max_workers=max_workers, *args, **kwargs
        )

    def __enter__(self):
        self._executor.__enter__()
        return self._executor

    def __exit__(self, exc_type, exc, tb):
        return self._executor.__exit__(exc_type, exc, tb)


class TestConcurrentExtraction(unittest.TestCase):
    @staticmethod
    def _make_extractor() -> tuple[LangChainKGExtractor, list[dict]]:
        extractor = object.__new__(LangChainKGExtractor)
        log_calls: list[dict] = []

        def _fake_write_log(**kwargs):
            log_calls.append(kwargs)

        extractor._write_log = _fake_write_log  # type: ignore[attr-defined]
        extractor._print_progress = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
        return extractor, log_calls

    @staticmethod
    def _fake_extract_from_text_factory(raise_on: str | None = None):
        def _fake_extract_from_text(
            text: str,
            source_doc: str,
            section: str = "",
            time_or_policy_version: str = "",
            write_log: bool = True,
        ) -> list[ExtractedTriple]:
            if raise_on is not None and text == raise_on:
                raise RuntimeError("mock chunk failure")
            return [
                ExtractedTriple(
                    subject=f"S_{text}",
                    subject_type="技术",
                    relation="应用于",
                    object=f"O_{text}",
                    object_type="场景",
                    evidence=Evidence(
                        source_doc=source_doc,
                        source_span=text,
                        section=section,
                        confidence=0.9,
                        time_or_policy_version=time_or_policy_version,
                    ),
                )
            ]

        return _fake_extract_from_text

    def test_workers_one_uses_serial_path(self) -> None:
        extractor, log_calls = self._make_extractor()
        extractor.extract_from_text = self._fake_extract_from_text_factory()  # type: ignore[attr-defined]

        with patch.object(
            MODULE,
            "ThreadPoolExecutor",
            side_effect=AssertionError("workers=1 不应创建线程池"),
        ):
            triples = extractor.extract_from_chunks(
                chunks=["A", "B", "A"],
                source_doc="demo.pdf",
                section="测试",
                time_or_policy_version="2026",
                workers=1,
            )

        self.assertEqual(len(triples), 2, "串行分支应完成去重")
        self.assertEqual(len(log_calls), 1)
        self.assertEqual(log_calls[0]["chunk_count"], 3)

    def test_workers_gt_one_uses_thread_pool(self) -> None:
        extractor, log_calls = self._make_extractor()
        extractor.extract_from_text = self._fake_extract_from_text_factory()  # type: ignore[attr-defined]

        _ThreadPoolSpy.called = False
        _ThreadPoolSpy.max_workers = None

        with patch.object(
            MODULE,
            "ThreadPoolExecutor",
            new=_ThreadPoolSpy,
        ):
            triples = extractor.extract_from_chunks(
                chunks=["A", "B", "A", "C"],
                source_doc="demo.pdf",
                section="测试",
                time_or_policy_version="2026",
                workers=4,
            )

        self.assertTrue(_ThreadPoolSpy.called, "workers>1 时应走线程池分支")
        self.assertEqual(_ThreadPoolSpy.max_workers, 4)
        self.assertEqual(len(triples), 3, "并发分支应完成去重")
        self.assertEqual(len(log_calls), 1)
        self.assertEqual(log_calls[0]["chunk_count"], 4)

    def test_parallel_chunk_error_does_not_fail_whole_file(self) -> None:
        extractor, log_calls = self._make_extractor()
        extractor.extract_from_text = self._fake_extract_from_text_factory(raise_on="BAD")  # type: ignore[attr-defined]

        triples = extractor.extract_from_chunks(
            chunks=["A", "BAD", "B"],
            source_doc="demo.pdf",
            section="测试",
            time_or_policy_version="2026",
            workers=3,
        )

        self.assertEqual(len(triples), 2)
        self.assertEqual(len(log_calls), 1)
        self.assertEqual(log_calls[0]["chunk_count"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
