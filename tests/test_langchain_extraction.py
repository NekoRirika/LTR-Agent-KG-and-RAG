from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from extractor.ingestor.connection import IngestorConnectionConfig
from extractor.ingestor.langchain_extractor import LangChainKGExtractor
from extractor.filereader.read_file import FileReader
from extractor.filereader.text_chunker import ChineseTextChunker
from extractor.filereader.text_normalize import normalize_text


class TestLangChainKGExtraction(unittest.TestCase):
    def test_extract_with_mock_llm_and_write_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            valid = {
                "subject": "AIGC",
                "subject_type": "技术",
                "relation": "应用于",
                "object": "学术出版",
                "object_type": "场景",
                "evidence": {
                    "source_doc": "paper_001",
                    "source_span": "AIGC运用于学术出版",
                    "section": "研究现状",
                    "confidence": 0.92,
                    "time_or_policy_version": "2026",
                },
            }
            invalid = {
                "subject": "AIGC",
                "subject_type": "技术",
                "relation": "应用于",
                "object": "版权归属争议",
                "object_type": "风险",
                "evidence": {
                    "source_doc": "paper_001",
                    "source_span": "AIGC运用于学术出版",
                    "section": "研究现状",
                    "confidence": 0.95,
                    "time_or_policy_version": "2026",
                },
            }

            mock_response = {"triples": [valid, invalid]}
            fake_llm = RunnableLambda(
                lambda _input: AIMessage(
                    content=json.dumps(mock_response, ensure_ascii=False)
                )
            )

            config = IngestorConnectionConfig(
                llm_api_key="test-key",
                llm_base_url="https://example.com/v1",
                llm_model="mock-model",
                llm_temperature=0.0,
                neo4j_uri="bolt://localhost:7687",
                neo4j_username="neo4j",
                neo4j_password="",
                neo4j_database="neo4j",
            )

            with patch(
                "extractor.ingestor.langchain_extractor.build_llm",
                return_value=fake_llm,
            ):
                extractor = LangChainKGExtractor(config=config, log_dir=temp_dir)
                triples = extractor.extract_from_text(
                    text="AIGC运用于学术出版。",
                    source_doc="paper_001",
                    section="研究现状",
                )

            self.assertEqual(len(triples), 1, "应只保留与模板对齐的三元组")
            self.assertEqual(triples[0].relation, "应用于")
            self.assertEqual(triples[0].subject_type, "技术")
            self.assertEqual(triples[0].object_type, "场景")
            self.assertGreaterEqual(triples[0].evidence.confidence, 0.75)

            logs = list(Path(temp_dir).glob("extract_*.json"))
            self.assertGreaterEqual(len(logs), 1, "应生成按时间命名的抽取日志")
            payload = json.loads(logs[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["source_doc"], "paper_001")
            self.assertEqual(len(payload["accepted"]), 1)

    @unittest.skipIf(
        not os.getenv("LLM_API_KEY"),
        "未配置 LLM_API_KEY，跳过在线提取联调测试",
    )
    def test_extract_online_optional(self) -> None:
        config = IngestorConnectionConfig.from_env(PROJECT_ROOT / ".env")
        with tempfile.TemporaryDirectory() as temp_dir:
            extractor = LangChainKGExtractor(config=config, log_dir=temp_dir)
            triples = extractor.extract_from_text(
                text="AIGC运用于学术出版，可能引发版权归属争议。",
                source_doc="paper_online",
                section="测试",
                time_or_policy_version="2026",
            )

            self.assertIsInstance(triples, list)
            for item in triples:
                self.assertTrue(item.subject)
                self.assertTrue(item.object)
                self.assertTrue(item.relation)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LangChain 抽取测试")
    parser.add_argument(
        "--run-demo",
        action="store_true",
        help="执行单文件抽取演示（不入库，仅产生日志）",
    )
    parser.add_argument(
        "--file",
        default="files/AIGC时代的学术出版伦理：风险挑战与治理路径_葛建平.pdf",
        help="待抽取文件路径（相对项目根目录）",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=2,
        help="最多抽取前 N 个分块，默认 2",
    )
    parser.add_argument(
        "--log-dir",
        default="extractor/ingestor/log",
        help="日志输出目录",
    )
    return parser


def run_single_file_demo(args: argparse.Namespace) -> int:
    file_path = (PROJECT_ROOT / args.file).resolve()
    if not file_path.exists():
        print(f"文件不存在: {file_path}")
        return 1

    config = IngestorConnectionConfig.from_env(PROJECT_ROOT / ".env")
    reader = FileReader(file_path.parent)
    raw_content = reader.read(file_path)
    content, normalize_stats = normalize_text(raw_content)

    chunker = ChineseTextChunker(use_hanlp=False)
    chunks = chunker.chunk_text(content)
    chunks = chunks[: max(args.max_chunks, 1)]

    extractor = LangChainKGExtractor(config=config, log_dir=PROJECT_ROOT / args.log_dir)
    triples = extractor.extract_from_chunks(
        chunks=chunks,
        source_doc=file_path.name,
        section="自动测试",
        time_or_policy_version="2026",
    )

    log_dir = PROJECT_ROOT / args.log_dir
    logs = sorted(log_dir.glob("extract_*.json"), key=lambda p: p.stat().st_mtime)
    if not logs:
        print("未生成日志文件")
        return 1

    latest_log = logs[-1]
    payload = json.loads(latest_log.read_text(encoding="utf-8"))

    print("=== 单文件抽取完成（未入库）===")
    print(f"文件: {file_path.name}")
    print(f"规范化统计: {json.dumps(normalize_stats, ensure_ascii=False)}")
    print(f"分块数量(本次送入): {len(chunks)}")
    print(f"通过校验三元组数量: {len(triples)}")
    print(f"最新日志: {latest_log}")
    print("--- 日志摘要 ---")
    print(
        json.dumps(
            {
                "timestamp": payload.get("timestamp"),
                "source_doc": payload.get("source_doc"),
                "accepted_count": len(payload.get("accepted", [])),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    parser = build_parser()
    cli_args, remaining = parser.parse_known_args()

    if cli_args.run_demo:
        raise SystemExit(run_single_file_demo(cli_args))

    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
