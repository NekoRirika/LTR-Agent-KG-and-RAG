from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from extractor.filereader.document_processor import DocumentProcessor
from extractor.ingestor.connection import IngestorConnectionConfig
from extractor.ingestor.langchain_extractor import LangChainKGExtractor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="提取 files 中单个文件并生成日志（不入库）"
    )
    parser.add_argument(
        "--file",
        default="files/AIGC时代的学术出版伦理：风险挑战与治理路径_葛建平.pdf",
        help="待提取文件路径（相对项目根目录）",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=10,
        help="最多抽取前 N 个分块，默认 1",
    )
    parser.add_argument(
        "--log-dir",
        default="extractor/ingestor/log",
        help="日志输出目录，默认 extractor/ingestor/log",
    )
    return parser


def run_demo(file_path: Path, max_chunks: int, log_dir: Path) -> int:
    if not file_path.exists():
        print(f"文件不存在: {file_path}")
        return 1

    config = IngestorConnectionConfig.from_env(PROJECT_ROOT / ".env")

    processor = DocumentProcessor(
        directory_path=file_path.parent,
        use_hanlp=False,
        enable_normalize=True,
    )
    processed = processor.process_file(file_path)
    if processed.get("error"):
        print(f"文件处理失败: {processed['error']}")
        return 1

    normalize_stats = processed.get("normalize_stats", {})
    chunks = processed.get("chunks", [])
    chunks = chunks[: max(max_chunks, 1)]

    extractor = LangChainKGExtractor(config=config, log_dir=log_dir)
    triples = extractor.extract_from_chunks(
        chunks=chunks,
        source_doc=file_path.name,
        section="单文件演示",
        time_or_policy_version="2026",
    )

    logs = sorted(log_dir.glob("extract_*.json"), key=lambda p: p.stat().st_mtime)
    if not logs:
        print("未生成日志文件")
        return 1

    latest_log = logs[-1]
    payload = json.loads(latest_log.read_text(encoding="utf-8"))

    print("=== 提取完成（未入库）===")
    print(f"文件: {file_path.name}")
    print(f"规范化统计: {json.dumps(normalize_stats, ensure_ascii=False)}")
    print(f"送入分块数: {len(chunks)}")
    print(f"通过校验三元组数: {len(triples)}")
    print(f"日志文件: {latest_log}")
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


def main() -> int:
    args = build_parser().parse_args()
    file_path = (PROJECT_ROOT / args.file).resolve()
    log_dir = (PROJECT_ROOT / args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    return run_demo(file_path, args.max_chunks, log_dir)


if __name__ == "__main__":
    raise SystemExit(main())
