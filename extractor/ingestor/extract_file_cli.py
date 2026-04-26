"""Extract triples from a single file and persist them to a JSON artifact."""

from __future__ import annotations

import argparse
import json
import os
import math
from datetime import datetime
from pathlib import Path

from extractor.filereader.document_processor import DocumentProcessor
from extractor.ingestor.connection import IngestorConnectionConfig
from extractor.ingestor.langchain_extractor import LangChainKGExtractor


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="单文件提取：输出 triples JSON（不入库）"
    )
    parser.add_argument(
        "--file",
        default="files/AIGC时代的学术出版伦理：风险挑战与治理路径_葛建平.pdf",
        help="待提取文件路径（相对项目根目录）",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="送入抽取的最大分块数；<=0 表示处理全部 chunk，默认 0",
    )
    parser.add_argument(
        "--section",
        default="自动提取",
        help="写入 evidence.section 的值",
    )
    parser.add_argument(
        "--time-version",
        default="2026",
        help="写入 evidence.time_or_policy_version 的值",
    )
    parser.add_argument(
        "--output-dir",
        default="extractor/ingestor/output",
        help="抽取结果 JSON 输出目录",
    )
    parser.add_argument(
        "--log-dir",
        default="extractor/ingestor/log",
        help="抽取原始日志输出目录",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="chunk 并发数，默认 4；设为 1 可回退串行",
    )
    parser.add_argument(
        "--flush-chunks",
        type=int,
        default=20,
        help="每处理多少个 chunk 就写入一次结果，默认 20",
    )
    return parser


def run_extract(args: argparse.Namespace) -> int:
    file_path = (PROJECT_ROOT / args.file).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    config = IngestorConnectionConfig.from_env(PROJECT_ROOT / ".env")
    processor = DocumentProcessor(
        directory_path=file_path.parent,
        use_hanlp=False,
        enable_normalize=True,
    )
    processed = processor.process_file(file_path)
    if processed.get("error"):
        raise RuntimeError(f"文件处理失败: {processed['error']}")

    chunks = processed.get("chunks", [])
    if args.max_chunks > 0:
        chunks = chunks[: args.max_chunks]
    valid_chunks = [chunk for chunk in chunks if chunk.strip()]

    extractor = LangChainKGExtractor(
        config=config,
        log_dir=(PROJECT_ROOT / args.log_dir).resolve(),
    )
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_file = output_dir / f"triples_{file_path.stem}_{ts}.json"
    temp_output_file = output_dir / f"{output_file.name}.tmp"
    flush_chunks = max(1, int(getattr(args, "flush_chunks", 20)))
    total_batches = max(1, math.ceil(len(valid_chunks) / flush_chunks))

    print("=== 提取任务开始 ===")
    print(f"source_doc: {file_path.name}")
    print(f"raw_chunk_count: {len(chunks)}")
    print(f"valid_chunk_count: {len(valid_chunks)}")
    print(f"flush_chunks: {flush_chunks}")
    print(f"batch_count: {total_batches}")
    print(f"workers: {args.workers}")

    seen: set[str] = set()
    written_count = 0
    total_input_length = 0
    first_item = True

    if temp_output_file.exists():
        temp_output_file.unlink()

    try:
        with temp_output_file.open("w", encoding="utf-8") as fp:
            fp.write("{\n")
            fp.write(f'  "timestamp": {json.dumps(ts, ensure_ascii=False)},\n')
            fp.write(
                f'  "source_doc": {json.dumps(file_path.name, ensure_ascii=False)},\n'
            )
            fp.write(
                f'  "source_path": {json.dumps(str(file_path), ensure_ascii=False)},\n'
            )
            fp.write(f'  "section": {json.dumps(args.section, ensure_ascii=False)},\n')
            fp.write(
                "  \"time_or_policy_version\": "
                f"{json.dumps(args.time_version, ensure_ascii=False)},\n"
            )
            fp.write(f'  "chunk_count": {len(valid_chunks)},\n')
            fp.write(
                "  \"normalize_stats\": "
                f"{json.dumps(processed.get('normalize_stats', {}), ensure_ascii=False, indent=2).replace(chr(10), chr(10) + '  ')},\n"
            )
            fp.write('  "triples": [\n')

            for batch_index, start in enumerate(
                range(0, len(valid_chunks), flush_chunks), start=1
            ):
                batch = valid_chunks[start : start + flush_chunks]
                end = start + len(batch)
                print(
                    f"\n[batch {batch_index}/{total_batches}] "
                    f"chunks {start + 1}-{end}/{len(valid_chunks)}"
                )
                total_input_length += sum(len(chunk) for chunk in batch)
                batch_triples = extractor.extract_from_chunks(
                    chunks=batch,
                    source_doc=file_path.name,
                    section=args.section,
                    time_or_policy_version=args.time_version,
                    show_progress=True,
                    workers=args.workers,
                    write_log=False,
                )

                for triple in batch_triples:
                    triple_dict = triple.to_dict()
                    triple_key = json.dumps(
                        triple_dict, ensure_ascii=False, sort_keys=True
                    )
                    if triple_key in seen:
                        continue
                    seen.add(triple_key)

                    if not first_item:
                        fp.write(",\n")
                    fp.write(
                        "    "
                        + json.dumps(
                            triple_dict,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )
                    first_item = False
                    written_count += 1

            fp.write("\n  ]\n")
            fp.write("}\n")
            fp.flush()
            os.fsync(fp.fileno())

        # Atomic rename on the same filesystem to avoid half-written final JSON.
        temp_output_file.replace(output_file)
    except Exception:
        if temp_output_file.exists():
            temp_output_file.unlink()
        raise

    extractor._write_log(
        source_doc=file_path.name,
        section=args.section,
        chunk_count=len(valid_chunks),
        total_input_length=total_input_length,
        unique_triple_count=written_count,
    )

    print("=== 提取完成（未入库）===")
    print(f"source_doc: {file_path.name}")
    print(f"chunk_count: {len(valid_chunks)}")
    print(f"triple_count: {written_count}")
    print(f"triples_json: {output_file}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_extract(args)


if __name__ == "__main__":
    raise SystemExit(main())
