from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from extractor.ingestor.extract_file_cli import run_extract as run_single_extract


PROJECT_ROOT = Path(__file__).resolve().parent
SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
    ".docx",
    ".doc",
    ".epub",
    ".csv",
    ".json",
    ".yaml",
    ".yml",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量提取 files 目录文档（不入库）")
    parser.add_argument(
        "--files-dir",
        default="files",
        help="待提取目录，默认 files",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="每个文件最多送入抽取的分块数；<=0 表示处理全部 chunk，默认 0",
    )
    parser.add_argument(
        "--section",
        default="批量提取",
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
        help="日志输出目录",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="每个文件内 chunk 并发数，默认 4；设为 1 可回退串行",
    )
    parser.add_argument(
        "--flush-chunks",
        type=int,
        default=20,
        help="每处理多少个 chunk 就写入一次结果，默认 20",
    )
    return parser


def run_batch(args: argparse.Namespace) -> int:
    target_dir = (PROJECT_ROOT / args.files_dir).resolve()
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"目录不存在: {target_dir}")
        return 1

    files = [
        path
        for path in sorted(target_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        print(f"目录中无支持文件: {target_dir}")
        return 1

    success = 0
    failed = 0

    total_files = len(files)
    for index, path in enumerate(files, start=1):
        relative_path = path.relative_to(PROJECT_ROOT).as_posix()
        print(f"\n=== Extracting [{index}/{total_files}]: {relative_path} ===")
        single_args = SimpleNamespace(
            file=relative_path,
            max_chunks=args.max_chunks,
            section=args.section,
            time_version=args.time_version,
            output_dir=args.output_dir,
            log_dir=args.log_dir,
            workers=args.workers,
            flush_chunks=args.flush_chunks,
        )
        try:
            rc = run_single_extract(single_args)
            if rc == 0:
                success += 1
            else:
                failed += 1
        except Exception as exc:
            failed += 1
            print(f"提取失败: {relative_path} -> {exc}")

    print("\n=== 批量提取结束 ===")
    print(f"total: {len(files)}")
    print(f"success: {success}")
    print(f"failed: {failed}")

    return 0 if failed == 0 else 2


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
