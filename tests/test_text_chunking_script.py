from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from extractor.filereader.read_file import FileReader
from extractor.filereader.text_chunker import ChineseTextChunker
from extractor.filereader.text_normalize import normalize_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="文本处理与分块测试脚本")
    parser.add_argument(
        "--file",
        default="files/金庸传.txt",
        help="待测试文件路径，默认 files/金庸传.txt",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="分块长度，默认 500",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="重叠长度，默认 100",
    )
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=120000,
        help="安全分词最大长度，默认 120000",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=3,
        help="展示前 N 个分块，默认 3",
    )
    parser.add_argument(
        "--use-hanlp",
        action="store_true",
        help="启用 HanLP 分词（默认关闭，使用降级分词）",
    )
    parser.add_argument(
        "--force-hanlp",
        action="store_true",
        help="强制使用 HanLP，模型不可用或分词失败时直接报错退出",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式输出结果摘要",
    )
    parser.add_argument(
        "--disable-normalize",
        action="store_true",
        help="关闭规范化处理（默认开启）",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    target_path = (PROJECT_ROOT / args.file).resolve()
    if not target_path.exists():
        files_dir = (PROJECT_ROOT / "files").resolve()
        fallback = None
        if files_dir.exists() and files_dir.is_dir():
            for p in sorted(files_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in FileReader.SUPPORTED_EXTENSIONS:
                    fallback = p
                    break
        if fallback is None:
            print(f"文件不存在: {target_path}")
            return 1
        target_path = fallback

    try:
        reader = FileReader(target_path.parent)
        raw_content = reader.read(target_path)
        if args.disable_normalize:
            content = raw_content
            normalize_stats = {
                "enabled": False,
                "abnormal_space_fixes": 0,
                "noise_lines_removed": 0,
                "footnote_markers_replaced": 0,
            }
        else:
            content, stats = normalize_text(raw_content)
            normalize_stats = {"enabled": True, **stats}

        chunker = ChineseTextChunker(
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            max_text_length=args.max_text_length,
            use_hanlp=(args.use_hanlp or args.force_hanlp),
            force_hanlp=args.force_hanlp,
        )
        chunks = chunker.chunk_text(content)
        stats = chunker.get_text_stats(content)

        summary = {
            "file": str(target_path.relative_to(PROJECT_ROOT)),
            "raw_content_length": len(raw_content),
            "content_length": len(content),
            "chunk_count": len(chunks),
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "use_hanlp": (args.use_hanlp or args.force_hanlp),
            "force_hanlp": args.force_hanlp,
            "normalization": normalize_stats,
            "text_stats": stats,
        }

        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print("=== 文本处理与分块测试 ===")
            print(f"文件: {summary['file']}")
            print(f"原始长度: {summary['raw_content_length']}")
            print(f"规范化后长度: {summary['content_length']}")
            print(f"分块数量: {summary['chunk_count']}")
            print(f"分块参数: chunk_size={args.chunk_size}, overlap={args.overlap}")
            print(
                f"HanLP: {'强制开启' if args.force_hanlp else ('开启' if args.use_hanlp else '关闭')}"
            )
            print("--- 规范化统计 ---")
            print(json.dumps(normalize_stats, ensure_ascii=False, indent=2))
            print("--- 文本统计 ---")
            print(json.dumps(stats, ensure_ascii=False, indent=2))

            preview_count = min(max(args.preview, 0), len(chunks))
            print(f"--- 分块预览（前 {preview_count} 块）---")
            for index in range(preview_count):
                chunk_text = chunks[index].replace("\n", " ").strip()
                preview = chunk_text[:-1]
                print(f"[{index + 1}] len={len(chunks[index])} | {preview}")

        return 0
    except Exception as exc:
        print(f"处理失败: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


class TestTextNormalize(unittest.TestCase):
    def test_normalize_removes_spaces_noise_and_footnotes(self) -> None:
        sample = (
            "二是研究将人工智能运 用于学术出版。\n"
            "有些研究视角相对单 一，未从更加宏观、系统视角探究。\n"
            "出版观察  2026.3 ＿ 71\n"
            "部分学者重点关注版权归属问题④。"
        )

        normalized, stats = normalize_text(sample)

        self.assertNotIn("运 用于", normalized)
        self.assertNotIn("单 一", normalized)
        self.assertNotIn("出版观察  2026.3 ＿ 71", normalized)
        self.assertNotIn("④", normalized)
        self.assertIn("[ref_4]", normalized)
        self.assertGreater(stats["abnormal_space_fixes"], 0)
        self.assertGreater(stats["noise_lines_removed"], 0)
        self.assertGreater(stats["footnote_markers_replaced"], 0)
