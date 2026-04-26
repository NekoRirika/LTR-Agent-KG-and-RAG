"""
文件功能：
- 对 files 目录中的真实样本文件执行端到端读取与处理验证。

实现细节：
- 覆盖单文件读取、批量读取、目录处理、统计结果和端到端实际校验。
- 测试时对比真实文件数量与名称集合，确保没有漏处理或多处理。
- 断言每个样本文件内容非空、分块数量有效，并输出明确失败原因。
- 提供 CLI 入口：支持查看依赖版本、调整测试输出级别、failfast 模式。
- 可直接使用 python tests/test_file_reader.py 在命令行运行。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import traceback
import unittest
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from extractor.filereader.read_file import FileReader
from extractor.filereader.document_processor import DocumentProcessor


class TestFileReader(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.files_dir = self.project_root / "files"
        self.reader = FileReader(self.files_dir)

    def _expected_supported_files(self) -> list[Path]:
        return [
            path
            for path in sorted(self.files_dir.iterdir())
            if path.is_file()
            and path.suffix.lower() in self.reader.SUPPORTED_EXTENSIONS
        ]

    def test_read_all_files_in_files_directory(self) -> None:
        supported_files = self._expected_supported_files()

        self.assertGreater(
            len(supported_files), 0, "files 目录下没有可测试的支持格式文件"
        )

        for file_path in supported_files:
            with self.subTest(file=file_path.name):
                content = self.reader.read(file_path)
                self.assertIsInstance(content, str)
                self.assertTrue(content.strip(), f"文件读取结果为空: {file_path.name}")

    def test_batch_read_files(self) -> None:
        expected_files = self._expected_supported_files()
        expected_names = {path.name for path in expected_files}

        results = self.reader.read_files()
        self.assertEqual(
            len(results),
            len(expected_files),
            "批量读取结果数量与 files 目录中的真实文件数量不一致",
        )

        result_names = {name for name, _ in results}
        self.assertSetEqual(
            result_names,
            expected_names,
            f"批量读取文件名不匹配，缺失或多余文件: {expected_names ^ result_names}",
        )

        for name, content in results:
            with self.subTest(file=name):
                self.assertIsInstance(content, str)
                self.assertTrue(content.strip(), f"批量读取内容为空: {name}")

    def test_document_processor_can_process_files_directory(self) -> None:
        expected_names = {path.name for path in self._expected_supported_files()}

        processor = DocumentProcessor(self.files_dir, use_hanlp=False)
        results = processor.process_directory()
        self.assertEqual(
            len(results),
            len(expected_names),
            "DocumentProcessor 处理文件数量与真实文件数量不一致",
        )

        result_names = {item["file_name"] for item in results}
        self.assertSetEqual(
            result_names,
            expected_names,
            f"DocumentProcessor 输出文件名不匹配，差异: {expected_names ^ result_names}",
        )

        for item in results:
            with self.subTest(file=item["file_name"]):
                self.assertIn("file_name", item)
                self.assertIn("content", item)
                self.assertIn("chunks", item)
                self.assertIsNone(
                    item["error"],
                    f"文件处理失败: {item['file_name']} -> {item['error']}",
                )
                self.assertTrue(
                    item["content"].strip(), f"处理结果内容为空: {item['file_name']}"
                )
                self.assertGreaterEqual(item["chunk_count"], 1)
                self.assertGreaterEqual(
                    item["content_length"], len(item["content"]) - 5
                )

    def test_document_processor_stats(self) -> None:
        expected_count = len(self._expected_supported_files())

        processor = DocumentProcessor(self.files_dir, use_hanlp=False)
        stats = processor.get_file_stats()

        self.assertIn("total_files", stats)
        self.assertIn("success_files", stats)
        self.assertIn("failed_files", stats)
        self.assertIn("file_type_distribution", stats)
        self.assertEqual(stats["total_files"], expected_count)
        self.assertEqual(stats["failed_files"], 0)
        self.assertEqual(stats["success_files"], expected_count)

    def test_actual_files_are_processed_end_to_end(self) -> None:
        """实际验证 files 目录中的每个真实文件都可以读取并分块。"""
        expected_files = self._expected_supported_files()
        self.assertGreater(len(expected_files), 0, "files 目录没有可处理的真实文件")

        processor = DocumentProcessor(self.files_dir, use_hanlp=False)
        processed = {item["file_name"]: item for item in processor.process_directory()}

        failed_details: list[str] = []
        for file_path in expected_files:
            item = processed.get(file_path.name)
            if item is None:
                failed_details.append(f"缺少处理结果: {file_path.name}")
                continue

            if item.get("error"):
                failed_details.append(f"处理失败: {file_path.name} -> {item['error']}")
                continue

            if not str(item.get("content", "")).strip():
                failed_details.append(f"内容为空: {file_path.name}")

            if int(item.get("chunk_count", 0)) <= 0:
                failed_details.append(f"分块数量异常: {file_path.name}")

        self.assertFalse(
            failed_details,
            "真实文件处理校验失败:\n" + "\n".join(failed_details),
        )


def _get_package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not-installed"


def print_runtime_versions() -> None:
    print("=== Runtime Versions ===")
    print(f"python: {platform.python_version()}")
    print(f"platform: {platform.platform()}")
    print(f"pypdf: {_get_package_version('pypdf')}")
    print(f"python-docx: {_get_package_version('python-docx')}")
    print(f"PyYAML: {_get_package_version('PyYAML')}")
    print(f"chardet: {_get_package_version('chardet')}")
    print(f"hanlp: {_get_package_version('hanlp')}")
    print(f"ebooklib: {_get_package_version('ebooklib')}")
    print(f"beautifulsoup4: {_get_package_version('beautifulsoup4')}")


def run_hanlp_healthcheck(output_path: Path) -> int:
    """执行 HanLP 运行检测并将结果写入文件。"""
    result: dict[str, object] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hanlp_installed": False,
        "hanlp_version": _get_package_version("hanlp"),
        "model_loaded": False,
        "tokenize_ok": False,
        "sample_text": "今天天气很好，我们测试HanLP是否可用。",
        "token_count": 0,
        "tokens_preview": [],
        "error": None,
        "traceback": None,
    }

    try:
        import hanlp  # type: ignore[import-not-found]

        result["hanlp_installed"] = True
        tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        result["model_loaded"] = True

        tokens = tokenizer(result["sample_text"])
        result["tokenize_ok"] = bool(tokens)
        result["token_count"] = len(tokens)
        result["tokens_preview"] = tokens[:20]
    except Exception as exc:
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"HanLP检测结果已写入: {output_path}")
    return 0 if result["tokenize_ok"] else 1


def run_cli() -> int:
    parser = argparse.ArgumentParser(description="文件读取测试脚本")
    parser.add_argument(
        "--versions",
        action="store_true",
        help="仅打印运行环境和依赖版本",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="测试输出详细级别，默认 2",
    )
    parser.add_argument(
        "--failfast",
        action="store_true",
        help="遇到首个失败即停止",
    )
    parser.add_argument(
        "--check-hanlp",
        action="store_true",
        help="执行HanLP运行检测并将结果写入文件",
    )
    parser.add_argument(
        "--hanlp-output",
        default="tests/hanlp_healthcheck.json",
        help="HanLP检测结果输出文件路径，默认 tests/hanlp_healthcheck.json",
    )
    args = parser.parse_args()

    if args.versions:
        print_runtime_versions()
        return 0

    if args.check_hanlp:
        return run_hanlp_healthcheck(PROJECT_ROOT / args.hanlp_output)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestFileReader)
    runner = unittest.TextTestRunner(verbosity=args.verbosity, failfast=args.failfast)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(run_cli())
