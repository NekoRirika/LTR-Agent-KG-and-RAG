"""
文件功能：
- 组合文件读取与文本分块能力，提供目录级文档处理入口。

实现细节：
- 使用 FileReader 负责多格式读取，使用 ChineseTextChunker 负责分块。
- process_directory() 遍历目录并输出统一结构：文件信息、原文、分块、错误状态。
- 出错时不抛出中断异常，而是记录到每个文件的 error 字段，便于批处理容错。
- get_file_stats() 基于处理结果汇总统计，包含成功/失败数量、类型分布与长度指标。
- 支持扩展名过滤，允许上层按任务只处理指定文件类型。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .read_file import FileReader
from .text_chunker import ChineseTextChunker
from .text_normalize import normalize_text


class DocumentProcessor:
    """整合文件读取与中文分块的文档处理器。"""

    def __init__(
        self,
        directory_path: str | Path,
        chunk_size: int = 500,
        overlap: int = 100,
        use_hanlp: bool = False,
        enable_normalize: bool = True,
    ):
        self.directory_path = Path(directory_path)
        self.file_reader = FileReader(self.directory_path)
        self.enable_normalize = enable_normalize
        self.chunker = ChineseTextChunker(
            chunk_size=chunk_size, overlap=overlap, use_hanlp=use_hanlp
        )

    def process_directory(
        self, extensions: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """处理目录下的支持文件，返回每个文件的处理结果。"""
        results: list[dict[str, Any]] = []
        normalized_extensions = self._normalize_extensions(extensions)

        for file_path in sorted(self.directory_path.iterdir()):
            if (
                not file_path.is_file()
                or file_path.suffix.lower() not in normalized_extensions
            ):
                continue

            results.append(self._process_one_file(file_path))

        return results

    def process_file(self, file_path: str | Path) -> dict[str, Any]:
        """处理单个文件并返回统一结果结构。"""
        path = Path(file_path)
        if not path.is_absolute():
            path = (self.directory_path / path).resolve()

        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"文件不存在: {path}")

        if path.suffix.lower() not in self.file_reader.SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支持的文件类型: {path.suffix.lower()}")

        return self._process_one_file(path)

    def _process_one_file(self, file_path: Path) -> dict[str, Any]:
        try:
            raw_content = self.file_reader.read(file_path)
            if self.enable_normalize:
                content, normalize_stats = normalize_text(raw_content)
            else:
                content = raw_content
                normalize_stats = {
                    "abnormal_space_fixes": 0,
                    "noise_lines_removed": 0,
                    "footnote_markers_replaced": 0,
                }
            chunks = self.chunker.chunk_text(content)
            return {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "file_type": file_path.suffix.lower(),
                "raw_content_length": len(raw_content),
                "content_length": len(content),
                "chunk_count": len(chunks),
                "normalization_enabled": self.enable_normalize,
                "normalize_stats": normalize_stats,
                "content": content,
                "chunks": chunks,
                "error": None,
            }
        except Exception as exc:
            return {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "file_type": file_path.suffix.lower(),
                "raw_content_length": 0,
                "content_length": 0,
                "chunk_count": 0,
                "normalization_enabled": self.enable_normalize,
                "normalize_stats": {
                    "abnormal_space_fixes": 0,
                    "noise_lines_removed": 0,
                    "footnote_markers_replaced": 0,
                },
                "content": "",
                "chunks": [],
                "error": str(exc),
            }

    def get_file_stats(self, extensions: list[str] | None = None) -> dict[str, Any]:
        """获取目录统计信息，包括类型分布、总长度和均值。"""
        results = self.process_directory(extensions=extensions)
        success_results = [item for item in results if not item["error"]]

        type_distribution = Counter(item["file_type"] for item in success_results)
        total_content_length = sum(item["content_length"] for item in success_results)
        total_chunks = sum(item["chunk_count"] for item in success_results)
        file_count = len(success_results)

        average_content_length = total_content_length / file_count if file_count else 0
        average_chunk_count = total_chunks / file_count if file_count else 0

        return {
            "directory": str(self.directory_path),
            "total_files": len(results),
            "success_files": file_count,
            "failed_files": len(results) - file_count,
            "file_type_distribution": dict(type_distribution),
            "total_content_length": total_content_length,
            "average_content_length": average_content_length,
            "total_chunks": total_chunks,
            "average_chunk_count": average_chunk_count,
        }

    def _normalize_extensions(self, extensions: list[str] | None) -> set[str]:
        if not extensions:
            return set(self.file_reader.SUPPORTED_EXTENSIONS)

        normalized: set[str] = set()
        for extension in extensions:
            ext = extension.lower().strip()
            if not ext.startswith("."):
                ext = f".{ext}"
            if ext in self.file_reader.SUPPORTED_EXTENSIONS:
                normalized.add(ext)

        return normalized
