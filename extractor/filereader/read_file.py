"""
文件功能：
- 提供统一的多格式文件读取能力，屏蔽不同文件类型的解析差异。

实现细节：
- 通过 FileReader.read() 按扩展名分发到对应读取方法。
- 支持 txt/md、pdf、doc/docx、epub、csv、json、yaml/yml。
- 文本类文件优先做编码检测（chardet），并保留回退策略。
- 第三方依赖使用延迟导入，避免未安装时影响其他格式读取。
- read_files() 支持目录批量读取并跳过异常文件，便于上层批处理。
- read_csv_as_dict() 提供结构化 CSV 输出，供后续数据处理直接使用。
"""

from __future__ import annotations

import csv
import importlib
import json
import subprocess
from pathlib import Path
from typing import Iterable


class FileReader:
    """
    文件读取器，支持多种文件格式：
    - TXT (文本文件)
    - PDF (PDF文档)
    - MD (Markdown文件)
    - DOCX / DOC (Word文档)
    - EPUB (电子书)
    - CSV (CSV文件)
    - JSON (JSON文件)
    - YAML/YML (YAML文件)
    """

    SUPPORTED_EXTENSIONS = {
        ".txt",
        ".pdf",
        ".md",
        ".docx",
        ".doc",
        ".epub",
        ".csv",
        ".json",
        ".yaml",
        ".yml",
    }

    def __init__(self, directory_path: str | Path):
        self.directory_path = Path(directory_path)
        if not self.directory_path.exists():
            raise FileNotFoundError(f"目录不存在: {self.directory_path}")
        if not self.directory_path.is_dir():
            raise NotADirectoryError(f"不是目录: {self.directory_path}")

    def read(self, file_path: str | Path) -> str:
        """读取单个文件并统一返回字符串内容。"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        extension = path.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"不支持的文件类型: {extension}. 支持类型: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        if extension in {".txt", ".md"}:
            return self._read_text(path)
        if extension == ".pdf":
            return self._read_pdf(path)
        if extension == ".docx":
            return self._read_docx(path)
        if extension == ".doc":
            return self._read_doc(path)
        if extension == ".epub":
            return self._read_epub(path)
        if extension == ".csv":
            return self._read_csv(path)
        if extension == ".json":
            return self._read_json(path)
        if extension in {".yaml", ".yml"}:
            return self._read_yaml(path)

        raise ValueError(f"未处理的文件类型: {extension}")

    def read_files(
        self, extensions: Iterable[str] | None = None
    ) -> list[tuple[str, str]]:
        """按扩展名批量读取目录中的文件，返回(文件名, 内容)列表。"""
        normalized_extensions = self._normalize_extensions(extensions)
        result: list[tuple[str, str]] = []

        for file_path in sorted(self.directory_path.iterdir()):
            if (
                not file_path.is_file()
                or file_path.suffix.lower() not in normalized_extensions
            ):
                continue
            try:
                content = self.read(file_path)
                result.append((file_path.name, content))
            except Exception:
                # 批处理时跳过异常文件，错误由上层汇总。
                continue
        return result

    def read_csv_as_dict(self, file_path: str | Path) -> list[dict[str, str]]:
        """专用方法：将 CSV 读取为字典列表。"""
        path = Path(file_path)
        encoding = self._detect_encoding(path)
        with path.open("r", encoding=encoding, newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            return [dict(row) for row in reader]

    def _normalize_extensions(self, extensions: Iterable[str] | None) -> set[str]:
        if extensions is None:
            return set(self.SUPPORTED_EXTENSIONS)

        normalized: set[str] = set()
        for extension in extensions:
            ext = extension.lower().strip()
            if not ext.startswith("."):
                ext = f".{ext}"
            if ext in self.SUPPORTED_EXTENSIONS:
                normalized.add(ext)
        return normalized

    @staticmethod
    def _detect_encoding(path: Path) -> str:
        try:
            chardet = importlib.import_module("chardet")
            raw_bytes = path.read_bytes()
            detected = chardet.detect(raw_bytes)
            if detected.get("encoding"):
                return str(detected["encoding"])
        except Exception:
            pass
        return "utf-8"

    def _read_text(self, path: Path) -> str:
        encoding_candidates = [self._detect_encoding(path), "utf-8", "gbk", "latin-1"]
        tried: set[str] = set()
        for encoding in encoding_candidates:
            if encoding in tried:
                continue
            tried.add(encoding)
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError("unknown", b"", 0, 1, f"无法解码文件: {path}")

    @staticmethod
    def _read_pdf(path: Path) -> str:
        try:
            pdf_module = importlib.import_module("pypdf")
        except ImportError as exc:
            raise ImportError(
                "缺少依赖 pypdf，请先安装 requirement.txt 中的依赖"
            ) from exc

        reader = pdf_module.PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages).strip()

    @staticmethod
    def _read_docx(path: Path) -> str:
        try:
            docx_module = importlib.import_module("docx")
        except ImportError as exc:
            raise ImportError(
                "缺少依赖 python-docx，请先安装 requirement.txt 中的依赖"
            ) from exc

        document = docx_module.Document(str(path))
        paragraphs = [
            paragraph.text for paragraph in document.paragraphs if paragraph.text
        ]
        return "\n".join(paragraphs).strip()

    def _read_doc(self, path: Path) -> str:
        # 旧版 .doc 优先尝试系统 antiword，便于在无 Office 环境下运行。
        try:
            output = subprocess.run(
                ["antiword", str(path)],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if output.stdout.strip():
                return output.stdout.strip()
        except Exception:
            pass

        raise RuntimeError(
            "读取 .doc 失败：请安装 antiword，或将 .doc 转换为 .docx 后再处理"
        )

    @staticmethod
    def _read_epub(path: Path) -> str:
        try:
            ebooklib_package = importlib.import_module("ebooklib")
            ebooklib_module = importlib.import_module("ebooklib.epub")
        except ImportError as exc:
            raise ImportError(
                "缺少依赖 ebooklib，请先安装 requirement.txt 中的依赖"
            ) from exc

        try:
            bs4_module = importlib.import_module("bs4")
        except ImportError as exc:
            raise ImportError(
                "缺少依赖 beautifulsoup4，请先安装 requirement.txt 中的依赖"
            ) from exc

        book = ebooklib_module.read_epub(str(path))
        paragraphs: list[str] = []

        for item in book.get_items_of_type(ebooklib_package.ITEM_DOCUMENT):
            soup = bs4_module.BeautifulSoup(item.get_body_content(), "html.parser")
            text = soup.get_text("\n", strip=True)
            if text:
                paragraphs.append(text)

        return "\n\n".join(paragraphs).strip()

    def _read_csv(self, path: Path) -> str:
        rows: list[str] = []
        encoding = self._detect_encoding(path)
        with path.open("r", encoding=encoding, newline="") as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                rows.append("\t".join(cell.strip() for cell in row))
        return "\n".join(rows).strip()

    @staticmethod
    def _read_json(path: Path) -> str:
        with path.open("r", encoding="utf-8") as json_file:
            data = json.load(json_file)
        return json.dumps(data, ensure_ascii=False, indent=2)

    @staticmethod
    def _read_yaml(path: Path) -> str:
        try:
            yaml_module = importlib.import_module("yaml")
        except ImportError as exc:
            raise ImportError(
                "缺少依赖 PyYAML，请先安装 requirement.txt 中的依赖"
            ) from exc

        with path.open("r", encoding="utf-8") as yaml_file:
            data = yaml_module.safe_load(yaml_file)
        return yaml_module.safe_dump(data, allow_unicode=True, sort_keys=False)
