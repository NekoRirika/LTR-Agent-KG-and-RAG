"""Index documents from files/ into the vector store for RAG."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore[import-not-found]
from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from extractor.filereader.read_file import FileReader
from storage.vector_store import VectorStoreManager


logger = logging.getLogger(__name__)


def index_files_dir(
    *,
    files_dir: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    extensions: tuple[str, ...] = (".pdf", ".txt"),
    vector_store_provider: str | None = None,
    vector_store_path: str | None = None,
) -> int:
    """Read documents from files_dir, split into chunks, and index into the vector store.

    Args:
        files_dir: Directory containing source files.
        chunk_size: Chunk size for RecursiveCharacterTextSplitter.
        chunk_overlap: Overlap between chunks.
        extensions: File extensions to process.

    Returns:
        int: Number of indexed chunks.
    """
    reader = FileReader(files_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    docs: list[Document] = []
    for name, content in reader.read_files(extensions=extensions):
        source_path = str((files_dir / name).resolve())
        chunks = splitter.split_text(content)
        for idx, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": source_path,
                        "file_name": name,
                        "chunk_index": idx,
                    },
                )
            )

    manager = VectorStoreManager(
        provider=vector_store_provider,
        persist_directory=vector_store_path,
    )
    manager.add_documents(docs)
    manager.persist()
    logger.info("Indexed %d chunks into provider=%s path=%s", len(docs), manager.provider, manager.persist_path)
    return len(docs)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Index files/ into the RAG vector store.")
    parser.add_argument("--files-dir", default=str(PROJECT_ROOT / "files"), help="Input directory (default: ./files)")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size (default: 500)")
    parser.add_argument("--overlap", type=int, default=50, help="Chunk overlap (default: 50)")
    parser.add_argument(
        "--vector-store-provider",
        default=None,
        help="Vector store provider override (e.g., chroma|faiss). Default: from .env VECTOR_STORE_PROVIDER.",
    )
    parser.add_argument(
        "--vector-store-path",
        default=None,
        help="Vector store path override. Default: from .env VECTOR_STORE_PATH.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = build_parser().parse_args(argv)
    return index_files_dir(
        files_dir=Path(args.files_dir),
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        vector_store_provider=args.vector_store_provider,
        vector_store_path=args.vector_store_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
