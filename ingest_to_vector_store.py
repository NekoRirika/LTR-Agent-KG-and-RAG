"""将 files/ 目录下的文档分块后写入向量数据库。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from extractor.filereader.document_processor import DocumentProcessor
from storage.vector_store import VectorStoreManager


def ingest(
    files_dir: str | Path = "files",
    chunk_size: int = 500,
    overlap: int = 100,
    collection_name: str = "rag_documents",
) -> None:
    files_path = Path(files_dir)
    if not files_path.is_absolute():
        files_path = (PROJECT_ROOT / files_path).resolve()

    print(f"读取目录: {files_path}")
    processor = DocumentProcessor(files_path, chunk_size=chunk_size, overlap=overlap)
    results = processor.process_directory(extensions=["pdf", "docx", "txt", "md"])

    docs: list[Document] = []
    for item in results:
        if item["error"]:
            print(f"  [跳过] {item['file_name']}: {item['error']}")
            continue

        for i, chunk in enumerate(item["chunks"]):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "file_name": item["file_name"],
                        "file_path": item["file_path"],
                        "file_type": item["file_type"],
                        "chunk_index": i,
                        "chunk_count": item["chunk_count"],
                        "source": item["file_name"],
                    },
                )
            )
        print(f"  [OK] {item['file_name']}: {item['chunk_count']} 个分块")

    if not docs:
        print("没有可写入的文档，退出。")
        return

    print(f"\n共 {len(docs)} 个分块，写入向量数据库...")
    vsm = VectorStoreManager(collection_name=collection_name)
    ids = vsm.add_documents(docs)
    vsm.persist()
    print(f"写入完成，共 {len(ids)} 条记录。向量库路径: {vsm.persist_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="将 files/ 目录文档写入向量数据库")
    parser.add_argument("--files-dir", default="files", help="文档目录（默认 files/）")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--collection", default="rag_documents", help="Chroma collection 名称")
    args = parser.parse_args()

    ingest(
        files_dir=args.files_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        collection_name=args.collection,
    )


if __name__ == "__main__":
    main()
