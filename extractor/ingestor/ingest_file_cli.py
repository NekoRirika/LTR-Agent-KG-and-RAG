"""Ingest extracted triples JSON into Neo4j graph database."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from extractor.ingestor.connection import IngestorConnectionConfig
from extractor.ingestor.langchain_extractor import Evidence, ExtractedTriple
from extractor.ingestor.neo4j_store import Neo4jKGStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从 triples JSON 入库到 Neo4j")
    parser.add_argument(
        "--input-json",
        default="",
        help="提取结果 JSON 路径；传入时只入库该文件",
    )
    parser.add_argument(
        "--output-dir",
        default="extractor/ingestor/output",
        help="提取结果目录（不传 --input-json 时会入库目录下全部有效文件）",
    )
    return parser


def _read_and_validate_payload(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 格式错误: {path}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"JSON 根节点必须是对象: {path}")

    triples = payload.get("triples")
    if not isinstance(triples, list):
        raise ValueError(f"缺少 triples 数组: {path}")

    return payload


def _load_input_payloads(input_json: str, output_dir: str) -> list[tuple[Path, dict]]:
    if input_json:
        target = (PROJECT_ROOT / input_json).resolve()
        if target.exists():
            return [(target, _read_and_validate_payload(target))]

        target = Path(input_json).resolve()
        if target.exists():
            return [(target, _read_and_validate_payload(target))]

        raise FileNotFoundError(f"找不到输入 JSON: {input_json}")

    output_path = (PROJECT_ROOT / output_dir).resolve()
    if not output_path.exists():
        raise FileNotFoundError(f"输出目录不存在: {output_path}")

    candidates = sorted(
        output_path.glob("triples_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"目录下没有 triples_*.json: {output_path}")

    result: list[tuple[Path, dict]] = []
    for candidate in candidates:
        try:
            payload = _read_and_validate_payload(candidate)
            result.append((candidate, payload))
        except ValueError:
            continue

    if not result:
        raise FileNotFoundError(f"目录下没有可读取的有效 triples_*.json: {output_path}")

    return result


def _to_extracted_triples(payload: dict) -> list[ExtractedTriple]:
    triples = payload.get("triples", [])
    result: list[ExtractedTriple] = []
    for item in triples:
        if not isinstance(item, dict):
            continue
        evidence_raw = (
            item.get("evidence", {}) if isinstance(item.get("evidence"), dict) else {}
        )
        evidence = Evidence(
            source_doc=str(
                evidence_raw.get("source_doc", payload.get("source_doc", ""))
            ),
            source_span=str(evidence_raw.get("source_span", "")),
            section=str(evidence_raw.get("section", payload.get("section", ""))),
            confidence=float(evidence_raw.get("confidence", 0.0)),
            time_or_policy_version=str(
                evidence_raw.get(
                    "time_or_policy_version",
                    payload.get("time_or_policy_version", ""),
                )
            ),
        )
        result.append(
            ExtractedTriple(
                subject=str(item.get("subject", "")),
                subject_type=str(item.get("subject_type", "")),
                relation=str(item.get("relation", "")),
                object=str(item.get("object", "")),
                object_type=str(item.get("object_type", "")),
                evidence=evidence,
            )
        )
    return result


def run_ingest(args: argparse.Namespace) -> int:
    payload_items = _load_input_payloads(args.input_json, args.output_dir)

    mode = "single-file" if args.input_json else "all-files"
    print("=== 入库任务开始 ===")
    print(f"mode: {mode}")
    print(f"candidate_files: {len(payload_items)}")

    config = IngestorConnectionConfig.from_env(PROJECT_ROOT / ".env")
    store = Neo4jKGStore(config)
    total_rows = 0
    processed_files = 0
    skipped_files = 0
    source_docs: set[str] = set()

    try:
        store.ensure_schema()
        for input_path, payload in payload_items:
            triples = _to_extracted_triples(payload)
            if not triples:
                skipped_files += 1
                print(f"skip_empty: {input_path}")
                continue

            count = store.upsert_triples(triples)
            total_rows += count
            processed_files += 1
            source_docs.update(
                item.evidence.source_doc for item in triples if item.evidence.source_doc
            )
            print(f"ingested: {input_path} -> rows={count}")
    finally:
        store.close()

    sorted_docs = sorted(source_docs)

    print("=== 入库完成 ===")
    print(f"processed_files: {processed_files}")
    print(f"skipped_files: {skipped_files}")
    print(f"triple_count: {total_rows}")
    print(f"source_docs: {json.dumps(sorted_docs, ensure_ascii=False)}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_ingest(args)


if __name__ == "__main__":
    raise SystemExit(main())
