"""LangChain-based strict KG extraction with timestamped logging."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

from .connection import IngestorConnectionConfig, build_llm
from .extraction_prompt import build_extraction_prompt
from .kg_extraction_definition import (
    DEFAULT_KG_EXTRACTION_DEFINITION,
    KGExtractionDefinition,
)


@dataclass(frozen=True)
class Evidence:
    source_doc: str
    source_span: str
    section: str
    confidence: float
    time_or_policy_version: str


@dataclass(frozen=True)
class ExtractedTriple:
    subject: str
    subject_type: str
    relation: str
    object: str
    object_type: str
    evidence: Evidence

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()

    first = text.find("{")
    last = text.rfind("}")
    if first < 0 or last < 0:
        raise ValueError("LLM output does not contain a JSON object")
    return json.loads(text[first : last + 1])


def _allowed_types(definition: KGExtractionDefinition) -> tuple[set[str], set[str]]:
    entity_types = {item.name for item in definition.entity_types}
    relation_types = {item.name for item in definition.relation_types}
    return entity_types, relation_types


def _build_relation_constraints(
    definition: KGExtractionDefinition,
) -> dict[str, tuple[set[str], set[str]]]:
    constraints: dict[str, tuple[set[str], set[str]]] = {}
    for relation in definition.relation_types:
        constraints[relation.name] = (
            set(relation.subject_types),
            set(relation.object_types),
        )
    return constraints


def _build_template_index(
    definition: KGExtractionDefinition,
) -> set[tuple[str, str, str]]:
    return {
        (template.subject_type, template.relation, template.object_type)
        for template in definition.triple_templates
    }


def _validate_triple(
    item: dict[str, Any],
    definition: KGExtractionDefinition,
    default_source_doc: str,
    default_section: str,
) -> ExtractedTriple | None:
    entity_types, relation_types = _allowed_types(definition)
    relation_constraints = _build_relation_constraints(definition)
    template_index = _build_template_index(definition)

    subject = str(item.get("subject", "")).strip()
    subject_type = str(item.get("subject_type", "")).strip()
    relation = str(item.get("relation", "")).strip()
    obj = str(item.get("object", "")).strip()
    object_type = str(item.get("object_type", "")).strip()

    if not subject or not obj or not relation:
        return None
    if subject_type not in entity_types or object_type not in entity_types:
        return None
    if relation not in relation_types:
        return None

    relation_subject_types, relation_object_types = relation_constraints.get(
        relation, (set(), set())
    )
    if subject_type not in relation_subject_types:
        return None
    if object_type not in relation_object_types:
        return None

    if (subject_type, relation, object_type) not in template_index:
        return None

    evidence_raw = (
        item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
    )
    confidence_raw = evidence_raw.get("confidence", definition.confidence_threshold)

    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        return None

    if confidence < definition.confidence_threshold or confidence > 1:
        return None

    evidence = Evidence(
        source_doc=str(evidence_raw.get("source_doc", default_source_doc)).strip()
        or default_source_doc,
        source_span=str(evidence_raw.get("source_span", "")).strip(),
        section=str(evidence_raw.get("section", default_section)).strip()
        or default_section,
        confidence=confidence,
        time_or_policy_version=str(
            evidence_raw.get("time_or_policy_version", "")
        ).strip(),
    )

    if not evidence.source_span:
        return None

    return ExtractedTriple(
        subject=subject,
        subject_type=subject_type,
        relation=relation,
        object=obj,
        object_type=object_type,
        evidence=evidence,
    )


class LangChainKGExtractor:
    def __init__(
        self,
        config: IngestorConnectionConfig,
        definition: KGExtractionDefinition = DEFAULT_KG_EXTRACTION_DEFINITION,
        log_dir: str | Path = "extractor/ingestor/log",
    ):
        self.config = config
        self.definition = definition
        self.llm = build_llm(config)
        self.prompt = build_extraction_prompt(definition)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def extract_from_text(
        self,
        text: str,
        source_doc: str,
        section: str = "",
        time_or_policy_version: str = "",
        write_log: bool = True,
    ) -> list[ExtractedTriple]:
        chain = self.prompt | self.llm
        response = chain.invoke(
            {
                "text": text,
                "source_doc": source_doc,
                "section": section,
            }
        )

        content = getattr(response, "content", str(response))
        payload = _parse_json_object(content)
        raw_triples = payload.get("triples", []) if isinstance(payload, dict) else []

        validated: list[ExtractedTriple] = []
        for item in raw_triples:
            if not isinstance(item, dict):
                continue
            triple = _validate_triple(item, self.definition, source_doc, section)
            if triple is None:
                continue
            if not triple.evidence.time_or_policy_version and time_or_policy_version:
                evidence = Evidence(
                    source_doc=triple.evidence.source_doc,
                    source_span=triple.evidence.source_span,
                    section=triple.evidence.section,
                    confidence=triple.evidence.confidence,
                    time_or_policy_version=time_or_policy_version,
                )
                triple = ExtractedTriple(
                    subject=triple.subject,
                    subject_type=triple.subject_type,
                    relation=triple.relation,
                    object=triple.object,
                    object_type=triple.object_type,
                    evidence=evidence,
                )
            validated.append(triple)

        deduplicated = self._deduplicate_exact(validated)
        if write_log:
            self._write_log(
                source_doc=source_doc,
                section=section,
                chunk_count=1,
                total_input_length=len(text),
                unique_triple_count=len(deduplicated),
            )
        return deduplicated

    def extract_from_chunks(
        self,
        chunks: list[str],
        source_doc: str,
        section: str = "",
        time_or_policy_version: str = "",
        show_progress: bool = False,
        workers: int = 4,
        write_log: bool = True,
    ) -> list[ExtractedTriple]:
        all_triples: list[ExtractedTriple] = []
        total_input_length = 0
        valid_chunks = [chunk for chunk in chunks if chunk.strip()]
        total_chunks = len(valid_chunks)
        completed = 0
        # Limit progress refresh frequency to avoid one-line-per-chunk output.
        progress_step = max(1, total_chunks // 20)
        last_reported = -1
        max_workers = max(1, workers)

        if show_progress:
            self._print_progress(0, total_chunks, source_doc)
            last_reported = 0

        if max_workers <= 1:
            for chunk in valid_chunks:
                total_input_length += len(chunk)
                all_triples.extend(
                    self.extract_from_text(
                        text=chunk,
                        source_doc=source_doc,
                        section=section,
                        time_or_policy_version=time_or_policy_version,
                        write_log=False,
                    )
                )
                completed += 1
                if show_progress and (
                    completed == total_chunks
                    or completed - last_reported >= progress_step
                ):
                    self._print_progress(completed, total_chunks, source_doc)
                    last_reported = completed
        else:

            def _extract_one(chunk: str) -> tuple[int, list[ExtractedTriple]]:
                triples = self.extract_from_text(
                    text=chunk,
                    source_doc=source_doc,
                    section=section,
                    time_or_policy_version=time_or_policy_version,
                    write_log=False,
                )
                return len(chunk), triples

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_extract_one, chunk) for chunk in valid_chunks
                ]
                for future in as_completed(futures):
                    try:
                        chunk_length, triples = future.result()
                        total_input_length += chunk_length
                        all_triples.extend(triples)
                    except Exception as exc:
                        # Do not fail whole-file extraction on one chunk error.
                        sys.stderr.write(f"\n[WARN] chunk 抽取失败: {exc}\n")
                        sys.stderr.flush()
                    completed += 1
                    if show_progress and (
                        completed == total_chunks
                        or completed - last_reported >= progress_step
                    ):
                        self._print_progress(completed, total_chunks, source_doc)
                        last_reported = completed

        if show_progress:
            sys.stdout.write("\n")
            sys.stdout.flush()

        deduplicated = self._deduplicate_exact(all_triples)
        if write_log:
            self._write_log(
                source_doc=source_doc,
                section=section,
                chunk_count=total_chunks,
                total_input_length=total_input_length,
                unique_triple_count=len(deduplicated),
            )
        return deduplicated

    @staticmethod
    def _print_progress(current: int, total: int, source_doc: str) -> None:
        if total <= 0:
            return
        safe_total = max(total, 1)
        width = 30
        ratio = min(max(current / safe_total, 0.0), 1.0)
        filled = min(width, int(width * ratio))
        bar = "#" * filled + "-" * (width - filled)
        counter_width = len(str(safe_total))
        if len(source_doc) > 36:
            source_display = f"{source_doc[:16]}...{source_doc[-16:]}"
        else:
            source_display = source_doc
        message = (
            f"\r[{bar}] {current:>{counter_width}}/{safe_total} "
            f"({ratio * 100:5.1f}%) extracting {source_display}"
        )
        sys.stdout.write(message)
        sys.stdout.flush()

    @staticmethod
    def _deduplicate_exact(triples: list[ExtractedTriple]) -> list[ExtractedTriple]:
        seen: set[str] = set()
        unique: list[ExtractedTriple] = []
        for triple in triples:
            key = json.dumps(triple.to_dict(), ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            unique.append(triple)
        return unique

    def _write_log(
        self,
        source_doc: str,
        section: str,
        chunk_count: int,
        total_input_length: int,
        unique_triple_count: int,
    ) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = self.log_dir / f"extract_{ts}.json"
        payload = {
            "timestamp": ts,
            "source_doc": source_doc,
            "section": section,
            "chunk_count": chunk_count,
            "total_input_length": total_input_length,
            "unique_triple_count": unique_triple_count,
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
