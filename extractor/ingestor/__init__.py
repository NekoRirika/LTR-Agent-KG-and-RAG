"""Ingestor pipeline exports for extraction, validation, and Neo4j storage."""

from .kg_extraction_definition import (
    DEFAULT_KG_EXTRACTION_DEFINITION,
    EvidenceSchema,
    EntityType,
    KGExtractionDefinition,
    RelationType,
    TripleTemplate,
    build_default_kg_extraction_definition,
)
from .connection import (
    IngestorConnectionConfig,
    ensure_env_keys,
    run_connection_checks,
)
from .langchain_extractor import ExtractedTriple, LangChainKGExtractor
from .neo4j_store import Neo4jKGStore


def run_extract(*args, **kwargs):
    """Lazily import CLI extractor entry to avoid module preloading side effects."""
    from .extract_file_cli import run_extract as _run_extract

    return _run_extract(*args, **kwargs)


def run_ingest(*args, **kwargs):
    """Lazily import CLI ingest entry to avoid module preloading side effects."""
    from .ingest_file_cli import run_ingest as _run_ingest

    return _run_ingest(*args, **kwargs)


__all__ = [
    "DEFAULT_KG_EXTRACTION_DEFINITION",
    "ExtractedTriple",
    "EvidenceSchema",
    "EntityType",
    "IngestorConnectionConfig",
    "KGExtractionDefinition",
    "LangChainKGExtractor",
    "Neo4jKGStore",
    "RelationType",
    "TripleTemplate",
    "build_default_kg_extraction_definition",
    "ensure_env_keys",
    "run_extract",
    "run_connection_checks",
    "run_ingest",
]
