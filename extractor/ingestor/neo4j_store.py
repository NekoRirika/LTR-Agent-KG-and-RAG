"""Neo4j storage for extracted triples with merge-safe upsert behavior."""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

from neo4j import GraphDatabase

from .connection import IngestorConnectionConfig
from .langchain_extractor import ExtractedTriple


class Neo4jKGStore:
    def __init__(self, config: IngestorConnectionConfig):
        self._driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_username, config.neo4j_password),
        )
        self._database = config.neo4j_database

    def close(self) -> None:
        self._driver.close()

    def ensure_schema(self) -> None:
        def create_constraints(tx):
            tx.run(
                "CREATE CONSTRAINT entity_name_type_unique IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE"
            )
            tx.run(
                "CREATE CONSTRAINT document_name_unique IF NOT EXISTS "
                "FOR (d:Document) REQUIRE d.name IS UNIQUE"
            )

        with self._driver.session(database=self._database) as session:
            session.execute_write(create_constraints)

    def upsert_triples(self, triples: Iterable[ExtractedTriple]) -> int:
        rows = [asdict(item) for item in triples]
        if not rows:
            return 0

        query = """
        UNWIND $rows AS row
        MERGE (d:Document {name: row.evidence.source_doc})
          ON CREATE SET d.created_at = datetime()
          ON MATCH SET d.updated_at = datetime()
        MERGE (s:Entity {name: row.subject, type: row.subject_type})
          ON CREATE SET s.created_at = datetime()
          ON MATCH SET s.updated_at = datetime()
        MERGE (o:Entity {name: row.object, type: row.object_type})
          ON CREATE SET o.created_at = datetime()
          ON MATCH SET o.updated_at = datetime()
        MERGE (s)-[:MENTIONED_IN]->(d)
        MERGE (o)-[:MENTIONED_IN]->(d)
        MERGE (s)-[r:RELATED_TO {
          relation: row.relation,
          source_doc: row.evidence.source_doc,
          section: row.evidence.section
        }]->(o)
          ON CREATE SET
            r.source_span = row.evidence.source_span,
            r.confidence = row.evidence.confidence,
            r.time_or_policy_version = row.evidence.time_or_policy_version,
            r.created_at = datetime(),
            r.updated_at = datetime()
          ON MATCH SET
            r.source_span = CASE
              WHEN size(r.source_span) >= size(row.evidence.source_span) THEN r.source_span
              ELSE row.evidence.source_span
            END,
            r.confidence = CASE
              WHEN r.confidence >= row.evidence.confidence THEN r.confidence
              ELSE row.evidence.confidence
            END,
            r.time_or_policy_version = CASE
              WHEN row.evidence.time_or_policy_version = '' THEN r.time_or_policy_version
              ELSE row.evidence.time_or_policy_version
            END,
            r.updated_at = datetime()
        """

        with self._driver.session(database=self._database) as session:
            session.run(query, rows=rows)

        return len(rows)
