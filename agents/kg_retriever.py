"""Neo4j 只读查询层，供 agent 搜索节点调用。"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from neo4j import GraphDatabase

from agents.__init__ import AgentNeo4jConfig


class KGRetriever:
    """对 Neo4j 知识图谱执行只读查询。"""

    def __init__(self, config: AgentNeo4jConfig):
        self._driver = GraphDatabase.driver(
            config.uri,
            auth=(config.username, config.password),
        )
        self._database = config.database

    def close(self) -> None:
        self._driver.close()

    # ── 实体查询 ──────────────────────────────────────────────────────────────

    def find_entities_by_name(self, name: str, limit: int = 10) -> list[dict]:
        """按名称模糊匹配实体。"""
        query = """
        MATCH (e:Entity)
        WHERE e.name CONTAINS $name
        RETURN e.name AS name, e.type AS type
        LIMIT $limit
        """
        return self._run_query(query, name=name, limit=limit)

    def find_entities_by_type(self, entity_type: str, limit: int = 100) -> list[dict]:
        """按类型查找实体。"""
        query = """
        MATCH (e:Entity {type: $entity_type})
        RETURN e.name AS name, e.type AS type
        LIMIT $limit
        """
        return self._run_query(query, entity_type=entity_type, limit=limit)

    # ── 邻域查询 ──────────────────────────────────────────────────────────────

    def get_entity_neighborhood(
        self,
        entity_name: str,
        max_hops: int = 1,
        neighbor_limit: int = 50,
        min_confidence: float = 0.75,
        two_hop_confidence: float = 0.8,
    ) -> list[dict]:
        """获取实体邻域（出边+入边，可选2跳）。"""
        # 1跳出站
        out_query = """
        MATCH (s:Entity)-[r:RELATED_TO]->(o:Entity)
        WHERE s.name CONTAINS $entity_name AND r.confidence >= $min_confidence
        RETURN s.name AS subject, s.type AS subject_type,
               r.relation AS relation, r.source_span AS source_span,
               r.source_doc AS source_doc, r.confidence AS confidence,
               o.name AS object, o.type AS object_type
        ORDER BY r.confidence DESC
        LIMIT $limit
        """
        results = self._run_query(
            out_query, entity_name=entity_name,
            min_confidence=min_confidence, limit=neighbor_limit,
        )

        # 1跳入站
        in_query = """
        MATCH (s:Entity)-[r:RELATED_TO]->(o:Entity)
        WHERE o.name CONTAINS $entity_name AND r.confidence >= $min_confidence
        RETURN s.name AS subject, s.type AS subject_type,
               r.relation AS relation, r.source_span AS source_span,
               r.source_doc AS source_doc, r.confidence AS confidence,
               o.name AS object, o.type AS object_type
        ORDER BY r.confidence DESC
        LIMIT $limit
        """
        results.extend(self._run_query(
            in_query, entity_name=entity_name,
            min_confidence=min_confidence, limit=neighbor_limit,
        ))

        # 2跳（仅高置信度路径）
        if max_hops >= 2:
            two_hop_query = """
            MATCH (s:Entity)-[r1:RELATED_TO]->(mid:Entity)-[r2:RELATED_TO]->(o:Entity)
            WHERE s.name CONTAINS $entity_name
              AND r1.confidence >= $two_hop_confidence
              AND r2.confidence >= $two_hop_confidence
            RETURN s.name AS subject, s.type AS subject_type,
                   r1.relation + ' -> ' + r2.relation AS relation,
                   r1.source_span + ' ; ' + r2.source_span AS source_span,
                   r1.source_doc AS source_doc,
                   CASE WHEN r1.confidence < r2.confidence THEN r1.confidence
                        ELSE r2.confidence END AS confidence,
                   o.name AS object, o.type AS object_type
            LIMIT $limit
            """
            results.extend(self._run_query(
                two_hop_query, entity_name=entity_name,
                two_hop_confidence=two_hop_confidence, limit=neighbor_limit,
            ))

        return results

    # ── 关系查询 ──────────────────────────────────────────────────────────────

    def get_relations_by_type(
        self, relation: str, min_confidence: float = 0.75, limit: int = 100
    ) -> list[dict]:
        """按关系类型查询三元组。"""
        query = """
        MATCH (s:Entity)-[r:RELATED_TO]->(o:Entity)
        WHERE r.relation = $relation AND r.confidence >= $min_confidence
        RETURN s.name AS subject, s.type AS subject_type,
               r.relation AS relation, r.source_span AS source_span,
               r.source_doc AS source_doc, r.confidence AS confidence,
               o.name AS object, o.type AS object_type
        ORDER BY r.confidence DESC
        LIMIT $limit
        """
        return self._run_query(query, relation=relation, min_confidence=min_confidence, limit=limit)

    def get_triples_by_entity_type(
        self,
        subject_type: str | None = None,
        object_type: str | None = None,
        min_confidence: float = 0.75,
        limit: int = 100,
    ) -> list[dict]:
        """按实体类型过滤查询三元组。"""
        conditions = ["r.confidence >= $min_confidence"]
        if subject_type:
            conditions.append("s.type = $subject_type")
        if object_type:
            conditions.append("o.type = $object_type")
        where_clause = " AND ".join(conditions)

        query = f"""
        MATCH (s:Entity)-[r:RELATED_TO]->(o:Entity)
        WHERE {where_clause}
        RETURN s.name AS subject, s.type AS subject_type,
               r.relation AS relation, r.source_span AS source_span,
               r.source_doc AS source_doc, r.confidence AS confidence,
               o.name AS object, o.type AS object_type
        ORDER BY r.confidence DESC
        LIMIT $limit
        """
        params: dict[str, Any] = {"min_confidence": min_confidence, "limit": limit}
        if subject_type:
            params["subject_type"] = subject_type
        if object_type:
            params["object_type"] = object_type
        return self._run_query(query, **params)

    # ── 概览查询 ──────────────────────────────────────────────────────────────

    def get_entity_type_distribution(self) -> dict[str, int]:
        """实体类型分布统计。"""
        query = "MATCH (e:Entity) RETURN e.type AS type, count(*) AS count ORDER BY count DESC"
        rows = self._run_query(query)
        return {r["type"]: r["count"] for r in rows}

    def get_relation_type_distribution(self) -> dict[str, int]:
        """关系类型分布统计。"""
        query = "MATCH ()-[r:RELATED_TO]->() RETURN r.relation AS relation, count(*) AS count ORDER BY count DESC"
        rows = self._run_query(query)
        return {r["relation"]: r["count"] for r in rows}

    def get_documents_for_entity(self, entity_name: str) -> list[str]:
        """获取实体关联的源文档。"""
        query = """
        MATCH (e:Entity)-[:MENTIONED_IN]->(d:Document)
        WHERE e.name CONTAINS $entity_name
        RETURN d.name AS doc_name
        """
        rows = self._run_query(query, entity_name=entity_name)
        return [r["doc_name"] for r in rows]

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _run_query(self, query: str, **params) -> list[dict]:
        with self._driver.session(database=self._database) as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]
