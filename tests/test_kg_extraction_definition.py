from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from extractor.ingestor import DEFAULT_KG_EXTRACTION_DEFINITION


class TestKGExtractionDefinition(unittest.TestCase):
    def test_contains_expected_entity_types(self) -> None:
        names = {item.name for item in DEFAULT_KG_EXTRACTION_DEFINITION.entity_types}
        self.assertSetEqual(
            names,
            {"技术", "场景", "风险", "治理措施", "主体", "政策文件", "目标"},
        )

    def test_contains_expected_core_relations(self) -> None:
        names = {item.name for item in DEFAULT_KG_EXTRACTION_DEFINITION.relation_types}
        for relation in ("应用于", "引发", "聚焦于", "包括", "提出", "影响", "助力"):
            self.assertIn(relation, names)

    def test_contains_seven_core_templates(self) -> None:
        templates = DEFAULT_KG_EXTRACTION_DEFINITION.triple_templates
        self.assertEqual(len(templates), 7)

    def test_default_confidence_threshold(self) -> None:
        self.assertEqual(DEFAULT_KG_EXTRACTION_DEFINITION.confidence_threshold, 0.75)


if __name__ == "__main__":
    unittest.main(verbosity=2)
