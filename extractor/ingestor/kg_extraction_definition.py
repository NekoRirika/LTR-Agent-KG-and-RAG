"""AI+法律/学术出版场景的知识图谱抽取定义。

该模块提供统一的实体类型、关系类型、模板定义与证据字段配置，
用于后续规则抽取、模型抽取或混合抽取流程复用。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class EntityType:
    name: str
    aliases: tuple[str, ...]
    examples: tuple[str, ...]


@dataclass(frozen=True)
class RelationType:
    name: str
    aliases: tuple[str, ...]
    subject_types: tuple[str, ...]
    object_types: tuple[str, ...]
    trigger_words: tuple[str, ...]


@dataclass(frozen=True)
class TripleTemplate:
    name: str
    subject_type: str
    relation: str
    object_type: str
    description: str


@dataclass(frozen=True)
class EvidenceSchema:
    fields: tuple[str, ...]


@dataclass(frozen=True)
class KGExtractionDefinition:
    entity_types: tuple[EntityType, ...]
    relation_types: tuple[RelationType, ...]
    triple_templates: tuple[TripleTemplate, ...]
    evidence_schema: EvidenceSchema
    attach_evidence: bool
    confidence_threshold: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_default_kg_extraction_definition() -> KGExtractionDefinition:
    """构建默认知识图谱抽取定义。"""
    entity_types = (
        EntityType(
            name="技术",
            aliases=("技术/对象",),
            examples=("AIGC", "人工智能", "智能出版"),
        ),
        EntityType(
            name="场景",
            aliases=("场景/领域",),
            examples=("学术出版", "同行评议", "编辑加工", "成果发布与传播"),
        ),
        EntityType(
            name="风险",
            aliases=("伦理风险",),
            examples=("版权归属争议", "虚假信息生成", "学术生态风险"),
        ),
        EntityType(
            name="治理措施",
            aliases=("治理路径",),
            examples=("原则约束", "义务规范", "技术检测", "法律规制", "教育监督"),
        ),
        EntityType(
            name="主体",
            aliases=("研究主体",),
            examples=("学者", "出版机构", "国家"),
        ),
        EntityType(
            name="政策文件",
            aliases=("文件",),
            examples=("《中共中央关于制定国民经济和社会发展第十五个五年规划的建议》",),
        ),
        EntityType(
            name="目标",
            aliases=("目标/价值",),
            examples=("营造良好文化生态", "提升文化原创能力", "推动高质量发展"),
        ),
    )

    relation_types = (
        RelationType(
            name="应用于",
            aliases=("用于",),
            subject_types=("技术",),
            object_types=("场景",),
            trigger_words=("应用于", "用于"),
        ),
        RelationType(
            name="引发",
            aliases=("带来", "导致"),
            subject_types=("技术", "场景"),
            object_types=("风险",),
            trigger_words=("引发", "带来", "导致"),
        ),
        RelationType(
            name="聚焦于",
            aliases=("关注",),
            subject_types=("主体",),
            object_types=("场景", "风险", "治理措施", "目标"),
            trigger_words=("聚焦于", "关注"),
        ),
        RelationType(
            name="属于",
            aliases=("归属",),
            subject_types=("风险", "治理措施", "场景", "目标"),
            object_types=("场景", "风险", "治理措施", "目标"),
            trigger_words=("属于", "归属"),
        ),
        RelationType(
            name="包括",
            aliases=("包含",),
            subject_types=("治理措施", "场景", "目标"),
            object_types=("治理措施", "场景", "目标", "风险"),
            trigger_words=("包括", "包含"),
        ),
        RelationType(
            name="提出",
            aliases=("强调",),
            subject_types=("政策文件",),
            object_types=("目标", "治理措施", "场景"),
            trigger_words=("提出", "强调"),
        ),
        RelationType(
            name="影响",
            aliases=("作用于",),
            subject_types=("技术", "场景", "风险"),
            object_types=("主体", "场景", "目标"),
            trigger_words=("影响", "作用于"),
        ),
        RelationType(
            name="需要",
            aliases=("依赖",),
            subject_types=("目标", "场景"),
            object_types=("治理措施",),
            trigger_words=("需要", "依赖"),
        ),
        RelationType(
            name="助力",
            aliases=("促进", "推动"),
            subject_types=("治理措施",),
            object_types=("目标",),
            trigger_words=("助力", "促进", "推动"),
        ),
    )

    triple_templates = (
        TripleTemplate(
            name="技术应用类",
            subject_type="技术",
            relation="应用于",
            object_type="场景",
            description="(技术, 应用于, 场景)",
        ),
        TripleTemplate(
            name="风险触发类",
            subject_type="技术",
            relation="引发",
            object_type="风险",
            description="(技术/实践, 引发, 风险)",
        ),
        TripleTemplate(
            name="研究关注类",
            subject_type="主体",
            relation="聚焦于",
            object_type="场景",
            description="(研究主体, 聚焦于, 议题)",
        ),
        TripleTemplate(
            name="治理构成类",
            subject_type="治理措施",
            relation="包括",
            object_type="治理措施",
            description="(风险治理, 包括, 治理措施)",
        ),
        TripleTemplate(
            name="政策要求类",
            subject_type="政策文件",
            relation="提出",
            object_type="目标",
            description="(政策文件, 提出, 要求/目标)",
        ),
        TripleTemplate(
            name="影响路径类",
            subject_type="技术",
            relation="影响",
            object_type="场景",
            description="(AIGC应用, 影响, 学术出版主体/内容/生态)",
        ),
        TripleTemplate(
            name="目标支撑类",
            subject_type="治理措施",
            relation="助力",
            object_type="目标",
            description="(伦理治理路径, 助力, 学术出版高质量发展)",
        ),
    )

    evidence_schema = EvidenceSchema(
        fields=(
            "source_doc",
            "source_span",
            "section",
            "confidence",
            "time_or_policy_version",
        )
    )

    return KGExtractionDefinition(
        entity_types=entity_types,
        relation_types=relation_types,
        triple_templates=triple_templates,
        evidence_schema=evidence_schema,
        attach_evidence=True,
        confidence_threshold=0.75,
    )


DEFAULT_KG_EXTRACTION_DEFINITION = build_default_kg_extraction_definition()
