"""Prompt definitions for strict KG extraction."""

from __future__ import annotations

import json

from langchain_core.prompts import ChatPromptTemplate

from .kg_extraction_definition import KGExtractionDefinition


_SYSTEM_RULES = """
You are an information extraction engine for legal and academic publishing texts.
Output MUST be valid JSON only.
Do not output markdown, explanations, or extra keys.
Only use entity types, relation names, and templates provided by the user.
Reject uncertain relation/type mapping instead of guessing.
""".strip()


_OUTPUT_SCHEMA_HINT = {
    "triples": [
        {
            "subject": "string",
            "subject_type": "string",
            "relation": "string",
            "object": "string",
            "object_type": "string",
            "evidence": {
                "source_doc": "string",
                "source_span": "string",
                "section": "string",
                "confidence": 0.0,
                "time_or_policy_version": "string",
            },
        }
    ]
}


def build_extraction_prompt(definition: KGExtractionDefinition) -> ChatPromptTemplate:
    entity_types = [entity.name for entity in definition.entity_types]
    relation_types = [relation.name for relation in definition.relation_types]
    templates = [
        {
            "name": item.name,
            "subject_type": item.subject_type,
            "relation": item.relation,
            "object_type": item.object_type,
            "description": item.description,
        }
        for item in definition.triple_templates
    ]

    user_template = """
Task: Extract aligned triples from the provided text.

Constraints:
- Allowed entity_types: {entity_types}
- Allowed relation_types: {relation_types}
- Allowed templates: {templates}
- confidence must be between 0 and 1.
- confidence should be >= {confidence_threshold} for accepted triples.
- Keep source_span as a verbatim quote from the input text.

Source metadata:
- source_doc: {source_doc}
- section: {section}

Text:
{text}

Output JSON schema:
{output_schema}
""".strip()

    return ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_RULES),
            ("human", user_template),
        ]
    ).partial(
        entity_types=json.dumps(entity_types, ensure_ascii=False),
        relation_types=json.dumps(relation_types, ensure_ascii=False),
        templates=json.dumps(templates, ensure_ascii=False),
        confidence_threshold=str(definition.confidence_threshold),
        output_schema=json.dumps(_OUTPUT_SCHEMA_HINT, ensure_ascii=False, indent=2),
    )
