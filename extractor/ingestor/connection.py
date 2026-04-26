"""Connection utilities for LLM and Neo4j in the ingestor pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from neo4j import GraphDatabase


_DEFAULT_ENV = {
    "LLM_API_KEY": "",
    "LLM_BASE_URL": "https://api.openai.com/v1",
    "LLM_MODEL": "gpt-4o-mini",
    "LLM_TEMPERATURE": "0",
    "NEO4J_URI": "neo4j://localhost:7687",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "12345678",
    "NEO4J_DATABASE": "neo4j",
}


@dataclass(frozen=True)
class IngestorConnectionConfig:
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    llm_temperature: float
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> "IngestorConnectionConfig":
        load_dotenv(dotenv_path=Path(env_file), override=False)
        return cls(
            llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
            llm_base_url=os.getenv(
                "LLM_BASE_URL", _DEFAULT_ENV["LLM_BASE_URL"]
            ).strip(),
            llm_model=os.getenv("LLM_MODEL", _DEFAULT_ENV["LLM_MODEL"]).strip(),
            llm_temperature=float(
                os.getenv("LLM_TEMPERATURE", _DEFAULT_ENV["LLM_TEMPERATURE"]).strip()
            ),
            neo4j_uri=os.getenv("NEO4J_URI", _DEFAULT_ENV["NEO4J_URI"]).strip(),
            neo4j_username=os.getenv(
                "NEO4J_USERNAME", _DEFAULT_ENV["NEO4J_USERNAME"]
            ).strip(),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "").strip(),
            neo4j_database=os.getenv(
                "NEO4J_DATABASE", _DEFAULT_ENV["NEO4J_DATABASE"]
            ).strip(),
        )


def ensure_env_keys(env_file: str | Path = ".env") -> Path:
    """Ensure required connection keys exist in .env without overwriting existing values."""
    env_path = Path(env_file)
    existing: dict[str, str] = {}

    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            existing[key.strip()] = value

    lines_to_append: list[str] = []
    for key, default in _DEFAULT_ENV.items():
        if key not in existing:
            lines_to_append.append(f"{key}={default}")

    if lines_to_append:
        prefix = (
            "\n" if env_path.exists() and env_path.read_text(encoding="utf-8") else ""
        )
        with env_path.open("a", encoding="utf-8") as f:
            f.write(prefix + "\n".join(lines_to_append) + "\n")

    return env_path


def build_llm(config: IngestorConnectionConfig) -> ChatOpenAI:
    if not config.llm_api_key:
        raise ValueError("LLM_API_KEY is required in .env")

    return ChatOpenAI(
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        model=config.llm_model,
        temperature=config.llm_temperature,
    )


def test_llm_connection(config: IngestorConnectionConfig) -> dict[str, str]:
    llm = build_llm(config)
    result = llm.invoke("Reply exactly with: OK")
    content = getattr(result, "content", str(result)).strip()
    if "OK" not in content:
        raise RuntimeError(f"LLM connectivity check failed: {content}")
    return {"status": "ok", "llm": config.llm_model}


def test_neo4j_connection(config: IngestorConnectionConfig) -> dict[str, str]:
    if not config.neo4j_password:
        raise ValueError("NEO4J_PASSWORD is required in .env")

    driver = GraphDatabase.driver(
        config.neo4j_uri,
        auth=(config.neo4j_username, config.neo4j_password),
    )
    try:
        driver.verify_connectivity()
        with driver.session(database=config.neo4j_database) as session:
            value = session.run("RETURN 'OK' AS status").single()["status"]
            if value != "OK":
                raise RuntimeError("Neo4j query check failed")
    finally:
        driver.close()

    return {"status": "ok", "neo4j": config.neo4j_uri}


def run_connection_checks(env_file: str | Path = ".env") -> dict[str, dict[str, str]]:
    ensure_env_keys(env_file)
    config = IngestorConnectionConfig.from_env(env_file)
    llm_status = test_llm_connection(config)
    neo4j_status = test_neo4j_connection(config)
    return {"llm": llm_status, "neo4j": neo4j_status}


if __name__ == "__main__":
    result = run_connection_checks()
    print(result)
