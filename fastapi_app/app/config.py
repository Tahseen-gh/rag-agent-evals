from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # App
    app_name: str = "domain-rag-evals-api"
    debug: bool = False

    # Auth
    api_key: str | None = None  # header: X-API-KEY
    jwt_secret: str | None = None
    jwt_algorithm: str = "HS256"  # used if jwt_secret is set

    # CORS
    cors_origins: list[str] = Field(default_factory=list)

    # Protection
    rate_limit_qps: float = 5.0
    request_body_max_bytes: int = 100_000
    fuzz_mode: bool = False

    # Observability
    metrics_enabled: bool = True
    json_logs: bool = True
    log_level: str = "INFO"
    request_id_header: str = "X-Request-ID"

    # Retrieval
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hybrid_alpha: float = 0.5
    use_dummy_embeddings: bool = True
    rag_top_k: int = 4

    # Generation / agent
    anthropic_api_key: str | None = None
    llm_model: str = "claude-sonnet-5"
    use_dummy_llm: bool = True
    agent_max_turns: int = 6


settings = Settings()
