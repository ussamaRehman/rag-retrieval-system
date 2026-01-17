from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    index_dir: str = "artifacts/indexes/dev"
    api_version: str = "0.1.0"
    max_top_k: int = 20
    default_mode: str = "bm25"
    snippet_chars: int = 220
    rate_limit_rps: float = 5.0
    rate_limit_burst: int = 10
    timeout_seconds: float = 5.0
    max_request_bytes: int = 1_000_000
    max_query_chars: int = 2000
    max_batch: int = 25

    model_config = SettingsConfigDict(env_prefix="RAG_")
