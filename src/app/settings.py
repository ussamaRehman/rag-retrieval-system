from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    index_dir: str = "artifacts/indexes/dev"
    api_version: str = "0.1.0"
    max_top_k: int = 20
    default_mode: str = "bm25"
    snippet_chars: int = 220

    model_config = SettingsConfigDict(env_prefix="RAG_")
