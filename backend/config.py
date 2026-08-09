import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    GROQ_API_KEY: str = ""

    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "enterprise_knowledge"

    DEFAULT_EMBEDDING_MODEL: str = "BAAI/bge-m3"
    FAST_LLM_MODEL: str = "llama-3.1-8b-instant"
    FLAGSHIP_LLM_MODEL: str = "llama-3.3-70b-versatile"

    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    UPLOAD_DIR: str = "./data/uploads"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)