from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    llm_provider: str = 'ollama'
    ollama_model: str = 'llama3.2:3b'
    ollama_base_url: str = 'http://localhost:11434'
    openrouter_model: str = 'meta-llama/llama-3.3-70b-instruct:free'
    openrouter_fallback_model: str = ''
    openrouter_api_key: str = ''
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    chroma_dir: Path = Path('./storage/chroma')
    data_dir: Path = Path('./data')
    retrieval_mode: str = 'hybrid'
    top_k: int = Field(default=4, ge=1, le=20)
    chunk_size: int = Field(default=850, ge=100, le=5000)
    chunk_overlap: int = Field(default=130, ge=0)
    llm_timeout: int = Field(default=45, ge=5)
    max_retries: int = Field(default=2, ge=0, le=5)

    @model_validator(mode='after')
    def validate_values(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError('CHUNK_OVERLAP must be smaller than CHUNK_SIZE')
        if self.llm_provider not in ('ollama', 'openrouter'):
            raise ValueError('LLM_PROVIDER must be ollama or openrouter')
        if self.retrieval_mode not in ('similarity', 'mmr', 'hybrid'):
            raise ValueError('RETRIEVAL_MODE must be similarity, mmr, or hybrid')
        return self

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
