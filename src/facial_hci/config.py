"""Central config, loaded from env."""
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from pathlib import Path
import logging


class Settings(BaseSettings):
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    llm_provider: str = Field(default="groq", alias="LLM_PROVIDER")
    llm_model: str = Field(default="llama-3.3-70b-versatile", alias="LLM_MODEL")
    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")
    enable_llm_reasoning: bool = Field(default=True, alias="ENABLE_LLM_REASONING")
    llm_cooldown_seconds: float = Field(default=2.5, alias="LLM_COOLDOWN_SECONDS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    enable_redis: bool = Field(default=False, alias="ENABLE_REDIS")
    cdn_url: str = Field(default="", alias="CDN_URL")

    project_root: Path = Path(__file__).resolve().parents[2]
    model_dir: Path = Path(__file__).resolve().parents[2] / "models"
    data_dir: Path = Path(__file__).resolve().parents[2] / "training_data"
    profiles_dir: Path = Path(__file__).resolve().parents[2] / "user_profiles"

    @validator('llm_provider')
    def validate_llm_provider(cls, v):
        valid_providers = ['groq', 'ollama', 'none']
        if v.lower() not in valid_providers:
            raise ValueError(f'llm_provider must be one of {valid_providers}')
        return v.lower()

    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'log_level must be one of {valid_levels}')
        return v.upper()

    @validator('llm_cooldown_seconds')
    def validate_cooldown(cls, v):
        if v < 0:
            raise ValueError('llm_cooldown_seconds must be non-negative')
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


settings = Settings()
settings.model_dir.mkdir(exist_ok=True)
settings.data_dir.mkdir(exist_ok=True)
settings.profiles_dir.mkdir(exist_ok=True)

# Configure logging based on settings
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
