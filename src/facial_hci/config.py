"""Central config, loaded from env."""
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    llm_provider: str = Field(default="groq", alias="LLM_PROVIDER")
    llm_model: str = Field(default="llama-3.3-70b-versatile", alias="LLM_MODEL")
    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")
    enable_llm_reasoning: bool = Field(default=True, alias="ENABLE_LLM_REASONING")
    llm_cooldown_seconds: float = Field(default=2.5, alias="LLM_COOLDOWN_SECONDS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    project_root: Path = Path(__file__).resolve().parents[2]
    model_dir: Path = Path(__file__).resolve().parents[2] / "models"
    data_dir: Path = Path(__file__).resolve().parents[2] / "training_data"
    profiles_dir: Path = Path(__file__).resolve().parents[2] / "user_profiles"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


settings = Settings()
settings.model_dir.mkdir(exist_ok=True)
settings.data_dir.mkdir(exist_ok=True)
settings.profiles_dir.mkdir(exist_ok=True)
