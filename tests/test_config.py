"""Tests for configuration management."""
import pytest
from pathlib import Path
from src.facial_hci.config import Settings


def test_settings_default_values():
    """Test that Settings has correct default values."""
    settings = Settings()
    
    assert settings.llm_provider == "groq"
    assert settings.llm_model == "llama-3.3-70b-versatile"
    assert settings.enable_llm_reasoning == True
    assert settings.llm_cooldown_seconds == 2.5
    assert settings.log_level == "INFO"
    assert settings.enable_redis == False
    assert settings.cdn_url == ""


def test_settings_paths():
    """Test that Settings has correct path configuration."""
    settings = Settings()
    
    assert settings.project_root.exists()
    assert settings.model_dir == settings.project_root / "models"
    assert settings.data_dir == settings.project_root / "training_data"
    assert settings.profiles_dir == settings.project_root / "user_profiles"


def test_settings_llm_provider_validation():
    """Test LLM provider validation."""
    from pydantic import ValidationError
    
    # Valid providers
    Settings(llm_provider="groq")
    Settings(llm_provider="ollama")
    Settings(llm_provider="none")
    
    # Invalid provider should raise error
    with pytest.raises(ValidationError):
        Settings(llm_provider="invalid_provider")


def test_settings_log_level_validation():
    """Test log level validation."""
    from pydantic import ValidationError
    
    # Valid log levels
    Settings(log_level="DEBUG")
    Settings(log_level="INFO")
    Settings(log_level="WARNING")
    Settings(log_level="ERROR")
    
    # Invalid log level should raise error
    with pytest.raises(ValidationError):
        Settings(log_level="INVALID")
