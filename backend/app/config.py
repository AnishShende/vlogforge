import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    gemini_api_key: str = ""
    port: int = 8000
    host: str = "127.0.0.1"
    upload_dir: str = "d:/VlogForge/uploads"
    output_dir: str = "d:/VlogForge/outputs"
    log_dir: str = "d:/VlogForge/logs"

    # Phase 0: Quality scoring
    quality_threshold: float = 0.35     # Absolute bad-take threshold (0–1). Conservative default.

    # Model tiering
    perception_model: str = "gemini-2.0-flash-lite"   # Cheap model for Pass 1 classification
    reasoning_model: str = "gemini-2.5-flash"         # Frontier model for Pass 2 reasoning (Phase 1+)

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Ensure directories exist
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.output_dir, exist_ok=True)
os.makedirs(settings.log_dir, exist_ok=True)
