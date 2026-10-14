import os
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root: uploads/, outputs/, artifacts/, logs/ and mocks/ live next to backend/ by default.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Settings(BaseSettings):
    claude_api_key: str = ""
    port: int = 8000
    host: str = "127.0.0.1"
    upload_dir: str = os.path.join(REPO_ROOT, "uploads")
    output_dir: str = os.path.join(REPO_ROOT, "outputs")
    artifact_dir: str = os.path.join(REPO_ROOT, "artifacts")  # stage results (transcripts, job edits, moments)
    enable_artifact_cache: bool = True             # reuse transcription results by audio content (non-mock runs)
    log_dir: str = os.path.join(REPO_ROOT, "logs")

    # Auth
    jwt_secret_key: str = "your-super-secret-jwt-key-replace-in-prod"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7  # 7 days

    claude_timeout_sec: float = 120.0             # per request: a dropped connection fails fast instead of the SDK's 600 s

    # Dev/Test Mocks
    enable_mock_whisper: bool = False      # replay recorded transcriptions from mock_llm_dir (eval runs)
    mock_llm_dir: str = os.path.join(REPO_ROOT, "mocks")

    # Pipeline: word grid -> edit passes -> story plan -> compiler -> render
    enable_forced_alignment: bool = True   # refine Whisper word times via WhisperX (wav2vec2); no-op if whisperx missing
    enable_story_plan: bool = True         # moments -> story plan (order + target duration) before the render
    enable_moments: bool = True            # story plan off: still build the moment table after the edit is saved
    enable_broll: bool = False             # B-roll over voice-over (footage categories, story-aware shots); not in the MVP
    enable_background_post_copy: bool = True # write the export panel's post copy after each render
    post_copy_delay_sec: float = 45.0        # ... this long after the LAST render, so a burst of re-compiles costs one call

    model_config = SettingsConfigDict(
        env_file=os.path.join(REPO_ROOT, ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Ensure directories exist
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.output_dir, exist_ok=True)
os.makedirs(settings.log_dir, exist_ok=True)
