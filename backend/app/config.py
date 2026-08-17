import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    gemini_api_key: str = ""
    port: int = 8000
    host: str = "127.0.0.1"
    upload_dir: str = "d:/VlogForge/uploads"
    output_dir: str = "d:/VlogForge/outputs"
    log_dir: str = "d:/VlogForge/logs"

    # M0 Auth settings
    jwt_secret_key: str = "your-super-secret-jwt-key-replace-in-prod"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7  # 7 days

    # Phase 0: Quality scoring
    quality_threshold: float = 0.35     # Absolute bad-take threshold (0–1). Conservative default.

    # Scene Detection (two-pass cascade)
    content_detector_threshold: float = 27.0    # ContentDetector HSV delta threshold
    adaptive_detector_threshold: float = 3.0    # AdaptiveDetector rolling average threshold
    long_scene_threshold_sec: float = 15.0      # Scenes longer than this trigger adaptive sub-detection
    min_scene_duration_sec: float = 1.0         # Scenes shorter than this are merged into neighbors

    # Speech-gap subdivision (duration-relative)
    long_scene_ratio: float = 0.10              # Scenes > ratio × target_duration trigger speech-gap splitting
    long_scene_floor_sec: float = 5.0           # Absolute floor: never consider scenes < this as "long"
    speech_gap_ratio: float = 0.03              # Gaps > ratio × target_duration are split candidates
    speech_gap_floor_sec: float = 1.5           # Absolute floor: never split on gaps shorter than this

    # Model tiering
    perception_model: str = "gemini-2.0-flash-lite"   # Cheap model for Pass 1 classification
    reasoning_model: str = "gemini-2.5-flash"         # Frontier model for Pass 2 reasoning (Phase 1+)

    # M4: Long-Footage Scaling — controls for batch/chunk processing
    # Classification (Workstream 1)
    classification_batch_size: int = 30   # EGT segments per batched Gemini classification call
    # Visual analysis (Workstream 2)
    visual_analysis_workers: int = 4      # Concurrent ThreadPoolExecutor threads for keyframe description
    dense_sampling_floor_sec: float = 30.0  # Segments shorter than this skip dense multi-keyframe sampling
    # EDL Map-Reduce reasoning (Workstream 3)
    edl_chunk_size: int = 35             # Max EGT segments per Map-phase chunk
    edl_chunk_threshold: int = 50        # Activate Map-Reduce when total segments exceed this

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def __init__(self, **values):
        super().__init__(**values)
        import platform
        # On non-Windows platforms, translate Windows D: drive defaults to local workspace paths
        if self.upload_dir.lower().startswith("d:"):
            if platform.system() != "Windows" or not (os.path.exists("d:\\") or os.path.exists("D:\\")):
                workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                self.upload_dir = os.path.join(workspace_root, "uploads")
                self.output_dir = os.path.join(workspace_root, "outputs")
                self.log_dir = os.path.join(workspace_root, "logs")

settings = Settings()

# Ensure directories exist
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.output_dir, exist_ok=True)
os.makedirs(settings.log_dir, exist_ok=True)
