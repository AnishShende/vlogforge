import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    gemini_api_key: str = ""
    typesafe_api_key: str = ""
    claude_api_key: str = ""
    port: int = 8000
    host: str = "127.0.0.1"
    upload_dir: str = "d:/VlogForge/uploads"
    output_dir: str = "d:/VlogForge/outputs"
    artifact_dir: str = "d:/VlogForge/artifacts"   # Phase 5 stage results (transcripts, JEV scores, job edits)
    enable_artifact_cache: bool = True             # reuse transcription / JEV results by content (non-mock runs)
    log_dir: str = "d:/VlogForge/logs"

    # M0 Auth settings
    jwt_secret_key: str = "your-super-secret-jwt-key-replace-in-prod"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7  # 7 days

    # Phase 0: Quality scoring & Retake detection
    quality_threshold: float = 0.35     # Absolute bad-take threshold (0–1). Conservative default.
    retake_candidate_window_sec: float = 15.0
    retake_opening_window_sec: float = 2.0
    retake_opening_similarity_threshold: float = 0.85
    retake_whole_utterance_similarity_threshold: float = 0.85
    max_intra_segment_repetition_ratio: float = 0.40
    retake_recency_bias_weight: float = 0.1
    retake_self_correction_phrases: list[str] = [
        "let me start over",
        "take two",
        "sorry, again",
        "let's try that again",
        "one more time"
    ]

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
    perception_model: str = "claude-haiku-4-5-20251001"   # Cheap model for Pass 1 classification
    reasoning_model: str = "claude-sonnet-5-5"         # Frontier model for Pass 2 reasoning (Phase 1+)
    gemini_rpm: int = 14                          # Free tier RPM limit

    # M4: Long-Footage Scaling — controls for batch/chunk processing
    # Classification (Workstream 1)
    classification_batch_size: int = 20   # EGT segments per batched Gemini classification call
    # Visual analysis (Workstream 2)
    visual_batch_size: int = 20           # Keyframes per batched Gemini visual analysis call
    visual_analysis_workers: int = 4      # Concurrent ThreadPoolExecutor threads for keyframe description
    dense_sampling_floor_sec: float = 30.0  # Segments shorter than this skip dense multi-keyframe sampling
    # EDL Map-Reduce reasoning (Workstream 3)
    edl_chunk_size: int = 35             # Max EGT segments per Map-phase chunk
    edl_chunk_threshold: int = 50        # Activate Map-Reduce when total segments exceed this

    # Dev/Test Mocks
    enable_mock_llm: bool = False
    enable_mock_whisper: bool = False
    enable_mock_jev: bool = False        # Cache/replay JEV clean-take scores
    mock_llm_dir: str = "d:/VlogForge/mocks"

    # Phase 1+2 Feature Flags
    enable_word_timeline_redundancy: bool = False
    enable_forced_alignment: bool = True   # refine Whisper word times via WhisperX
    enable_word_grid: bool = False         # Archdoc Phase 1: build the word grid; transcripts keep
                                           # conf, interpolated + recovered words, temperature 0 (whisper cache v3)
                                           # (wav2vec2); no-op if whisperx missing
    enable_word_grid_compiler: bool = False  # Archdoc Phase 4: the edit = word-grid speech cleanup -> compiler
    enable_edit_passes: bool = False         # Phase 6.5: LLM edit passes replace the JEV cleanup (needs the compiler flag)
    enable_moments: bool = False             # Phase 8: build the moment table (story beats) after the edit is saved
                                             # -> single-pass render, replacing EDL + assembly. Requires
                                             # enable_word_grid. Speech only: B-roll and target duration are
                                             # ignored (job warnings say so).

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
                self.mock_llm_dir = os.path.join(workspace_root, "mocks")
                self.artifact_dir = os.path.join(workspace_root, "artifacts")

        if self.typesafe_api_key:
            os.environ["TYPESAFE_API_KEY"] = self.typesafe_api_key

settings = Settings()

# Ensure directories exist
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.output_dir, exist_ok=True)
os.makedirs(settings.log_dir, exist_ok=True)
