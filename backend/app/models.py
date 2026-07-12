import hashlib
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict
from datetime import datetime


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def generate_clip_id(source_file: str, start_sec: float, end_sec: float) -> str:
    """Deterministic clip ID: first 12 hex chars of SHA-256(source_file|start|end).

    Rounding to 6 decimal places guarantees that the same segment always
    produces the same ID, even if floating point serialization introduces
    minor rounding noise.
    """
    raw = f"{source_file}|{start_sec:.6f}|{end_sec:.6f}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Source File Metadata
# ---------------------------------------------------------------------------

class VideoFileInfo(BaseModel):
    filename: str
    original_path: str
    duration: float = 0.0
    size_bytes: int = 0
    audio_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Editorial Ground Truth (EGT) — Pass 1 Perception Output
# ---------------------------------------------------------------------------

class EGTSegment(BaseModel):
    """One row per detected shot/take in the perception pass."""

    # === Identity & Source ===
    clip_id: str                                        # Deterministic: generate_clip_id()
    source_file: str                                    # Original filename
    source_file_hash: str = ""                          # SHA-256 of the source video file

    # === Temporal ===
    start_sec: float
    end_sec: float
    duration_sec: float = 0.0                           # Computed: end_sec - start_sec

    # === Transcript ===
    transcript: str = ""                                # Aligned speech content
    language_id: str = "en"                             # ISO 639-1 (future: "hi", "hi-en")

    # === Visual ===
    visual_description: str = ""                        # Short text from vision model
    keyframe_path: Optional[str] = None                 # Path to primary extracted keyframe JPEG
    keyframe_paths: List[str] = Field(default_factory=list) # Paths to denser keyframes for long segments

    # === Classification (Perception layer — cheap model / rule-based) ===
    segment_type: str = "SPEECH"                        # INTRO | OUTRO | SPEECH | B_ROLL | SILENCE
    quality_score: float = 1.0                          # 0.0–1.0, calibrated absolute
    quality_flags: List[str] = Field(default_factory=list)  # ["low_audio", "shaky", "overexposed", "bad_take"]
    is_bad_take: bool = False                           # Derived: quality_score < threshold

    # === Structural (populated in P0 schema, consumed in Phase 1+) ===
    journey_collection: Optional[str] = None            # "journey" | "collection" | None
    structural_cue: Optional[str] = None                # Detected cue type (Phase 1)
    structural_cue_target: Optional[str] = None         # clip_id the cue references (Phase 1)

    # === Provenance ===
    perception_model: str = ""                          # e.g. "gemini-2.0-flash-lite"
    generated: bool = False                             # Always False for perception output

    # === Subject/Action Tags ===
    tags: List[str] = Field(default_factory=list)       # e.g. ["person_speaking", "outdoor"]

    @model_validator(mode="after")
    def compute_duration(self):
        self.duration_sec = round(self.end_sec - self.start_sec, 6)
        return self

    @field_validator("segment_type")
    @classmethod
    def validate_segment_type(cls, v: str) -> str:
        allowed = {"INTRO", "OUTRO", "SPEECH", "B_ROLL", "SILENCE"}
        if v not in allowed:
            raise ValueError(f"segment_type must be one of {allowed}, got '{v}'")
        return v


class EGTDocument(BaseModel):
    """Wrapper holding the full perception output for a job."""

    segments: List[EGTSegment] = Field(default_factory=list)
    total_duration_sec: float = 0.0
    source_file_count: int = 0
    context_summary: str = ""                           # Synthesised context document
    perception_model_version: str = ""

    def validate_integrity(self) -> List[str]:
        """Check referential integrity. Returns a list of error messages (empty = OK)."""
        errors = []
        seen_ids = set()
        for seg in self.segments:
            if seg.clip_id in seen_ids:
                errors.append(f"Duplicate clip_id: {seg.clip_id}")
            seen_ids.add(seg.clip_id)
            if seg.end_sec <= seg.start_sec:
                errors.append(
                    f"Invalid timestamps for clip_id {seg.clip_id}: "
                    f"start={seg.start_sec}, end={seg.end_sec}"
                )
        return errors


# ---------------------------------------------------------------------------
# Edit Decision List (EDL) — Pass 2 Reasoning Output
# ---------------------------------------------------------------------------

class EDLEntry(BaseModel):
    """Single entry in the Edit Decision List."""

    clip_id: str                                        # Must resolve to a real EGTSegment.clip_id
    source_file: str                                    # Denormalized for assembly convenience
    start_sec: float
    end_sec: float
    core_start_sec: Optional[float] = None              # Minimum safe bound
    core_end_sec: Optional[float] = None                # Minimum safe bound
    narrative_priority: str = "MEDIUM"                  # LOW | MEDIUM | CRITICAL
    quality_score: float = 0.0                          # Pass 1 quality score for tie-breaking
    editorial_type: str = "KEEP"                        # KEEP | INTRO | OUTRO
    sequence_index: int = 0                             # Position in final timeline

    # === Human Review Metadata ===
    human_modified: bool = False
    modification_type: Optional[str] = None             # "trim" | "reorder" | "added" | "removed"


# ---------------------------------------------------------------------------
# Legacy EDL Item — kept for backward compatibility during migration
# ---------------------------------------------------------------------------

class EDLItem(BaseModel):
    """Legacy EDL item format. Deprecated — use EDLEntry."""
    video_file: str
    start_sec: float
    end_sec: float
    type: str  # INTRO, OUTRO, HIGHLIGHT, B_ROLL


# ---------------------------------------------------------------------------
# Job Status & WebSocket Events
# ---------------------------------------------------------------------------

class JobCreate(BaseModel):
    context_text: Optional[str] = ""

class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, ingesting, transcribing, analyzing, scoring, egt_building, edl_generating, assembling, complete, failed
    progress: int
    message: str
    files: List[VideoFileInfo] = []
    context_text: str = ""
    vlog_genre: str = "default"
    target_duration: Optional[float] = 10.0
    quality_threshold: float = 0.35
    created_at: datetime
    completed_at: Optional[datetime] = None
    output_video_url: Optional[str] = None

class WSProgressEvent(BaseModel):
    stage: str
    progress: int
    message: str
    download_url: Optional[str] = None
