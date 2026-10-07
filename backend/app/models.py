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
    word_timings: List[Dict] = Field(default_factory=list)  # [{word, start, end}, ...] from Whisper
    language_id: str = "en"                             # ISO 639-1 (future: "hi", "hi-en")
    has_speech: bool = False                            # Indicates >= 3 words in transcript

    # === Visual ===
    visual_description: str = ""                        # Short text from vision model
    keyframe_path: Optional[str] = None                 # Path to primary extracted keyframe JPEG
    keyframe_paths: List[str] = Field(default_factory=list) # Paths to denser keyframes for long segments

    # === Classification (Perception layer — cheap model / rule-based) ===
    segment_type: str = "SPEECH"                        # INTRO | OUTRO | SPEECH | B_ROLL | SILENCE
    quality_score: float = 1.0                          # 0.0–1.0, calibrated absolute
    quality_flags: List[str] = Field(default_factory=list)  # ["low_audio", "shaky", "overexposed", "bad_take"]
    is_bad_take: bool = False                           # Derived: quality_score < threshold
    is_superseded_take: bool = False                    # Derived: retake detection (length-agnostic clustering)
    is_stutter_repeat: bool = False                     # Derived: intra-segment repetition
    repetition_ratio: float = 0.0                       # Computed ratio of repeated n-grams

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
# Word Grid — the time reference for speech edits (Archdoc Stage 4)
# ---------------------------------------------------------------------------

# Word flags. Set by the grid builder / speech-activity checks, never by the ASR.
WORD_FLAG_INTERPOLATED = "interpolated"   # aligner gave no time; interpolated between neighbours
WORD_FLAG_ZERO_LENGTH = "zero_length"     # start == end as delivered by the ASR
WORD_FLAG_ON_SILENCE = "on_silence"       # word sits on audio with no detected speech
WORD_FLAG_LONG_SPAN = "long_span"         # aligned span implausibly long for the word
WORD_FLAG_LOW_CONF = "low_conf"           # ASR/aligner confidence below threshold
WORD_FLAG_RECOVERED = "recovered"         # added by re-transcribing speech the first pass missed


def generate_word_id(source_file: str, index: int) -> str:
    """Deterministic word ID: w_<first 8 hex of SHA-256(source_file)>_<index in file>.

    Stable for a given transcript of a given file. A different transcript (new ASR)
    yields a new grid; edit plans must record which grid they reference.
    """
    file_hash = hashlib.sha256(source_file.encode()).hexdigest()[:8]
    return f"w_{file_hash}_{index:05d}"


class Word(BaseModel):
    """One spoken word on the source timeline."""

    id: str
    text: str
    source_file: str
    start: float
    end: float
    conf: Optional[float] = None                        # None when the ASR output had no confidence
    gap_before: float = 0.0                             # silence since the previous word in this file
    flags: List[str] = Field(default_factory=list)


class WordGrid(BaseModel):
    """All words of a job, per file in time order. Edits reference word IDs."""

    words: List[Word] = Field(default_factory=list)
    asr: Dict = Field(default_factory=dict)             # provenance: model, aligner, cache version
    checks: Dict = Field(default_factory=dict)          # speech-activity check results (coverage, flag counts)

    def validate_integrity(self) -> List[str]:
        """Check IDs and timeline invariants. Returns a list of error messages (empty = OK)."""
        errors = []
        seen_ids = set()
        prev_by_file: Dict[str, Word] = {}
        for w in self.words:
            if w.id in seen_ids:
                errors.append(f"Duplicate word id: {w.id}")
            seen_ids.add(w.id)
            if w.end < w.start:
                errors.append(f"Invalid timestamps for {w.id}: start={w.start}, end={w.end}")
            prev = prev_by_file.get(w.source_file)
            if prev is not None:
                if w.start < prev.end - 1e-6:
                    errors.append(f"Overlap: {prev.id} ends {prev.end} after {w.id} starts {w.start}")
                if abs(w.gap_before - (w.start - prev.end)) > 1e-3:
                    errors.append(f"gap_before mismatch for {w.id}: {w.gap_before} vs {w.start - prev.end:.3f}")
            prev_by_file[w.source_file] = w
        return errors

    def by_id(self) -> Dict[str, Word]:
        return {w.id: w for w in self.words}

    def fingerprint(self) -> str:
        """Identity of the word sequence (ids + text, not times: edge refinement keeps a plan valid)."""
        raw = "\n".join(f"{w.id}|{w.text}" for w in self.words)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Edit Plan — planner -> compiler contract (Archdoc Stage 10, roadmap Phase 2)
# ---------------------------------------------------------------------------

class EditSegment(BaseModel):
    """An inclusive run of consecutive grid words from one file."""

    word_start: str
    word_end: str
    reason: str = ""


class EditPlan(BaseModel):
    """Word ranges in output order. The compiler turns these into cuts; no times here."""

    segments: List[EditSegment] = Field(default_factory=list)
    grid_fingerprint: Optional[str] = None              # WordGrid.fingerprint() the plan was made on

    def validate_against(self, grid: "WordGrid") -> List[str]:
        """Check the plan against a grid. Returns a list of error messages (empty = OK)."""
        errors = []
        if not self.segments:
            errors.append("Plan has no segments")
        if self.grid_fingerprint and self.grid_fingerprint != grid.fingerprint():
            errors.append(f"Plan made on grid {self.grid_fingerprint}, not this grid {grid.fingerprint()}")
        index = {w.id: i for i, w in enumerate(grid.words)}
        used: Dict[int, int] = {}
        for n, seg in enumerate(self.segments):
            missing = [wid for wid in (seg.word_start, seg.word_end) if wid not in index]
            if missing:
                errors.append(f"Segment {n}: unknown word id(s) {missing}")
                continue
            a, b = index[seg.word_start], index[seg.word_end]
            if a > b:
                errors.append(f"Segment {n}: word_start {seg.word_start} comes after word_end {seg.word_end}")
                continue
            if grid.words[a].source_file != grid.words[b].source_file:
                errors.append(f"Segment {n}: spans files {grid.words[a].source_file!r} and {grid.words[b].source_file!r}")
                continue
            for i in range(a, b + 1):
                if i in used:
                    errors.append(f"Segment {n}: word {grid.words[i].id} already used by segment {used[i]}")
                    break
                used[i] = n
        return errors


# Cut flags. Set by the compiler; every flagged cut is listed in the compile report.
CUT_FLAG_TIGHT = "tight"            # no pause between the words: cut at the quietest point between their edges
CUT_FLAG_LONG_TAIL = "long_tail"    # activity ran past the word edge longer than the search limit
CUT_FLAG_FRAME_OFF = "frame_off"    # segment length could not be snapped to whole frames inside the window
CUT_FLAG_IN_NOISE = "in_noise"      # dead-air cut placed in background noise (no words in the gap)
CUT_FLAG_VALLEY = "valley"          # activity next to the word was split at a deep dip (untranscribed sound beyond it)


class CutPoint(BaseModel):
    """Where one end of a compiled segment was cut, and why there."""

    time: float                                         # source time of the cut
    side: str                                           # "in" (before the first word) | "out" (after the last)
    word_edge: float                                    # the kept word's start (in) / end (out)
    activity_edge: float                                # where speech/loudness past the word edge stops
    window: List[float]                                 # [lo, hi] the cut was chosen from
    level_db: float                                     # level at the cut
    flags: List[str] = Field(default_factory=list)


class CompiledSegment(BaseModel):
    """One source range of the output, with the plan words it carries."""

    source_file: str
    src_in: float
    src_out: float
    word_ids: List[str]
    reasons: List[str] = Field(default_factory=list)    # reasons of the plan segments merged into this one
    cut_in: CutPoint
    cut_out: CutPoint
    pause_shortened_before_sec: Optional[float] = None  # S3: original silence before this segment, if shortened


class CompiledTimeline(BaseModel):
    """The compiler's output: what the render executes. Times are source times."""

    segments: List[CompiledSegment]
    fps: int
    join_fade_sec: float                                # audio fade in/out at every join (no overlap)
    head_fade_sec: float                                # global audio fade-in, never past the first word's activity
    tail_fade_sec: float                                # global audio fade-out, never before the last word's activity
    duration_sec: float                                 # sum of whole-frame segment lengths
    grid_fingerprint: str = ""
    pause_targets: Optional[Dict[str, float]] = None    # S3 targets used (None = pause shortening off)


# ---------------------------------------------------------------------------
# Validation Report — compiled output checked against its plan (roadmap Phase 3)
# ---------------------------------------------------------------------------

CHECK_PASS, CHECK_WARN, CHECK_FAIL = "pass", "warn", "fail"


class ValidationCheck(BaseModel):
    """One finding. `segment` is the compiled segment index (output order), None = whole output."""

    check: str                                          # e.g. "cut_inside_word", "render_fidelity"
    status: str                                         # pass | warn | fail
    segment: Optional[int] = None
    detail: str = ""
    evidence: Dict = Field(default_factory=dict)


class ValidationReport(BaseModel):
    """Post-condition checks on a compiled job; status = fail if any check failed,
    else warn if any warned, else pass."""

    status: str = CHECK_PASS
    checks: List[ValidationCheck] = Field(default_factory=list)

    def summarize(self) -> Dict:
        bad = [c for c in self.checks if c.status != CHECK_PASS]
        counts: Dict[str, int] = {}
        for c in bad:
            counts[f"{c.status}:{c.check}"] = counts.get(f"{c.status}:{c.check}", 0) + 1
        return {"status": self.status, "checks": len(self.checks), "not_pass": counts}


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
# Video Metadata — M5 YouTube Optimization Output
# ---------------------------------------------------------------------------

class VideoMetadata(BaseModel):
    """AI-generated YouTube-ready metadata for the final vlog."""

    title: str = ""
    description: str = ""
    tags: List[str] = Field(default_factory=list)
    chapters: List[Dict] = Field(default_factory=list)  # [{"time": "0:00", "label": "Intro"}, ...]


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
    warnings: List[str] = Field(default_factory=list)
    # M4: Pipeline metrics for diagnostics — populated at pipeline completion
    pipeline_metrics: Optional[Dict] = None  # keys: total_segments, chunks_used, raw_footage_sec, edl_path
    # M5: AI-generated YouTube metadata — populated after assembly
    metadata: Optional[VideoMetadata] = None
    # Dev: Indicates if the LLM calls were mocked
    llm_mode: str = "real"  # "real" | "mocked"


class WSProgressEvent(BaseModel):
    stage: str
    progress: int
    message: str
    download_url: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
