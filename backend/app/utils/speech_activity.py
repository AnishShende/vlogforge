"""Speech activity on the source audio: what is speech, where it starts and ends.

Shared by the pipeline (word grid checks) and the eval tools (gold snapping).
Timings come from the audio alone, never from ASR output.

Speech detection (method):
  vad     (default) Silero voice-activity detection (bundled with faster-whisper).
          Distinguishes speech from loud non-speech (cooking, traffic, music), but
          misses elongated / sung speech (chai clip 'Byeee': p <= 0.43).
  energy  per-file loudness threshold. Only reliable in quiet rooms: on a noisy
          kitchen clip it agreed with VAD on 69% of frames (96.6% on IMG_1614).
`Envelope.loud` (the energy mask) is always computed, so callers can flag speech
VAD may have missed (loud but not VAD) without trusting loudness alone.
"""

import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

SR = 16000
HOP_SEC = 0.010            # energy frame hop
SEARCH_SEC = 0.5           # how far an edge inside speech may move (edges on silence
                           # shrink inward without limit, bounded by the span itself)
MIN_SILENCE_SEC = 0.08     # silence shorter than this is inside a word (stop consonants)
MIN_UNACCOUNTED_SEC = 0.3  # ignore shorter uncovered speech blips
FLOOR_DB = -80.0           # clamp for digital silence
SPEECH_REL = 0.3           # energy method: threshold = floor + SPEECH_REL * (peak - floor), dB
VAD_MIN_SILENCE_MS = 100   # vad method: shorter pauses stay inside one speech region


@dataclass
class Envelope:
    speech: np.ndarray     # bool per frame: speech per the chosen method
    threshold_db: float
    floor_db: float
    peak_db: float
    method: str = "energy"
    loud: Optional[np.ndarray] = None   # bool per frame: above the per-file loudness threshold

    def t2i(self, t: float) -> int:
        return int(np.clip(round(t / HOP_SEC), 0, len(self.speech) - 1))

    def i2t(self, i: int) -> float:
        return round(i * HOP_SEC, 3)


def load_audio(path: str) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR),
                          "-f", "s16le", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def envelope(audio: np.ndarray, method: str = "vad") -> Envelope:
    hop = int(HOP_SEC * SR)
    n = len(audio) // hop
    rms = np.sqrt((audio[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    db = np.maximum(20 * np.log10(rms + 1e-12), FLOOR_DB)
    floor, peak = np.percentile(db, 10), np.percentile(db, 95)
    thr = floor + SPEECH_REL * (peak - floor)
    if method == "energy":
        speech = db > thr
    elif method == "vad":
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        speech = np.zeros(n, bool)
        for t in get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=VAD_MIN_SILENCE_MS,
                                                         speech_pad_ms=0)):
            speech[int(t["start"] / hop): int(t["end"] / hop)] = True
    else:
        raise ValueError(f"unknown speech detection method {method!r}")
    return Envelope(speech=speech, threshold_db=float(thr), floor_db=float(floor), peak_db=float(peak),
                    method=method, loud=db > thr)


def _silent_run(env: Envelope, i: int, step: int) -> bool:
    """True if MIN_SILENCE_SEC of silence starts at frame i going in direction step."""
    k = max(1, int(MIN_SILENCE_SEC / HOP_SEC))
    idx = [i + step * j for j in range(k)]
    return all(0 <= x < len(env.speech) and not env.speech[x] for x in idx)


def _onset_after_silence(env: Envelope, j: int) -> int:
    """First speech frame at/after silent frame j."""
    while j < len(env.speech) and not env.speech[j]:
        j += 1
    return j


def _offset_before_silence(env: Envelope, j: int) -> int:
    """Exclusive end of the speech that precedes silent frame j."""
    while j >= 0 and not env.speech[j]:
        j -= 1
    return j + 1


def snap_start(env: Envelope, t: float, span_end: float, outside_labelled: bool = True,
               search_sec: float = SEARCH_SEC) -> Tuple[float, str]:
    """Edge on silence: shrink inward to the span's first speech (never past span_end).
    Edge in speech: move OUTWARD to the speech onset within SEARCH_SEC; only if there is
    no silence outward, move inward past speech (edge sits on a neighbour's tail).
    Nearest-either-way was wrong on noisy audio: it skipped the span's own words."""
    i, lim, stop = env.t2i(t), int(search_sec / HOP_SEC), env.t2i(span_end)
    if not env.speech[i]:
        j = _onset_after_silence(env, i)
        return (env.i2t(j), "moved-in") if j < stop else (t, "no-speech")
    out = next((d for d in range(lim + 1) if i - d - 1 < 0 or _silent_run(env, i - d - 1, -1)), None)
    inn = next((d for d in range(1, lim + 1) if _silent_run(env, i + d, +1)), None)
    if out is not None:      # outward first: moving inward past speech can drop the span's own words
        return env.i2t(i - out), "moved-out" if out else "unchanged"
    if inn is not None and outside_labelled:   # no silence outward AND the speech outside belongs
        j = _onset_after_silence(env, i + inn)     # to another labelled span: edge is on its tail
        if j < stop:
            return env.i2t(j), "moved-in-past-speech"
    return t, "no-boundary"


def snap_end(env: Envelope, t: float, span_start: float, outside_labelled: bool = True,
             search_sec: float = SEARCH_SEC) -> Tuple[float, str]:
    """Mirror of snap_start. t is an exclusive end; frame i-1 is the last one inside."""
    i, lim, stop = env.t2i(t), int(search_sec / HOP_SEC), env.t2i(span_start)
    if i == 0 or not env.speech[i - 1]:
        j = _offset_before_silence(env, i - 1)
        return (env.i2t(j), "moved-in") if j > stop else (t, "no-speech")
    out = next((d for d in range(lim + 1) if i + d >= len(env.speech) or _silent_run(env, i + d, +1)), None)
    inn = next((d for d in range(1, lim + 1) if i - 1 - d >= 0 and _silent_run(env, i - 1 - d, -1)), None)
    if out is not None:
        return env.i2t(i + out), "moved-out" if out else "unchanged"
    if inn is not None and outside_labelled:
        j = _offset_before_silence(env, i - 1 - inn)
        if j > stop:
            return env.i2t(j), "moved-in-past-speech"
    return t, "no-boundary"


def _runs(mask: np.ndarray) -> List[List[int]]:
    """[start, end) frame runs where mask is True, bridging gaps < MIN_SILENCE_SEC."""
    runs, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    merged: List[List[int]] = []
    for r in runs:                                 # bridge word-internal silences
        if merged and (r[0] - merged[-1][1]) * HOP_SEC < MIN_SILENCE_SEC:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return merged


def _covered(env: Envelope, intervals: List[Tuple[float, float]]) -> np.ndarray:
    covered = np.zeros(len(env.speech), bool)
    for s, e in intervals:
        covered[env.t2i(s): env.t2i(e) + 1] = True
    return covered


def uncovered_speech(env: Envelope, intervals: List[Tuple[float, float]],
                     min_sec: float = MIN_UNACCOUNTED_SEC) -> List[Tuple[float, float]]:
    """Speech (per env.speech) inside none of `intervals`, as (start, end) runs >= min_sec."""
    free = env.speech & ~_covered(env, intervals)
    return [(env.i2t(a), env.i2t(b)) for a, b in _runs(free) if (b - a) * HOP_SEC >= min_sec]


def possible_speech(env: Envelope, intervals: List[Tuple[float, float]],
                    min_sec: float = MIN_UNACCOUNTED_SEC) -> List[Tuple[float, float]]:
    """Loud audio that VAD rejected and no interval covers: speech VAD may have missed
    (elongated / sung words) or plain noise. For review, never auto-labelled."""
    if env.loud is None:
        return []
    free = env.loud & ~env.speech & ~_covered(env, intervals)
    return [(env.i2t(a), env.i2t(b)) for a, b in _runs(free) if (b - a) * HOP_SEC >= min_sec]
