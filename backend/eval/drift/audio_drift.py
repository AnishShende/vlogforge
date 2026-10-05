"""Phase 0 S1: measure time offset between the analysis WAV (word timeline) and
(a) the original MOV audio, (b) clips actually rendered by process_clip."""
import subprocess, sys, os, numpy as np, wave
sys.path.insert(0, "/Users/anishshende/vlogforge/backend")
from app.utils.ffmpeg import process_clip

ROOT = "/Users/anishshende/vlogforge/uploads/516a8ef1-85c6-43d8-9817-7ad1a10b9e93"
ORIG = f"{ROOT}/raw/IMG_1614.MOV"
WAV = f"{ROOT}/audio/IMG_1614.wav"
OUT = os.environ.get("DRIFT_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out")); os.makedirs(OUT, exist_ok=True)
SR = 16000

def load_wav(p):
    with wave.open(p) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)

def to_wav(src, dst):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-vn", "-ac", "1", "-ar", str(SR),
                    "-acodec", "pcm_s16le", dst], check=True)

def offset_ms(ref, sig, ref_start, max_lag_s=0.3):
    """Lag (ms) such that sig[i] ~= ref[ref_start + i + lag]. Positive => sig is later in ref."""
    L = len(sig); m = int(max_lag_s * SR)
    lo = max(0, ref_start - m); seg = ref[lo: ref_start + L + m]
    a = (sig - sig.mean()); b = seg - seg.mean()
    c = np.correlate(b, a, mode="valid")
    k = int(np.argmax(c)); lag = (lo + k) - ref_start
    norm = c[k] / (np.linalg.norm(a) * np.linalg.norm(b[k:k+L]) + 1e-9)
    return lag * 1000 / SR, norm

wav = load_wav(WAV)
orig_wav = f"{OUT}/orig_16k.wav"; to_wav(ORIG, orig_wav); orig = load_wav(orig_wav)
print(f"analysis wav: {len(wav)/SR:.4f}s   original audio: {len(orig)/SR:.4f}s   diff {1000*(len(wav)-len(orig))/SR:.1f}ms")

probe_times = [5, 40, 80, 120, 160, 200, 240, 280, 320, 345]
print("\n(a) analysis-WAV time vs ORIGINAL audio time (2s windows)")
print(f"{'t_sec':>6} {'offset_ms':>10} {'corr':>6}")
a_offsets = []
for t in probe_times:
    s = int(t * SR); win = orig[s: s + 2 * SR]
    off, r = offset_ms(wav, win, s)
    a_offsets.append(off); print(f"{t:>6} {off:>10.2f} {r:>6.3f}")

print("\n(b) process_clip(original, t, t+3) output vs analysis-WAV at t")
print(f"{'t_sec':>6} {'offset_ms':>10} {'corr':>6}")
b_offsets = []
for t in probe_times:
    clip = f"{OUT}/clip_{t}.mp4"; cw = f"{OUT}/clip_{t}.wav"
    assert process_clip(ORIG, t, t + 3.0, clip), f"process_clip failed at {t}"
    to_wav(clip, cw); c = load_wav(cw)[: 2 * SR]
    off, r = offset_ms(wav, c, int(t * SR))
    b_offsets.append(off); print(f"{t:>6} {off:>10.2f} {r:>6.3f}")

for name, o in (("(a) wav vs original", a_offsets), ("(b) rendered clip vs wav", b_offsets)):
    o = np.array(o); print(f"\n{name}: mean {o.mean():.2f}ms  min {o.min():.2f}  max {o.max():.2f}  spread {o.max()-o.min():.2f}ms")
