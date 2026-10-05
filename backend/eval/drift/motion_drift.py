"""Phase 0 S1 video: lag between ORIGINAL and (proxy, rendered clip) using each stream's
own frame-to-frame motion energy (geometry-independent), timestamps from the streams."""
import subprocess, os, numpy as np
OUT = os.environ.get("DRIFT_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out")); os.makedirs(OUT, exist_ok=True)
ROOT = "/Users/anishshende/vlogforge/uploads/516a8ef1-85c6-43d8-9817-7ad1a10b9e93"
ORIG = f"{ROOT}/raw/IMG_1614.MOV"; PROXY = f"{ROOT}/proxy/IMG_1614_proxy.mp4"
W, H = 48, 48

def motion(src, t0, dur):
    """Return (timestamps, motion) for native frames of src in [t0, t0+dur). Times are source PTS."""
    ts = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", f"{max(0,t0-2)}%{t0+dur+1}",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "csv=p=0", src], capture_output=True, text=True).stdout
    ts = np.array(sorted(float(x.strip(",")) for x in ts.split() if x.strip(",")))
    ts = ts[(ts >= t0) & (ts < t0 + dur)]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{ts[0]:.6f}", "-i", src, "-frames:v", str(len(ts)),
        "-vsync", "passthrough", "-vf", f"scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        check=True, capture_output=True).stdout
    f = np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(np.float32)
    n = min(len(f), len(ts)); f, ts = f[:n], ts[:n]
    m = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))
    return ts[1:], m

def lag_ms(ta, ma, tb, mb, max_lag=0.3, step=0.001):
    """Lag L (s) maximizing corr(ma(t), mb(t + L)). Positive => b is later."""
    grid = np.arange(max(ta[0], tb[0]) + max_lag, min(ta[-1], tb[-1]) - max_lag, step)
    A = np.interp(grid, ta, ma); A = (A - A.mean()) / (A.std() + 1e-9)
    best = (-9, 0)
    for L in np.arange(-max_lag, max_lag + step / 2, step):
        B = np.interp(grid + L, tb, mb); B = (B - B.mean()) / (B.std() + 1e-9)
        best = max(best, ((A * B).mean(), L))
    return best[1] * 1000, best[0]

print(f"{'t':>5} {'vs':>8} {'lag_ms':>8} {'corr':>6}   (orig frames, motion std)")
for t in (5, 80, 120, 200, 320):
    to, mo = motion(ORIG, t, 3.0)
    tp, mp = motion(PROXY, t, 3.0)
    tc, mc = motion(f"{OUT}/clip_{t}.mp4", 0, 3.0); tc = tc + t   # clip t=0 is source t
    for name, tb, mb in (("proxy", tp, mp), ("render", tc, mc)):
        L, r = lag_ms(to, mo, tb, mb)
        print(f"{t:>5} {name:>8} {L:>8.1f} {r:>6.3f}   ({len(to)+1} frames, {mo.std():.2f})")
