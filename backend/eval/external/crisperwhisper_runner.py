"""CrisperWhisper 2.0 transcription for the ASR bake-off (candidate F). Evaluation only.

Model weights are under the Nyra Health Non-Commercial Research License: research /
evaluation use, NOT production. Runs in its own virtualenv (torch/transformers
versions differ from the main backend env), writes JSON the bake-off scores.

Usage (inside the CrisperWhisper venv):
    python crisperwhisper_runner.py --model turbo --out OUT.json --wav A.wav [--wav B.wav] [--repeat 2]
"""
import argparse, json, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="turbo")            # turbo | large | medium | small
    ap.add_argument("--wav", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--language", default="en")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args()
    from crisperwhisper import CrisperWhisperModel
    t = time.time()
    try:
        model = CrisperWhisperModel(f"nyralabs/CrisperWhisper2.0_{a.model}", backend="transformers",
                                    device=a.device, compute_type="float16")
    except Exception as e:                                   # MPS unsupported -> CPU float32
        print(f"[cw] {a.device} load failed ({e}); falling back to cpu/float32", flush=True)
        model = CrisperWhisperModel(f"nyralabs/CrisperWhisper2.0_{a.model}", backend="transformers",
                                    device="cpu", compute_type="float32")
    print(f"[cw] model {a.model} loaded in {time.time() - t:.1f}s", flush=True)
    out = {}
    for wav in a.wav:
        runs = []
        for i in range(a.repeat):
            t = time.time()
            r = model.transcribe(wav, language=a.language, mode="verbatim", word_timestamps=True,
                                 temperature_fallback=False)    # deterministic, like candidate E
            dt = time.time() - t
            words = [{"start": float(w.start), "end": float(w.end), "text": w.word.strip()}
                     for w in (r.words or []) if w.word.strip() and w.start is not None]
            runs.append({"runtime_sec": round(dt, 1), "words": words, "text": r.text})
            print(f"[cw] {wav.split('/')[-1]} run {i + 1}: {dt:.1f}s, {len(words)} words", flush=True)
        out[wav] = runs
    json.dump({"model": a.model, "language": a.language, "device": a.device, "results": out}, open(a.out, "w"))
    print(f"[cw] wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
