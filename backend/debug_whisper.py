import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.tasks.transcribe import get_whisper_model

model = get_whisper_model()
audio_path = "/Users/anishshende/vlogforge/uploads/aee0764b-ffb7-4e17-9ddb-a6a13df61641/audio/WhatsApp Video 2026-07-10 at 6.26.56 PM.wav"
segments, _ = model.transcribe(audio_path, beam_size=5)
for s in segments:
    print(f"[{s.start:.2f} - {s.end:.2f}] {s.text.strip()}")
