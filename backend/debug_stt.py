import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils.llm import transcribe_audio_gemini

audio_path = "/Users/anishshende/vlogforge/uploads/aee0764b-ffb7-4e17-9ddb-a6a13df61641/audio/WhatsApp Video 2026-07-10 at 6.26.56 PM.wav"
segments = transcribe_audio_gemini(audio_path)
for s in segments:
    print(s)
