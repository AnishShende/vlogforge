import os
import subprocess
import pytest

from app.utils.ffmpeg import assemble_single_pass
from app.tasks.transcribe import get_whisper_model

@pytest.fixture
def test_video(tmp_path):
    """Generate a 5-second test video with 3 distinct spoken words separated by silence."""
    video_path = str(tmp_path / "test_vid.mp4")
    audio_path = str(tmp_path / "test_audio.aiff")
    
    # Generate speech audio using macOS 'say'
    # "clip one" -> 1s silence -> "clip two" -> 1s silence -> "clip three"
    say_cmd = [
        "say", "-o", audio_path,
        "clip one. [[slnc 1000]] clip two. [[slnc 1000]] clip three."
    ]
    subprocess.run(say_cmd, check=True)
    
    # Generate a dummy video track and multiplex
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=1280x720:r=30",
        "-i", audio_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-shortest",
        video_path
    ]
    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
    
    return video_path


def test_e2e_transcript_sync(test_video, tmp_path):
    """
    Test that cutting a video accurately retains ONLY the intended speech.
    This also implicitly tests A/V sync because if the audio duration shrank, 
    the cut boundaries would drift and include wrong words.
    """
    # "clip one" is ~0-1s. Silence 1-2s. "clip two" is ~2-3s. Silence 3-4s. "clip three" is ~4-5s.
    # Let's extract exactly the 1.5s to 3.5s mark. This should catch ONLY "clip two".
    edl = [
        {
            "video_file": "test_vid.mp4",
            "start_sec": 1.5,
            "end_sec": 3.5,
        }
    ]
    file_map = {"test_vid.mp4": test_video}
    
    out_path = str(tmp_path / "out.mp4")
    
    # Run assembly
    success = assemble_single_pass(edl, file_map, out_path)
    assert success is True
    assert os.path.exists(out_path)
    
    # Transcribe output
    model = get_whisper_model()
    assert model is not None, "Whisper model not available for test"
    
    segments, _ = model.transcribe(out_path)
    text = " ".join([s.text.strip() for s in segments]).lower()
    
    # Verify "clip two" is present, but "clip one" and "clip three" are completely gone
    assert "two" in text or "2" in text, f"Expected 'two' or '2' in transcript, got: {text}"
    assert "one" not in text and "1" not in text, f"Did not expect 'one' or '1' in transcript, got: {text}"
    assert "three" not in text and "3" not in text, f"Did not expect 'three' or '3' in transcript, got: {text}"
