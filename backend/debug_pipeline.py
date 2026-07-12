import sys
import os
import re
import argparse
import logging
import tempfile
import subprocess
from typing import List, Dict, Tuple
class Fore:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    WHITE = '\033[97m'

class Style:
    RESET_ALL = '\033[0m'

# Ensure backend modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.tasks.transcribe import get_whisper_model
from app.utils.llm import describe_keyframe, init_gemini
from app.utils.ffmpeg import get_ffmpeg_path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("DebugPipeline")

def extract_frame(video_path: str, timestamp_sec: float, output_path: str) -> bool:
    cmd = [
        get_ffmpeg_path(),
        "-y",
        "-ss", str(timestamp_sec),
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "2",
        output_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return os.path.exists(output_path)
    except Exception as e:
        logger.error(f"FFmpeg frame extraction failed: {e}")
        return False

def parse_log(log_path: str):
    raw_video_path = None
    output_video_path = None
    total_duration = 0.0
    egt_clips = []
    edl_clips = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        if "Ingesting video:" in line:
            raw_video_path = line.split("Ingesting video:")[1].strip()
        elif "Single-pass assembly complete:" in line:
            output_video_path = line.split("Single-pass assembly complete:")[1].strip()
        elif "Total Raw Duration:" in line:
            m = re.search(r'Total Raw Duration: ([\d\.]+)s', line)
            if m:
                total_duration = float(m.group(1))
        elif "]" in line and "s -" in line and "|" in line and "File:" in line:
            # Parse EDL clip: [0] 0.7s - 21.3s | File: ...
            m = re.search(r'\[\d+\] ([\d\.]+)s - ([\d\.]+)s \| File: .* \| Type: (.*)', line)
            if m:
                start = float(m.group(1))
                end = float(m.group(2))
                clip_type = m.group(3).strip()
                edl_clips.append({"start": start, "end": end, "type": clip_type})
        elif "]" in line and "s -" in line and "|" in line and "Type:" in line and "Quality:" in line:
            # Parse EGT clip: [0] 0.0s - 21.4s | Type: INTRO | Quality: 1.00 | Bad Take: False
            m = re.search(r'\[\d+\] ([\d\.]+)s - ([\d\.]+)s \| Type: (.*?) \| Quality:', line)
            if m:
                start = float(m.group(1))
                end = float(m.group(2))
                clip_type = m.group(3).strip()
                egt_clips.append({"start": start, "end": end, "type": clip_type})
                
    return raw_video_path, output_video_path, total_duration, egt_clips, edl_clips

def compute_removed_blocks(kept_clips: List[Dict], total_duration: float) -> List[Dict]:
    removed = []
    current_time = 0.0
    
    for clip in sorted(kept_clips, key=lambda x: x["start"]):
        if clip["start"] > current_time:
            removed.append({"start": current_time, "end": clip["start"]})
        current_time = max(current_time, clip["end"])
        
    if current_time < total_duration:
        removed.append({"start": current_time, "end": total_duration})
        
    return removed

def compute_reasoning_blocks(egt_clips: List[Dict], edl_clips: List[Dict]) -> List[Dict]:
    """Find blocks that were in EGT but removed by EDL"""
    reasoning_cuts = []
    
    # Simple strategy: for each EGT clip, subtract any overlapping EDL clips
    for egt in egt_clips:
        egt_start = egt["start"]
        egt_end = egt["end"]
        
        overlapping_edl = []
        for edl in edl_clips:
            # Check overlap
            overlap_start = max(egt_start, edl["start"])
            overlap_end = min(egt_end, edl["end"])
            if overlap_start < overlap_end:
                overlapping_edl.append((overlap_start, overlap_end))
                
        overlapping_edl.sort(key=lambda x: x[0])
        
        current = egt_start
        for (o_start, o_end) in overlapping_edl:
            if o_start > current:
                reasoning_cuts.append({"start": current, "end": o_start})
            current = max(current, o_end)
            
        if current < egt_end:
            reasoning_cuts.append({"start": current, "end": egt_end})
            
    return reasoning_cuts

def get_transcript(video_path: str):
    model = get_whisper_model()
    if not model:
        logger.error("Failed to load Whisper model")
        return []
        
    logger.info(f"{Fore.CYAN}Running Whisper on {os.path.basename(video_path)}...{Style.RESET_ALL}")
    segments, _ = model.transcribe(
        video_path,
        beam_size=5,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters=dict(
            min_speech_duration_ms=100,
            min_silence_duration_ms=500,
            speech_pad_ms=400
        ),
        word_timestamps=True
    )
    
    words = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                words.append({"start": w.start, "end": w.end, "text": w.word.strip()})
        else:
            words.append({"start": segment.start, "end": segment.end, "text": segment.text.strip()})
    return words

def get_words_in_range(words: List[Dict], start: float, end: float) -> str:
    # Filter words whose midpoint falls within the range
    in_range = [w["text"] for w in words if start <= (w["start"] + w["end"])/2 <= end]
    return " ".join(in_range)

def print_cut_blocks(title: str, blocks: List[Dict], raw_path: str, raw_words: List[Dict]):
    print("\n" + "="*60)
    print(f"{Fore.RED}=== {title} ==={Style.RESET_ALL}")
    
    if not blocks:
        print(f"{Fore.GREEN}No cuts in this phase.{Style.RESET_ALL}")
        return

    for idx, b in enumerate(blocks):
        print(f"\n{Fore.YELLOW}[Cut {idx+1}] {b['start']:.1f}s - {b['end']:.1f}s ({b['end']-b['start']:.1f}s){Style.RESET_ALL}")
        
        # Audio Content
        dialogue = get_words_in_range(raw_words, b["start"], b["end"])
        if dialogue:
            print(f"  {Fore.MAGENTA}Dialogue:{Style.RESET_ALL} \"{dialogue}\"")
        else:
            print(f"  {Fore.MAGENTA}Dialogue:{Style.RESET_ALL} [Silence]")
            
        # Visual Content
        midpoint = b["start"] + (b["end"] - b["start"]) / 2
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_img = tmp.name
        
        if extract_frame(raw_path, midpoint, tmp_img):
            print(f"  {Fore.CYAN}Analyzing visual frame at {midpoint:.1f}s...{Style.RESET_ALL}")
            try:
                desc = describe_keyframe(tmp_img, classify_content=False)
                print(f"  {Fore.CYAN}Visual:{Style.RESET_ALL} {desc}")
            except Exception as e:
                print(f"  {Fore.CYAN}Visual:{Style.RESET_ALL} [Error: {e}]")
            os.remove(tmp_img)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", help="Path to the job log file")
    args = parser.parse_args()
    
    if not os.path.exists(args.log_file):
        logger.error("Log file not found")
        sys.exit(1)
        
    init_gemini()
        
    raw_path, out_path, total_duration, egt_clips, edl_clips = parse_log(args.log_file)
    logger.info(f"{Fore.GREEN}=== Parsed Job Data ==={Style.RESET_ALL}")
    logger.info(f"Raw Video: {raw_path}")
    logger.info(f"Output Video: {out_path}")
    logger.info(f"Total Duration: {total_duration}s")
    logger.info(f"EGT Clips (Perception): {len(egt_clips)}")
    logger.info(f"EDL Clips (Reasoning): {len(edl_clips)}")
    
    # 1. Transcribe raw video
    raw_words = get_transcript(raw_path)
    raw_transcript_str = " ".join([w["text"] for w in raw_words])
    
    print("\n" + "="*60)
    print(f"{Fore.GREEN}=== FULL RAW AUDIO TRANSCRIPT ==={Style.RESET_ALL}")
    print(raw_transcript_str)
    
    # 2. Perception Cuts (Raw vs EGT)
    perception_cuts = compute_removed_blocks(egt_clips, total_duration)
    print_cut_blocks("REMOVED BY PERCEPTION (Silences, Dead Air, Boring)", perception_cuts, raw_path, raw_words)
    
    # 3. Reasoning Cuts (EGT vs EDL)
    reasoning_cuts = compute_reasoning_blocks(egt_clips, edl_clips)
    print_cut_blocks("REMOVED BY REASONING (Bad Takes, Ramble, Duplicates)", reasoning_cuts, raw_path, raw_words)

    print("\n" + "="*60)
    print(f"{Fore.GREEN}=== KEPT SECTIONS (Expected vs Actual Final Output) ==={Style.RESET_ALL}")
    
    # Transcribe output video
    out_words = get_transcript(out_path)
    
    # Expected output
    expected_full = []
    for c in sorted(edl_clips, key=lambda x: x["start"]):
        words = get_words_in_range(raw_words, c["start"], c["end"])
        if words:
            expected_full.append(words)
            
    expected_transcript_str = " ".join(expected_full)
    actual_transcript_str = " ".join([w["text"] for w in out_words])
    
    print(f"\n{Fore.YELLOW}Expected Transcript (based on EDL applied to raw video):{Style.RESET_ALL}")
    print(expected_transcript_str)
    
    print(f"\n{Fore.YELLOW}Actual Transcript (from final assembled output video):{Style.RESET_ALL}")
    print(actual_transcript_str)
    
    # Simple word count diff
    exp_count = len(expected_transcript_str.split())
    act_count = len(actual_transcript_str.split())
    
    print("\n" + "="*60)
    print(f"{Fore.WHITE}Word Count Validation: Expected {exp_count} words | Actual {act_count} words{Style.RESET_ALL}")
    if abs(exp_count - act_count) <= 5:
        print(f"{Fore.GREEN}PASSED: Audio transcript is highly synchronized and accurate.{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}WARNING: Significant difference in expected vs actual transcript. A/V sync or drift issues likely.{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
