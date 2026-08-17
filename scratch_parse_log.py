import sys
import re
from datetime import datetime

def get_times(log_file):
    times = {}
    with open(log_file, "r") as f:
        for line in f:
            match = re.search(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+\[.*?\]\s+([^:]+): (.*)", line)
            if match:
                ts_str = match.group(1)
                msg = match.group(3)
                ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f")
                
                if "STARTING PIPELINE JOB" in msg: times["start"] = ts
                elif "CFR transcode complete" in msg or "Proxy generation complete" in msg: times["proxy"] = ts
                elif "Starting Whisper transcription" in msg: times["whisper_start"] = ts
                elif "Transcription complete" in msg: times["whisper_end"] = ts
                elif "Detecting scenes for" in msg: times["scene_start"] = ts
                elif "Scene detection complete" in msg: times["scene_end"] = ts
                elif "Visual analysis complete" in msg: times["visual_end"] = ts
                elif "Batch semantic classification complete" in msg: times["score_end"] = ts
                elif "Phase 1 Map complete" in msg or "Map phase complete" in msg: times["map_end"] = ts
                elif "Reduce phase complete:" in msg: times["reduce_end"] = ts
                elif "Starting video assembly..." in msg: times["assembly_start"] = ts
                elif "Vlog assembly completed" in msg or "Single-pass assembly complete" in msg: times["assembly_end"] = ts
                elif "PIPELINE JOB COMPLETED SUCCESSFULLY" in msg: times["end"] = ts
    return times

def print_diffs(file1, file2):
    t1 = get_times(file1)
    t2 = get_times(file2)
    
    stages = [
        ("Proxy Generation", "start", "proxy"),
        ("Whisper", "whisper_start", "whisper_end"),
        ("Scene Detect", "scene_start", "scene_end"),
        ("Visual Analysis", "scene_end", "visual_end"),
        ("Quality Scoring", "visual_end", "score_end"),
        ("EDL Map Phase", "score_end", "map_end"),
        ("EDL Reduce Phase", "map_end", "reduce_end"),
        ("Assembly", "assembly_start", "assembly_end"),
        ("Total Pipeline", "start", "end")
    ]
    
    print(f"{'Stage':<20} | {'Old Time':<10} | {'New Time':<10} | {'Improvement'}")
    print("-" * 65)
    for name, start_evt, end_evt in stages:
        d1 = 0
        if start_evt in t1 and end_evt in t1:
            d1 = (t1[end_evt] - t1[start_evt]).total_seconds()
            
        d2 = 0
        if start_evt in t2 and end_evt in t2:
            d2 = (t2[end_evt] - t2[start_evt]).total_seconds()
            
        if d1 > 0 or d2 > 0:
            diff = d1 - d2
            print(f"{name:<20} | {d1:>8.1f}s | {d2:>8.1f}s | {diff:>8.1f}s")

file1 = "/Users/anishshende/vlogforge/logs/20260817_183431_job_30199cdf-f64c-44e7-822b-6dfcb6beaf29.log"
file2 = "/Users/anishshende/vlogforge/logs/20260817_222000_job_f35f0c90-e8fb-4d43-a7d0-fbe58326c89a.log"

print_diffs(file1, file2)
