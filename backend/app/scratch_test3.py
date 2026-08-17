from app.tasks.edl import EDLEntry

entries = [
    EDLEntry(
        clip_id="1",
        source_file="file1.mp4",
        start_sec=0.7,
        end_sec=21.5,
        core_start_sec=0.7,
        core_end_sec=21.5,
        narrative_priority="MEDIUM",
        quality_score=1.0,
        editorial_type="INTRO",
        sequence_index=0
    ),
    EDLEntry(
        clip_id="2",
        source_file="file1.mp4",
        start_sec=21.5,
        end_sec=132.8,
        core_start_sec=21.5,
        core_end_sec=132.8,
        narrative_priority="MEDIUM",
        quality_score=1.0,
        editorial_type="KEEP",
        sequence_index=1
    )
]

target_duration = 3.0

def get_total_dur(edl_list):
    return sum(e.end_sec - e.start_sec for e in edl_list)

print("Before Phase C:", get_total_dur(entries))

if get_total_dur(entries) > target_duration:
    med_clips = sorted([e for e in entries if e.narrative_priority == "MEDIUM"], key=lambda x: x.quality_score)
    for e in med_clips:
        if get_total_dur(entries) <= target_duration:
            break
        entries.remove(e)

print("After Phase C:", get_total_dur(entries))

