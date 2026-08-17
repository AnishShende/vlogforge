class EDLEntry:
    def __init__(self, start_sec, end_sec, core_start_sec, core_end_sec, narrative_priority):
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.core_start_sec = core_start_sec
        self.core_end_sec = core_end_sec
        self.narrative_priority = narrative_priority
        self.quality_score = 1.0

entries = [
    EDLEntry(0.0, 21.4, 0.0, 21.4, "MEDIUM"),
    EDLEntry(21.4, 133.8, 21.4, 133.8, "MEDIUM"),
    EDLEntry(133.8, 147.4, 133.8, 147.4, "MEDIUM"),
    EDLEntry(147.4, 245.3, 147.4, 245.3, "MEDIUM"),
    EDLEntry(245.3, 281.2, 245.3, 281.2, "MEDIUM")
]

target_duration = 3.0

def get_total_dur(edl_list):
    return sum(e.end_sec - e.start_sec for e in edl_list)

print("Initial total duration:", get_total_dur(entries))

if target_duration:
    if get_total_dur(entries) > target_duration:
        for e in entries:
            if e.narrative_priority in ["LOW", "MEDIUM"]:
                e.start_sec = e.core_start_sec
                e.end_sec = e.core_end_sec

    if get_total_dur(entries) > target_duration:
        low_clips = sorted([e for e in entries if e.narrative_priority == "LOW"], key=lambda x: x.quality_score)
        for e in low_clips:
            if get_total_dur(entries) <= target_duration:
                break
            entries.remove(e)

    if get_total_dur(entries) > target_duration:
        med_clips = sorted([e for e in entries if e.narrative_priority == "MEDIUM"], key=lambda x: x.quality_score)
        for e in med_clips:
            if get_total_dur(entries) <= target_duration:
                break
            entries.remove(e)

    if get_total_dur(entries) > target_duration:
        for e in entries:
            if e.narrative_priority == "CRITICAL":
                e.start_sec = e.core_start_sec
                e.end_sec = e.core_end_sec

    final_dur = get_total_dur(entries)
    if final_dur > target_duration + 0.1:
        print("Budget Exceeded warning!")

print("Final total duration:", final_dur)
print("Entries left:", len(entries))

