import json
with open(r'd:\VlogForge\docs\cf42_egt.json', encoding='utf-16') as f:
    egt = json.load(f)
with open(r'd:\VlogForge\docs\egt_summary.txt', 'w', encoding='utf-8') as out:
    for seg in egt.get('segments', []):
        out.write(f"{seg['source_file']} [{seg['start_sec']:.1f}-{seg['end_sec']:.1f}] type={seg['segment_type']} text='{seg.get('transcript', '')}'\n")
