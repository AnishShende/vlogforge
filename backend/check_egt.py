import json
with open(r'd:\VlogForge\docs\cf42_egt.json', encoding='utf-16') as f:
    data = json.load(f)
with open(r'd:\VlogForge\docs\egt_summary.txt', 'w', encoding='utf-8') as out:
    for seg in data['egt'].get('segments', []):
        out.write(f"{seg['source_file']} [{seg['start_sec']:.1f}-{seg['end_sec']:.1f}] type={seg.get('segment_type')} bad_take={seg.get('is_bad_take')} text='{seg.get('transcript', '')}'\n")
