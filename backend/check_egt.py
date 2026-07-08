import json
import os

docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs')

with open(os.path.join(docs_dir, 'cf42_egt.json'), encoding='utf-16') as f:
    data = json.load(f)
with open(os.path.join(docs_dir, 'egt_summary.txt'), 'w', encoding='utf-8') as out:
    for seg in data['egt'].get('segments', []):
        out.write(f"{seg['source_file']} [{seg['start_sec']:.1f}-{seg['end_sec']:.1f}] type={seg.get('segment_type')} bad_take={seg.get('is_bad_take')} text='{seg.get('transcript', '')}'\n")
