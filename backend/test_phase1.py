import json
import asyncio
from app.models import EGTDocument
from app.tasks.edl import generate_edl
from app.utils.llm import init_gemini

init_gemini()
with open('d:/VlogForge/docs/latest_egt_output.json', 'r', encoding='utf-8-sig') as f:
    egt_json = json.load(f)
egt_doc = EGTDocument(**egt_json['egt'])

print(f"Loaded {len(egt_doc.segments)} segments.")
# Skip score_segments to avoid exceeding 5 RPM Free Tier limit

print('Running generate_edl with 2.5 Flash...')
edl_dicts = generate_edl(egt_doc, transcript_segments=None)
print('EDL Size:', len(edl_dicts))

with open('d:/VlogForge/docs/phase1_edl_test.json', 'w', encoding='utf-8') as f:
    json.dump(edl_dicts, f, indent=2)
print('Done!')
