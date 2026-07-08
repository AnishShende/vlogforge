import json
import asyncio
from app.models import EGTDocument
from app.tasks.edl import generate_edl
from app.utils.llm import init_gemini

import os

docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs')

init_gemini()
with open(os.path.join(docs_dir, 'latest_egt_output.json'), 'r', encoding='utf-8-sig') as f:
    egt_json = json.load(f)
egt_doc = EGTDocument(**egt_json['egt'])

print(f"Loaded {len(egt_doc.segments)} segments.")
# Skip score_segments to avoid exceeding 5 RPM Free Tier limit

print('Running generate_edl with 2.5 Flash...')
edl_dicts = generate_edl(egt_doc, transcript_segments=None)
print('EDL Size:', len(edl_dicts))

with open(os.path.join(docs_dir, 'phase1_edl_test.json'), 'w', encoding='utf-8') as f:
    json.dump(edl_dicts, f, indent=2)
print('Done!')
