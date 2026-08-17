import json

log_path = "/Users/anishshende/vlogforge/logs/20260713_164702_job_2944d6a7-40d2-46c0-9cde-694141acd92b.log"

import re
import ast

with open(log_path, 'r') as f:
    lines = f.readlines()

for line in lines:
    if "Phase 1 Reasoning CoT" in line:
        print("Found CoT!")
        
