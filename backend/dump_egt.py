import json
import os
import sys

# Since jobs_data_db is in-memory, we can't easily access it if the server was restarted.
# But if it's still running, I might not have access to it directly from python script.
# Wait, let's see if the server writes the EGT to a file, or if we can fetch it via API.
import urllib.request

job_id = "37b723b1-6096-40b6-bb28-653afa189077"
try:
    with urllib.request.urlopen(f"http://localhost:8000/api/jobs/{job_id}/egt") as response:
        egt = json.loads(response.read().decode())
        with open("egt_dump.json", "w", encoding="utf-8") as f:
            json.dump(egt, f, indent=2)
    print("EGT fetched via API")
except Exception as e:
    print("API failed:", e)
    # Check if there's a file saved? No, orchestrator only saves to memory: jobs_data_db[job_id]
