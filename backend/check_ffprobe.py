import json
import os
import glob
from app.utils.ffmpeg import get_video_info
from app.config import settings

# Find any mp4 file in upload_dir
mp4_files = glob.glob(os.path.join(settings.upload_dir, '**', '*.mp4'), recursive=True)
if not mp4_files:
    print("No mp4 files found.")
else:
    info = get_video_info(mp4_files[0])
    print("File:", mp4_files[0])
    print(json.dumps(info.get('format', {}).get('tags', {}), indent=2))
