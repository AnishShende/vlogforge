import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
from app.config import settings

class InteractionLogger:
    def __init__(self):
        self.logger = logging.getLogger("VlogForge.Interaction")
        self.logger.setLevel(logging.INFO)
        
        # Prevent adding multiple handlers if instantiated multiple times
        if not self.logger.handlers:
            # Create daily log file
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
            log_file = os.path.join(settings.log_dir, f"interactions_{date_str}.log")
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s"
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def log_interaction(self, action: str, details: Dict[str, Any]):
        """Log a general user interaction with details."""
        try:
            details_str = json.dumps(details, default=str)
            self.logger.info(f"ACTION: {action} | DETAILS: {details_str}")
        except Exception as e:
            self.logger.error(f"Failed to log interaction {action}: {e}")

    def log_pipeline_completion(self, job_id: str, segments: List[Dict[str, Any]]):
        """Log the completion of a job with the rendered timeline's segments."""
        try:
            summary = f"Total Clips: {len(segments)}\n"
            for i, clip in enumerate(segments):
                summary += (f"  [{i}] {clip.get('src_in', 0):.1f}s - {clip.get('src_out', 0):.1f}s "
                            f"| File: {clip.get('source_file', '')}\n")
            self.logger.info(f"PIPELINE COMPLETED: {job_id}\n{summary}{'='*60}\n")
        except Exception as e:
            self.logger.error(f"Failed to log pipeline completion for {job_id}: {e}")

interaction_logger = InteractionLogger()
