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

    def log_pipeline_completion(self, job_id: str, egt_data: Dict[str, Any], edl_data: List[Dict[str, Any]]):
        """Log the completion of a job, including pretty-printed EGT and EDL outputs."""
        try:
            # EGT Summary
            egt_summary = f"\n=== EGT (Editorial Ground Truth) SUMMARY for Job {job_id} ===\n"
            total_duration = egt_data.get("total_duration_sec", 0.0)
            segments = egt_data.get("segments", [])
            egt_summary += f"Total Raw Duration: {total_duration:.2f}s | Segments: {len(segments)}\n"
            
            for i, seg in enumerate(segments):
                egt_summary += (
                    f"  [{i}] {seg.get('start_sec', 0):.1f}s - {seg.get('end_sec', 0):.1f}s "
                    f"| Type: {seg.get('segment_type', 'UNKNOWN')} "
                    f"| Quality: {seg.get('quality_score', 0):.2f} "
                    f"| Bad Take: {seg.get('is_bad_take', False)}\n"
                    f"      Transcript: '{seg.get('transcript', '')}'\n"
                    f"      Tags: {', '.join(seg.get('tags', []))}\n"
                )
            
            # EDL Summary
            edl_summary = f"\n=== EDL (Edit Decision List) SUMMARY for Job {job_id} ===\n"
            edl_summary += f"Total Clips: {len(edl_data)}\n"
            
            for i, clip in enumerate(edl_data):
                edl_summary += (
                    f"  [{i}] {clip.get('start_sec', 0):.1f}s - {clip.get('end_sec', 0):.1f}s "
                    f"| File: {clip.get('source_file', '')} "
                    f"| Type: {clip.get('editorial_type', 'KEEP')}\n"
                )
                
            self.logger.info(f"PIPELINE COMPLETED: {job_id}\n{egt_summary}\n{edl_summary}\n{'='*60}\n")
        except Exception as e:
            self.logger.error(f"Failed to log pipeline completion for {job_id}: {e}")

interaction_logger = InteractionLogger()
