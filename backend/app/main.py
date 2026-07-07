import os
import uuid
import logging
import shutil
from typing import List, Optional
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.config import settings
import asyncio
from pydantic import BaseModel
from app.models import JobStatus, VideoFileInfo, EDLEntry
from app.tasks.orchestrator import (
    jobs_db,
    jobs_data_db,
    get_job,
    get_job_data,
    register_websocket,
    unregister_websocket,
    start_pipeline,
    broadcast_progress
)
from app.utils.interaction_logger import interaction_logger

date_str = datetime.utcnow().strftime("%Y-%m-%d")
app_log_file = os.path.join(settings.log_dir, f"app_{date_str}.log")

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(app_log_file, encoding='utf-8')
    ]
)
logger = logging.getLogger("VlogForge.API")

app = FastAPI(title="VlogForge Backend API", version="2.0.0")

# CORS setup for local React + Vite development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the VlogForge AI-Powered Vlog Creation API!", "version": "2.0.0-phase0"}

@app.post("/api/jobs", response_model=JobStatus)
async def create_job(
    files: List[UploadFile] = File(...),
    context_text: Optional[str] = Form(""),
    vlog_genre: Optional[str] = Form("default"),
    target_duration: Optional[float] = Form(10.0),
    quality_threshold: Optional[float] = Form(0.35)
):
    """Create a new editing job. Save uploaded raw videos and kick off background execution."""
    job_id = str(uuid.uuid4())
    logger.info(f"Creating job {job_id} with {len(files)} uploaded files, genre: {vlog_genre}...")
    interaction_logger.log_interaction("create_job", {
        "job_id": job_id,
        "files_count": len(files),
        "vlog_genre": vlog_genre,
        "target_duration": target_duration,
        "quality_threshold": quality_threshold
    })

    # Create job directories
    job_raw_dir = os.path.join(settings.upload_dir, job_id, "raw")
    os.makedirs(job_raw_dir, exist_ok=True)

    saved_file_paths = []
    files_info = []

    for uploaded_file in files:
        # Validate format
        ext = os.path.splitext(uploaded_file.filename)[1].lower()
        if ext not in [".mp4", ".mov", ".avi", ".mkv"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {uploaded_file.filename}. Please upload MP4, MOV, AVI or MKV."
            )

        saved_path = os.path.join(job_raw_dir, uploaded_file.filename)
        try:
            with open(saved_path, "wb") as buffer:
                shutil.copyfileobj(uploaded_file.file, buffer)
            saved_file_paths.append(saved_path)

            # Record preliminary file details (duration calculated later in ingest stage)
            file_info = VideoFileInfo(
                filename=uploaded_file.filename,
                original_path=saved_path,
                size_bytes=os.path.getsize(saved_path)
            )
            files_info.append(file_info)
        except Exception as e:
            logger.error(f"Failed to save uploaded file {uploaded_file.filename}: {e}")
            raise HTTPException(status_code=500, detail="Failed to save video uploads.")

    # Create the initial job status in DB
    job_status = JobStatus(
        job_id=job_id,
        status="pending",
        progress=0,
        message="Job created. Ready to ingest.",
        files=files_info,
        context_text=context_text,
        vlog_genre=vlog_genre,
        target_duration=target_duration,
        quality_threshold=quality_threshold,
        created_at=datetime.utcnow()
    )
    jobs_db[job_id] = job_status

    # Start the async pipeline orchestrator
    await start_pipeline(job_id, saved_file_paths, context_text, target_duration, vlog_genre, quality_threshold)

    return job_status

@app.get("/api/jobs/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str):
    """Get the current job status details."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.post("/api/jobs/{job_id}/cancel")
def cancel_job_endpoint(job_id: str):
    """Cancel a running editing job."""
    from app.tasks.orchestrator import cancel_job
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    interaction_logger.log_interaction("cancel_job", {"job_id": job_id, "current_status": job.status})

    if job.status in ["complete", "failed", "cancelled"]:
        return {"status": job.status, "message": "Job cannot be cancelled in its current state."}

    cancel_job(job_id)
    return {"status": "cancelled", "message": "Cancellation request received."}

@app.get("/api/jobs/{job_id}/transcript")
def get_job_transcript(job_id: str):
    """Retrieve full transcript / EGT segments with classification labels."""
    job_data = get_job_data(job_id)
    if not job_data or "transcript" not in job_data:
        raise HTTPException(status_code=404, detail="Transcript data not available for this job yet.")
    return {
        "job_id": job_id,
        "transcript": job_data["transcript"],
        "context_document": job_data.get("context_document", "")
    }

@app.get("/api/jobs/{job_id}/egt")
def get_job_egt(job_id: str):
    """Retrieve the full Editorial Ground Truth (EGT) document.

    The EGT is the canonical, inspectable perception output — every segment
    with its clip_id, quality_score, quality_flags, segment_type, and tags.
    """
    job_data = get_job_data(job_id)
    if not job_data or "egt" not in job_data:
        raise HTTPException(status_code=404, detail="EGT data not available for this job yet.")
    return {
        "job_id": job_id,
        "egt": job_data["egt"]
    }

@app.get("/api/jobs/{job_id}/edl")
def get_job_edl(job_id: str):
    """Retrieve the Edit Decision List (EDL) JSON."""
    job_data = get_job_data(job_id)
    if not job_data or "edl" not in job_data:
        raise HTTPException(status_code=404, detail="EDL data not available for this job yet.")
    return {
        "job_id": job_id,
        "edl": job_data["edl"]
    }

@app.get("/api/jobs/{job_id}/download")
def download_vlog(job_id: str):
    """Download the final processed vlog MP4."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != "complete":
        raise HTTPException(status_code=400, detail="Vlog has not completed processing.")

    interaction_logger.log_interaction("download_vlog", {"job_id": job_id})

    final_video_path = os.path.join(settings.output_dir, f"{job_id}.mp4")
    if not os.path.exists(final_video_path):
        raise HTTPException(status_code=404, detail="Final video output file not found on disk.")

    return FileResponse(
        path=final_video_path,
        media_type="video/mp4",
        filename=f"vlogforge_edit_{job_id[:8]}.mp4"
    )


class ReRenderRequest(BaseModel):
    edl: List[dict]  # Accepts both legacy and new EDL formats

class ReReasonRequest(BaseModel):
    quality_threshold: float = 0.35


@app.get("/api/jobs/{job_id}/raw/{filename}")
def get_raw_video(job_id: str, filename: str):
    """Serve raw uploaded video files."""
    raw_video_path = os.path.join(settings.upload_dir, job_id, "raw", filename)
    if not os.path.exists(raw_video_path):
        raise HTTPException(status_code=404, detail="Raw video file not found.")
    return FileResponse(
        path=raw_video_path,
        media_type="video/mp4",
        filename=filename
    )

@app.post("/api/jobs/{job_id}/re-render")
async def re_render_job(job_id: str, request: ReRenderRequest):
    """Re-assemble the vlog using a modified Edit Decision List (EDL).

    This only re-runs Pass 3 (mechanical assembly) — never re-triggers
    Pass 1 (perception) or Pass 2 (reasoning). Fast iteration.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    interaction_logger.log_interaction("re_render_job", {"job_id": job_id, "edl_items_count": len(request.edl)})

    job_dir = os.path.join(settings.upload_dir, job_id)
    final_video_name = f"{job_id}.mp4"
    final_video_path = os.path.join(settings.output_dir, final_video_name)

    # Update EDL in job data store
    job_data = get_job_data(job_id)
    if not job_data:
        jobs_data_db[job_id] = {}
        job_data = jobs_data_db[job_id]

    # Store the new EDL items
    new_edl = request.edl
    job_data["edl"] = new_edl

    # Get EGT clip_ids for validation (if EGT available)
    egt_clip_ids = None
    if "egt" in job_data and "segments" in job_data["egt"]:
        egt_clip_ids = {seg["clip_id"] for seg in job_data["egt"]["segments"]}

    # Update job state and broadcast re-assembling status
    await broadcast_progress(
        job_id,
        "assembling",
        90,
        "Re-assembling video clips with FFmpeg..."
    )

    # Execute the assembly task in a thread pool
    try:
        from app.tasks.assemble import assemble_vlog
        files_info = [f.model_dump() for f in job.files]

        loop = asyncio.get_running_loop()
        success = await loop.run_in_executor(
            None,
            assemble_vlog,
            new_edl,
            files_info,
            job_dir,
            final_video_path,
            egt_clip_ids,
        )

        if not success:
            await broadcast_progress(job_id, "failed", 0, "Re-assembly failed.")
            raise HTTPException(status_code=500, detail="FFmpeg vlog assembly failed.")

        download_url = f"/api/jobs/{job_id}/download"
        await broadcast_progress(
            job_id,
            "complete",
            100,
            "Vlog re-rendered successfully!",
            download_url=download_url
        )

        return {
            "status": "complete",
            "download_url": download_url
        }
    except Exception as e:
        logger.error(f"Re-render failed for job {job_id}: {e}", exc_info=True)
        await broadcast_progress(job_id, "failed", 0, f"Re-render error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Re-render failed: {str(e)}")

@app.post("/api/jobs/{job_id}/re-reason")
async def re_reason_job_endpoint(job_id: str, request: ReReasonRequest):
    """Re-run the Pass 2 LLM Reasoning step with a new quality threshold.
    
    This avoids re-running the expensive Pass 1 perception. It recalculates the 
    `is_bad_take` flags in the EGT based on the new threshold, and then generates 
    a new EDL using the LLM.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    interaction_logger.log_interaction("re_reason_job", {"job_id": job_id, "new_quality_threshold": request.quality_threshold})

    job.quality_threshold = request.quality_threshold

    # Trigger re-reasoning pipeline
    from app.tasks.orchestrator import start_re_reasoning
    await start_re_reasoning(job_id, request.quality_threshold)

    return {"status": "re-reasoning", "message": "Re-reasoning job started."}

@app.websocket("/ws/{job_id}")
async def websocket_progress_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint to push real-time status and progress updates to client."""
    await websocket.accept()

    # If job is already complete or failed, send immediate final state
    job = get_job(job_id)
    if job:
        await websocket.send_json({
            "stage": job.status,
            "progress": job.progress,
            "message": job.message,
            "download_url": job.output_video_url
        })

    register_websocket(job_id, websocket)
    try:
        # Keep connection open and listen for client messages
        while True:
            # We don't expect messages from client, but keeping it open
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from job: {job_id}")
    finally:
        unregister_websocket(job_id, websocket)
