import os
import logging
from typing import Optional
from datetime import datetime
from fastapi import FastAPI, Form, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv
load_dotenv()
from app.config import settings
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models import JobStatus, VideoFileInfo, EditPlan
from app.db_models import VideoFile, Project
from app.database import get_db
from app.tasks.orchestrator import (
    jobs_db,
    get_job,
    register_websocket,
    unregister_websocket,
    start_pipeline,
    broadcast_progress,
    cancel_job,
    start_recompile,
)
from app.utils.interaction_logger import interaction_logger

# Initialize global stdout logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VlogForge.API")

app = FastAPI(title="VlogForge Backend API", version="2.0.0")

from app.routers import auth, projects, upload
app.include_router(auth.router)
app.include_router(projects.router)
app.include_router(upload.router)

# CORS setup for local React + Vite development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "Upload-Offset", 
        "Location", 
        "Upload-Length", 
        "Tus-Version", 
        "Tus-Resumable", 
        "Tus-Max-Size", 
        "Tus-Extension", 
        "Upload-Metadata", 
        "Upload-Defer-Length", 
        "Upload-Concat"
    ],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the VlogForge AI-Powered Vlog Creation API!", "version": "2.0.0"}

@app.post("/api/jobs", response_model=JobStatus)
async def create_job(
    project_id: str = Form(...),
    context_text: Optional[str] = Form(""),
    target_duration: Optional[float] = Form(10.0),
    db: AsyncSession = Depends(get_db)
):
    """Create a new editing job. Kicks off background execution using Tus-uploaded files."""
    job_id = project_id # Use project_id as job_id for simplicity in M0
    logger.info(f"Creating job {job_id} for project {project_id}...")
    interaction_logger.log_interaction("create_job", {
        "job_id": job_id,
        "project_id": project_id,
        "target_duration": target_duration,
    })

    # Fetch files from DB
    result = await db.execute(select(VideoFile).where(VideoFile.project_id == project_id))
    db_files = result.scalars().all()

    if not db_files:
        raise HTTPException(status_code=400, detail="No files uploaded for this project.")

    unique_paths = set()
    saved_file_paths = []
    files_info = []
    
    for f in db_files:
        if f.original_path not in unique_paths:
            unique_paths.add(f.original_path)
            saved_file_paths.append(f.original_path)
            files_info.append(VideoFileInfo(
                filename=f.filename,
                original_path=f.original_path,
                size_bytes=f.size_bytes
            ))

    # Create the initial job status in memory
    job_status = JobStatus(
        job_id=job_id,
        status="pending",
        progress=0,
        message="Job created. Ready to ingest.",
        files=files_info,
        context_text=context_text,
        target_duration=target_duration,
        created_at=datetime.utcnow(),
        warnings=[],
    )
    # Update Project in Postgres with settings and processing status
    project_result = await db.execute(select(Project).where(Project.id == project_id))
    project = project_result.scalars().first()
    if project:
        project.status = "processing"
        project.settings = {
            "context_text": context_text,
            "target_duration": target_duration,
        }
        await db.commit()

    jobs_db[job_id] = job_status

    # Start the async pipeline orchestrator
    await start_pipeline(job_id, saved_file_paths, context_text, target_duration)

    return job_status

@app.get("/api/jobs/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str):
    """Get the current job status details."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job_endpoint(job_id: str, db: AsyncSession = Depends(get_db)):
    """Cancel a running editing job."""
    job = get_job(job_id)
    if job:
        interaction_logger.log_interaction("cancel_job", {"job_id": job_id, "current_status": job.status})
        if job.status in ["complete", "failed", "cancelled"]:
            return {"status": job.status, "message": "Job cannot be cancelled in its current state."}
        cancel_job(job_id)

    # Update DB regardless of whether it's in memory (e.g., if server restarted)
    result = await db.execute(select(Project).where(Project.id == job_id))
    project = result.scalars().first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project.status = "cancelled"
    await db.commit()

    return {"status": "cancelled", "message": "Cancellation request received."}

@app.get("/api/jobs/{job_id}/download")
def download_vlog(job_id: str):
    """Download the final processed vlog MP4."""
    job = get_job(job_id)
    if job and job.status != "complete":
        raise HTTPException(status_code=400, detail="Vlog has not completed processing.")
    # Not in memory (server restarted): serve the output file if it exists (Phase 6 review after restart)

    interaction_logger.log_interaction("download_vlog", {"job_id": job_id})

    final_video_path = os.path.join(settings.output_dir, f"{job_id}.mp4")
    if not os.path.exists(final_video_path):
        raise HTTPException(status_code=404, detail="Final video output file not found on disk.")

    return FileResponse(
        path=final_video_path,
        media_type="video/mp4",
        filename=f"vlogforge_edit_{job_id[:8]}.mp4"
    )


@app.get("/api/jobs/{job_id}/raw/{filename}")
def get_raw_video(job_id: str, filename: str):
    """Serve raw uploaded video files."""
    raw_video_path = os.path.join(settings.upload_dir, job_id, "raw", filename)
    if not os.path.exists(raw_video_path):
        raise HTTPException(status_code=404, detail="Raw video file not found.")
    import mimetypes                           # voice-over uploads are audio (.m4a / .mp3 / .wav / .aac)
    return FileResponse(
        path=raw_video_path,
        media_type={".m4a": "audio/mp4", ".aac": "audio/aac", ".wav": "audio/wav", ".mp3": "audio/mpeg"}.get(
            os.path.splitext(filename)[1].lower()) or mimetypes.guess_type(filename)[0] or "video/mp4",
        filename=filename
    )

@app.get("/api/jobs/{job_id}/edit")
def get_job_edit(job_id: str):
    """Transcript review view of a word-grid job (roadmap Phase 6): words, cleanup ranges with
    reasons, current plan, output time of each kept word. From the artifact store (survives a restart)."""
    from app.tasks.recompile import load_job_edit
    view = load_job_edit(job_id)
    if view is None:
        raise HTTPException(status_code=404, detail="No stored word-grid edit for this job")
    return view

@app.get("/api/jobs/{job_id}/waveform")
def get_job_waveform(job_id: str):
    """Peak levels of the rendered output for the review timeline (cached per output file)."""
    from app.tasks.review_export import output_waveform
    wave = output_waveform(job_id)
    if wave is None:
        raise HTTPException(status_code=404, detail="Final video output file not found on disk.")
    return {"rate": wave["rate"], "peaks": wave["peaks"]}

@app.get("/api/jobs/{job_id}/export/{fmt}")
def export_job_edit(job_id: str, fmt: str, name: Optional[str] = None):
    """The last render's edit decision list for another editor: fmt 'edl' (CMX3600) or 'fcpxml'."""
    from fastapi.responses import Response
    from app.tasks.review_export import export_edl, export_fcpxml
    if fmt not in ("edl", "fcpxml"):
        raise HTTPException(status_code=400, detail="fmt must be 'edl' or 'fcpxml'")
    title = (name or f"VlogForge {job_id[:8]}").strip()[:60]
    body = (export_edl if fmt == "edl" else export_fcpxml)(job_id, title)
    if body is None:
        raise HTTPException(status_code=404, detail="No stored render for this job")
    interaction_logger.log_interaction("export_edit", {"job_id": job_id, "format": fmt})
    filename = "".join(c if c.isalnum() or c in "-_ " else "_" for c in title) + f".{fmt}"
    return Response(content=body, media_type="application/xml" if fmt == "fcpxml" else "text/plain",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})

def _caption_style(captions: Optional[str]) -> Optional[str]:
    """Query value -> caption preset key (None = no captions); 400 for an unknown preset."""
    from app.tasks.subtitles import STYLES
    if not captions or captions == "off":
        return None
    if captions not in STYLES:
        raise HTTPException(status_code=400, detail=f"Unknown caption style: {captions}")
    return captions

@app.get("/api/jobs/{job_id}/captions")
def get_job_captions(job_id: str):
    """Caption blocks of the last render (output clock, the ones burned in on export) and the style presets,
    for the editor's live caption overlay."""
    from app.tasks import subtitles
    cues = subtitles.job_cues(job_id)
    if cues is None:
        raise HTTPException(status_code=404, detail="No stored render with a word grid for this job")
    return {"block_sec": subtitles.BLOCK_SEC, "cues": cues, "styles": subtitles.STYLES,
            "portrait_margin_v": subtitles.PORTRAIT_MARGIN_V, "side_margin": subtitles.SIDE_MARGIN,
            "scale_range": subtitles.SCALE_RANGE, "placement": subtitles.job_placement(job_id)}

@app.put("/api/jobs/{job_id}/captions/placement")
def put_caption_placement(job_id: str, placement: dict):
    """Where captions sit and how big (moved / resized in the editor): {scale, x, y}; x / y null = the
    preset's place. Applies to the live overlay and to every captioned export (those made before go stale)."""
    from app.tasks import subtitles
    if subtitles.job_cues(job_id) is None:
        raise HTTPException(status_code=404, detail="No stored render with a word grid for this job")
    try:
        saved = subtitles.save_placement(job_id, placement)
    except (TypeError, ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=f"Bad placement: {e}")
    interaction_logger.log_interaction("caption_placement", {"job_id": job_id, **saved})
    return saved

@app.get("/api/jobs/{job_id}/layouts")
def get_job_layouts(job_id: str, captions: Optional[str] = None):
    """Export layouts (16:9, 9:16, 1:1, 4:5) of the last render, each with its status; with ?captions=<preset>,
    the statuses of the files with burned-in captions in that preset."""
    from app.tasks.layouts import list_layouts, source_dims
    from app.tasks.subtitles import BLOCK_SEC, STYLES
    from app.utils.ffmpeg import RENDER_SIZE
    try:
        rows = list_layouts(job_id, _caption_style(captions))
    except ValueError as e:                       # e.g. captions asked for a job without a word grid
        raise HTTPException(status_code=400, detail=str(e))
    if rows is None:
        raise HTTPException(status_code=404, detail="No stored render for this job")
    return {"layouts": rows, "main_size": list(RENDER_SIZE), "sources": source_dims(job_id),
            "caption_styles": [{"key": k, "name": v["name"], "description": v["description"]} for k, v in STYLES.items()],
            "caption_block_sec": BLOCK_SEC}

@app.post("/api/jobs/{job_id}/layouts/{key}/render")
def render_job_layout(job_id: str, key: str, captions: Optional[str] = None):
    """Render the last render's edit in another frame shape, or any shape with burned-in captions
    (?captions=<preset>) (background; poll GET .../layouts)."""
    from app.tasks.layouts import LAYOUTS, MAIN, list_layouts, start_render
    style = _caption_style(captions)
    if key not in LAYOUTS or (key == MAIN and not style):
        raise HTTPException(status_code=400, detail=f"Unknown or main layout: {key}")
    if list_layouts(job_id) is None:
        raise HTTPException(status_code=404, detail="No stored render for this job")
    started = start_render(job_id, key, style)
    interaction_logger.log_interaction("render_layout", {"job_id": job_id, "layout": key, "captions": style, "started": started})
    return {"status": "rendering", "started": started}

@app.get("/api/jobs/{job_id}/layouts/{key}/download")
def download_job_layout(job_id: str, key: str, captions: Optional[str] = None):
    """Download a rendered layout MP4 (?captions=<preset>: the captioned file)."""
    from app.tasks.layouts import LAYOUTS, layout_path
    if key not in LAYOUTS:
        raise HTTPException(status_code=400, detail=f"Unknown layout: {key}")
    style = _caption_style(captions)
    path = layout_path(job_id, key, style)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="This layout has not been rendered yet.")
    return FileResponse(path=path, media_type="video/mp4",
                        filename=f"vlogforge_{job_id[:8]}_{LAYOUTS[key]['ratio'].replace(':', 'x')}{'_captions-' + style if style else ''}.mp4")

@app.get("/api/jobs/{job_id}/publish")
def get_publish_copy(job_id: str, format: str = "auto", regenerate: bool = False):
    """Post copy for the last render: Reels/Shorts caption + hashtags (< 60 s) or YouTube title,
    description, chapters + hashtags. format: auto | short | long. Cached per render."""
    from app.tasks.publish_copy import publish_copy
    if format not in ("auto", "short", "long"):
        raise HTTPException(status_code=400, detail="format must be auto, short or long")
    try:
        copy = publish_copy(job_id, format, regenerate)
    except Exception as e:
        logger.error(f"Post copy failed for job {job_id}: {e}", exc_info=True)
        msg = ((getattr(e, "body", None) or {}).get("error") or {}).get("message") or str(e)   # API errors: their message only
        raise HTTPException(status_code=502, detail=f"Could not write the post copy: {msg}")
    if copy is None:
        raise HTTPException(status_code=404, detail="No stored render for this job")
    interaction_logger.log_interaction("publish_copy", {"job_id": job_id, "format": copy["format"], "variant": copy["variant"]})
    return copy

@app.post("/api/jobs/{job_id}/recompile")
async def recompile_job_endpoint(job_id: str, plan: EditPlan):
    """Re-compile a word-grid job with an edited plan (word ranges). Re-runs compile + render
    only: no ingest, transcription or cleanup. Works after a restart (stored artifacts)."""
    from app.tasks.recompile import can_recompile
    if not can_recompile(job_id):
        raise HTTPException(status_code=404, detail="No stored word-grid edit for this job")
    interaction_logger.log_interaction("recompile_job", {"job_id": job_id, "segments": len(plan.segments)})
    await start_recompile(job_id, plan.model_dump())
    return {"status": "recompiling", "message": "Re-compile started."}

@app.websocket("/ws/{job_id}")
async def websocket_progress_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint to push real-time status and progress updates to client."""
    await websocket.accept()

    # Send current job state immediately on connect/reconnect
    job = get_job(job_id)
    if job:
        await websocket.send_json({
            "stage": job.status,
            "progress": job.progress,
            "message": job.message,
            "download_url": job.output_video_url
        })
    else:
        # Job not in memory — server was likely restarted, state was lost
        await websocket.send_json({
            "stage": "not_found",
            "progress": 0,
            "message": "Job not found. The server may have restarted. Please start a new job.",
            "download_url": None
        })

    register_websocket(job_id, websocket)
    try:
        # Keep connection open and listen for client messages
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from job: {job_id}")
    finally:
        unregister_websocket(job_id, websocket)
