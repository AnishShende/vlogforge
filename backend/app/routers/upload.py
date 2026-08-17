import os
import uuid
import base64
from fastapi import APIRouter, Request, Response, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.database import get_db
from app.db_models import VideoFile, Project, User
from app.routers.auth import get_current_user
from app.config import settings

router = APIRouter(prefix="/api/upload", tags=["upload"])

# Base directory for tus uploads
TUS_DIR = os.path.join(settings.upload_dir, "tus_temp")
os.makedirs(TUS_DIR, exist_ok=True)

@router.options("/tus")
async def tus_options():
    return Response(
        status_code=status.HTTP_204_NO_CONTENT,
        headers={
            "Tus-Resumable": "1.0.0",
            "Tus-Version": "1.0.0",
            "Tus-Extension": "creation,termination",
            "Tus-Max-Size": "53687091200",
            "Access-Control-Expose-Headers": "Tus-Resumable, Tus-Version, Tus-Extension, Tus-Max-Size"
        }
    )

@router.post("/tus")
async def tus_create(request: Request, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    # Read headers
    upload_length = request.headers.get("Upload-Length")
    if not upload_length:
        raise HTTPException(status_code=400, detail="Missing Upload-Length header")
    
    upload_metadata = request.headers.get("Upload-Metadata")
    metadata = {}
    if upload_metadata:
        for kv in upload_metadata.split(","):
            parts = kv.strip().split(" ")
            if len(parts) == 2:
                key, val = parts
                metadata[key] = base64.b64decode(val).decode("utf-8")
    
    project_id = metadata.get("project_id")
    filename = metadata.get("filename", "unknown.mp4")
    
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id is required in Upload-Metadata")
    
    # Verify project belongs to user
    res = await db.execute(select(Project).where(Project.id == project_id, Project.user_id == current_user.id))
    if not res.scalars().first():
        raise HTTPException(status_code=404, detail="Project not found")

    file_id = str(uuid.uuid4())
    file_path = os.path.join(TUS_DIR, file_id)
    
    # Create empty file
    with open(file_path, "wb") as f:
        pass
        
    # Save info file to store metadata/length
    with open(f"{file_path}.info", "w") as f:
        f.write(f"{upload_length}\n{filename}\n{project_id}")

    # Use absolute URL for Location header to satisfy tus-js-client perfectly
    # or a relative URL is fine, but we must pass headers in the Response constructor
    return Response(
        status_code=status.HTTP_201_CREATED,
        headers={
            "Tus-Resumable": "1.0.0",
            "Location": f"/api/upload/tus/{file_id}",
            "Access-Control-Expose-Headers": "Location, Tus-Resumable"
        }
    )

@router.head("/tus/{file_id}")
async def tus_head(file_id: str):
    file_path = os.path.join(TUS_DIR, file_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    info_path = f"{file_path}.info"
    if not os.path.exists(info_path):
        raise HTTPException(status_code=404, detail="File info not found")
        
    with open(info_path, "r") as f:
        lines = f.readlines()
        upload_length = lines[0].strip()

    offset = os.path.getsize(file_path)
    
    return Response(
        status_code=status.HTTP_200_OK,
        headers={
            "Tus-Resumable": "1.0.0",
            "Upload-Offset": str(offset),
            "Upload-Length": upload_length,
            "Cache-Control": "no-store",
            "Access-Control-Expose-Headers": "Upload-Offset, Upload-Length, Tus-Resumable"
        }
    )

@router.patch("/tus/{file_id}")
async def tus_patch(file_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    upload_offset = request.headers.get("Upload-Offset")
    if upload_offset is None:
        raise HTTPException(status_code=400, detail="Missing Upload-Offset")
        
    upload_offset = int(upload_offset)
    
    file_path = os.path.join(TUS_DIR, file_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    current_offset = os.path.getsize(file_path)
    if current_offset != upload_offset:
        raise HTTPException(status_code=409, detail=f"Upload-Offset mismatch. Expected {current_offset}")

    content_type = request.headers.get("Content-Type")
    if content_type != "application/offset+octet-stream":
        raise HTTPException(status_code=415, detail="Invalid Content-Type")
        
    with open(file_path, "ab") as f:
        async for chunk in request.stream():
            f.write(chunk)
            
    new_offset = os.path.getsize(file_path)
    
    # Check if complete
    info_path = f"{file_path}.info"
    with open(info_path, "r") as f:
        lines = f.readlines()
        upload_length = int(lines[0].strip())
        filename = lines[1].strip()
        project_id = lines[2].strip()
        
    if new_offset == upload_length:
        # Upload complete, move to project dir and create DB entry
        project_dir = os.path.join(settings.upload_dir, project_id, "raw")
        os.makedirs(project_dir, exist_ok=True)
        final_path = os.path.join(project_dir, filename)
        os.rename(file_path, final_path)
        os.remove(info_path)
        
        # Save to DB (Upsert to prevent duplicates if user re-uploads the same file)
        existing_res = await db.execute(select(VideoFile).where(
            VideoFile.project_id == project_id, 
            VideoFile.filename == filename
        ))
        existing_video = existing_res.scalars().first()
        
        if existing_video:
            existing_video.size_bytes = new_offset
            existing_video.upload_status = "complete"
            existing_video.original_path = final_path
        else:
            new_video = VideoFile(
                project_id=project_id,
                filename=filename,
                original_path=final_path,
                size_bytes=new_offset,
                upload_status="complete"
            )
            db.add(new_video)
            
        await db.commit()

    return Response(
        status_code=status.HTTP_204_NO_CONTENT,
        headers={
            "Tus-Resumable": "1.0.0",
            "Upload-Offset": str(new_offset),
            "Access-Control-Expose-Headers": "Upload-Offset, Tus-Resumable"
        }
    )
