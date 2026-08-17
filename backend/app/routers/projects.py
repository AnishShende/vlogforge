from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from app.database import get_db
from app.db_models import Project, User
from app.routers.auth import get_current_user
from app.tasks.orchestrator import get_job

router = APIRouter(prefix="/api/projects", tags=["projects"])

class ProjectCreate(BaseModel):
    title: str

class FileResponse(BaseModel):
    id: str
    filename: str
    size_bytes: int
    duration: float = 0.0

    class Config:
        from_attributes = True

@router.delete("/{project_id}/files/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project_file(project_id: str, file_id: str, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    from app.db_models import VideoFile
    import os
    # Verify project belongs to user
    proj_result = await db.execute(select(Project).where(Project.id == project_id, Project.user_id == current_user.id))
    if not proj_result.scalars().first():
        raise HTTPException(status_code=404, detail="Project not found")
        
    result = await db.execute(select(VideoFile).where(VideoFile.id == file_id, VideoFile.project_id == project_id))
    file_record = result.scalars().first()
    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")
        
    # Attempt to remove from disk
    try:
        if os.path.exists(file_record.original_path):
            os.remove(file_record.original_path)
    except Exception as e:
        pass # Ignore disk deletion errors
        
    await db.delete(file_record)
    await db.commit()

class ProjectResponse(BaseModel):
    id: str
    title: str
    status: str
    settings: Optional[dict] = None
    created_at: datetime
    updated_at: datetime
    video_files: List[FileResponse] = []

    class Config:
        from_attributes = True

@router.get("/", response_model=List[ProjectResponse])
async def list_projects(db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    from sqlalchemy.orm import selectinload
    result = await db.execute(
        select(Project)
        .options(selectinload(Project.video_files))
        .where(Project.user_id == current_user.id)
        .order_by(Project.created_at.desc())
    )
    projects = result.scalars().all()
    
    # Overlay real-time execution status from memory
    for p in projects:
        job = get_job(p.id)
        if job and job.status not in ["complete", "failed", "cancelled", "pending"]:
            p.status = job.status
            
    return projects

@router.post("/", response_model=ProjectResponse)
async def create_project(project_data: ProjectCreate, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    new_project = Project(
        title=project_data.title,
        user_id=current_user.id,
        status="pending"
    )
    db.add(new_project)
    await db.commit()
    # Eager load the new project to prevent MissingGreenlet on video_files serialization
    from sqlalchemy.orm import selectinload
    result = await db.execute(
        select(Project)
        .options(selectinload(Project.video_files))
        .where(Project.id == new_project.id)
    )
    return result.scalars().first()

@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    from sqlalchemy.orm import selectinload
    result = await db.execute(
        select(Project)
        .options(selectinload(Project.video_files))
        .where(Project.id == project_id, Project.user_id == current_user.id)
    )
    project = result.scalars().first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
        
    # Overlay real-time execution status from memory
    job = get_job(project.id)
    if job and job.status not in ["complete", "failed", "cancelled", "pending"]:
        project.status = job.status
        
    return project

class ProjectUpdate(BaseModel):
    title: str

@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(project_id: str, project_update: ProjectUpdate, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    result = await db.execute(select(Project).where(Project.id == project_id, Project.user_id == current_user.id))
    project = result.scalars().first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
        
    project.title = project_update.title
    await db.commit()
    await db.refresh(project)
    return project

@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(project_id: str, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    result = await db.execute(select(Project).where(Project.id == project_id, Project.user_id == current_user.id))
    project = result.scalars().first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    await db.delete(project)
    await db.commit()
