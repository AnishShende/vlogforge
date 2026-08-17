from sqlalchemy import Column, String, Integer, BigInteger, Float, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.database import Base

def generate_uuid():
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    projects = relationship("Project", back_populates="owner", cascade="all, delete-orphan")
    connected_accounts = relationship("ConnectedAccount", back_populates="user", cascade="all, delete-orphan")

class Project(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=False, default="Untitled Project")
    status = Column(String, default="pending")  # pending, uploading, processing, complete, failed, cancelled
    settings = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    owner = relationship("User", back_populates="projects")
    video_files = relationship("VideoFile", back_populates="project", cascade="all, delete-orphan")

class VideoFile(Base):
    __tablename__ = "video_files"

    id = Column(String, primary_key=True, default=generate_uuid)
    project_id = Column(String, ForeignKey("projects.id"), nullable=False)
    filename = Column(String, nullable=False)
    original_path = Column(String, nullable=False)
    duration = Column(Float, default=0.0)
    size_bytes = Column(BigInteger, default=0)
    upload_status = Column(String, default="pending")  # pending, uploading, complete
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    project = relationship("Project", back_populates="video_files")

class ConnectedAccount(Base):
    __tablename__ = "connected_accounts"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    platform = Column(String, nullable=False)  # e.g., "youtube", "instagram"
    access_token = Column(String, nullable=False)
    refresh_token = Column(String, nullable=True)
    platform_account_id = Column(String, nullable=True)
    platform_account_name = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="connected_accounts")
