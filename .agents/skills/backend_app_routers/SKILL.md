---
name: "vlogforge-backend-app-routers"
description: "The isolated API route definitions for the FastAPI backend. Consult this folder when modifying specific endpoints like authentication, project management, or video uploading."
---

# Module: backend/app/routers

## 📌 Purpose & Responsibility
- Houses the FastAPI route handlers (`APIRouter`) logically separated by domain.
- Currently manages authentication (`auth.py`), projects (`projects.py`), and file uploads (`upload.py`).
- Decouples endpoint logic from the main application entry point (`main.py`).

## 🔄 Integration & Data Flow
- **Inputs**: HTTP requests to `/api/auth/*`, `/api/projects/*`, `/api/upload/*`.
- **Outputs**: HTTP JSON responses and tokens.
- **Interactions**:
  - `auth.py`: Uses `OAuth2PasswordBearer` and JWTs to handle registration and login.
  - `projects.py`: Manages CRUD operations for user projects.
  - `upload.py`: Handles multipart video uploads into the file system or object storage.
  - All routers rely on `Depends(get_db)` from `database.py` for state, and `get_current_user` from `auth.py` for authorization.
  - They are included into the main FastAPI app in `main.py` via `app.include_router(router)`.

## 📂 Code Symbols & Key Files

- [auth.py](backend/app/routers/auth.py): Endpoints for `/register`, `/login`, and `/me`. Handles JWT creation and verification.
- [projects.py](backend/app/routers/projects.py): Endpoints for listing, creating, retrieving, and deleting video editing projects.
- [upload.py](backend/app/routers/upload.py): Endpoints for handling video file uploads for a specific project.
