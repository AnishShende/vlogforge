---
name: "vlogforge-backend-app-db"
description: "The database configuration and schema for VlogForge backend. Consult this folder when modifying SQLAlchemy models, Alembic migrations, or database connection logic."
---

# Module: backend/app/database

## 📌 Purpose & Responsibility
- Houses the configuration for the Postgres database connection and SQLAlchemy Object Relational Mapper (ORM).
- Handles Alembic database schema migrations.
- Defines the ORM models (e.g., `User`, `Project`) that map to the Postgres database tables.

## 🔄 Integration & Data Flow
- **Inputs**: Database connection string from `DATABASE_URL` or `.env`.
- **Outputs**: Async database sessions (`AsyncSession`) yielded by the `get_db` dependency.
- **Interactions**:
  - `database.py` sets up the `AsyncSessionLocal` using `create_async_engine`.
  - FastAPI routers (`auth.py`, `projects.py`) use `Depends(get_db)` to inject a database session for querying or mutating data.
  - `alembic/` directory contains migration scripts that manage the schema state across deployments.

## 📂 Code Symbols & Key Files

- [database.py](backend/app/database.py): Defines the `get_db` async generator dependency and sets up `create_async_engine` and `declarative_base`.
- [db_models.py](backend/app/db_models.py): Defines the SQLAlchemy models. Inherits from `Base`.
- [alembic.ini](backend/alembic.ini): Alembic configuration file at the `backend/` root. Contains the database URL and points to the `alembic` folder.
- [alembic/env.py](backend/alembic/env.py): Environment setup for Alembic migrations, importing `Base` from `db_models.py` for autogeneration.

## 🛠️ Migration Workflow & Anti-patterns
1. Edit/add the model in `db_models.py` (inherits from `Base`; UUID string PKs: `id = Column(String, primary_key=True, default=generate_uuid)`).
2. With Postgres running (`docker-compose up -d`), from `backend/`: `alembic revision --autogenerate -m "describe_change"`.
3. **Review** the generated script in `alembic/versions/` — autogenerate misses table renames and some type changes.
4. Apply with `alembic upgrade head` (roll back one with `alembic downgrade -1`).

- Do not hand-run `CREATE TABLE`/`ALTER` in Postgres — always go through Alembic.
- Do not edit an already-applied migration; add a new one instead.
