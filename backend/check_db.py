import asyncio
from app.database import AsyncSessionLocal
from app.db_models import VideoFile, Project
from sqlalchemy.future import select

async def main():
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Project))
        projects = result.scalars().all()
        for p in projects:
            print(f"Project: {p.id} - {p.title}")
            f_result = await db.execute(select(VideoFile).where(VideoFile.project_id == p.id))
            files = f_result.scalars().all()
            for f in files:
                print(f"  File: {f.id} - {f.filename}")
            print(f"  Total files: {len(files)}\n")

if __name__ == "__main__":
    asyncio.run(main())
