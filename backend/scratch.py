import asyncio
from app.database import AsyncSessionLocal
from app.db_models import VideoFile
from sqlalchemy.future import select

async def main():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(VideoFile).where(VideoFile.project_id == "4d88eb9f-296c-426c-8866-f257552a0fa2"))
        files = result.scalars().all()
        for f in files:
            print(f.id, f.filename, f.original_path)

asyncio.run(main())
