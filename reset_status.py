import asyncio
from models.database import AsyncSessionLocal, Application
from sqlalchemy import update

async def main():
    async with AsyncSessionLocal() as db:
        await db.execute(update(Application).where(Application.id == "6dec6937-4a45-43be-b702-5495ab3671fe").values(status="submitted"))
        await db.commit()

asyncio.run(main())
