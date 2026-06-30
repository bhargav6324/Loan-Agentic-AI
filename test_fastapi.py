import asyncio
from fastapi import Request
from models.database import AsyncSessionLocal, Application
from routes.process_routes import process_application
from auth.middleware import CurrentUser

async def main():
    async with AsyncSessionLocal() as db:
        result = await db.execute(Application.__table__.select().where(Application.id == "6dec6937-4a45-43be-b702-5495ab3671fe"))
        app_row = result.first()
        if not app_row:
            print("App not found")
            return
            
        current_user = CurrentUser(user_id=app_row.user_id, role="user")
        
        try:
            res = await process_application(app_row.id, current_user, db)
            print("Success", res)
        except Exception as e:
            import traceback
            traceback.print_exc()

asyncio.run(main())
