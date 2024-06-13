from server.db.session import with_async_session
from fastapi import HTTPException
from server.db.models.user_model import UserModel


@with_async_session
async def check_user(session, user_id: str):
    result = await session.get(UserModel, user_id)
    if not result:
        raise HTTPException(status_code=401, detail="User ID not found")
    return {"message": "User ID exists"}