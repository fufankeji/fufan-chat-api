from pydantic import BaseModel, Field
from fastapi import HTTPException, Depends, Body
from sqlalchemy.exc import IntegrityError
from server.db.session import get_async_db
from server.db.models.user_model import UserModel
from passlib.hash import bcrypt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
import uuid
from fastapi import Response
from fastapi.responses import JSONResponse
from typing import List
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from server.db.session import with_async_session
from fastapi import HTTPException
from server.db.models.user_model import UserModel


class UserRegistrationRequest(BaseModel):
    username: str = Field(..., example="user123")
    password: str = Field(..., example="securepassword123")


class UserLoginRequest(BaseModel):
    username: str = Field(..., example="user123")
    password: str = Field(..., example="password123")


async def register_user(
        request: UserRegistrationRequest = Body(...),
        session: AsyncSession = Depends(get_async_db)
):
    """
    用户注册逻辑
    """
    print(f"request: {request}")
    hashed_password = bcrypt.hash(request.password)
    new_user = UserModel(
        id=str(uuid.uuid4()),
        username=request.username,
        password_hash=hashed_password
    )
    try:
        session.add(new_user)
        await session.commit()
        await session.refresh(new_user)

        return JSONResponse(
            status_code=201,
            content={"status": 200, "id": new_user.id, "username": new_user.username}
        )
    except IntegrityError:
        await session.rollback()
        raise HTTPException(status_code=400, detail="Username is already taken")


async def login_user(
        request: UserLoginRequest = Body(...),
        session: AsyncSession = Depends(get_async_db)
):
    # 使用 username 来查询用户
    user = await session.execute(select(UserModel).where(UserModel.username == request.username))
    user = user.scalar_one_or_none()

    if user and bcrypt.verify(request.password, user.password_hash):

        return JSONResponse(
            status_code=200,
            content={
                "status": 200,
                "id": user.id,
                "username": user.username,
                "message": "Login successful"
            }
        )
    else:
        return {"status": 401, "message": "用户名或密码错误。"}
        # raise HTTPException(status_code=401, detail="Invalid username or password")


@with_async_session
async def check_user(session, user_id: str):
    result = await session.get(UserModel, user_id)
    if not result:
        raise HTTPException(status_code=401, detail="User ID not found")
    return {"message": "User ID exists"}
