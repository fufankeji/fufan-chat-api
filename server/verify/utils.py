from fastapi import FastAPI, HTTPException, Depends, Path
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from datetime import datetime
from server.db.session import with_async_session, get_async_db
from server.db.models.user_model import UserModel
from server.db.models.conversation_model import ConversationModel
from server.db.models.message_model import MessageModel
from server.db.models.knowledge_base_model import KnowledgeBaseModel
from server.db.models.knowledge_file_model import KnowledgeFileModel
from server.db.models.knowledge_file_model import FileDocModel
from fastapi import Body
import uuid
from sqlalchemy.dialects.postgresql import UUID
from typing import List
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload  # 正确导入 selectinload
from fastapi import Response
from fastapi.responses import JSONResponse


# 定义 Pydantic 模型，用于请求体验证
class CreateConversationRequest(BaseModel):
    user_id: str
    name: str
    chat_type: str

class ConversationResponse(BaseModel):
    id: str
    name: str
    chat_type: str
    create_time: datetime

class MessageResponse(BaseModel):
    id: str = Field(..., description="消息的唯一标识符")
    conversation_id: str = Field(..., description="关联的会话ID")
    chat_type: str = Field(..., description="聊天类型")
    query: str = Field(..., description="用户的问题")
    response: str = Field(..., description="模型的回答")
    meta_data: dict = Field(..., description="其他元数据")
    feedback_score: int = Field(..., description="用户评分")
    feedback_reason: str = Field(..., description="评分理由")
    create_time: datetime = Field(..., description="消息创建时间")

# API处理函数，用来创建新的会话
async def create_conversation(
    request: CreateConversationRequest = Body(...),
    session: AsyncSession = Depends(get_async_db)
):
    try:
        # 创建新会话对象
        new_conversation = ConversationModel(
            id=str(uuid.uuid4()),  # 生成 UUID 并转换为字符串
            user_id=request.user_id,
            name=request.name,
            chat_type=request.chat_type,
            create_time=datetime.now(),
        )

        # 添加到数据库会话
        session.add(new_conversation)
        # 提交到数据库
        await session.commit()
        # 刷新以获取数据库分配的 ID 等字段
        await session.refresh(new_conversation)

        # 返回带有创建会话 id 的 JSONResponse
        return JSONResponse(
            status_code=201,
            content={"id": new_conversation.id}
        )

    except Exception as e:
        # 处理数据库操作错误或其他异常
        await session.rollback()  # 回滚以避免部分提交
        raise HTTPException(status_code=400, detail=f"Error creating conversation: {str(e)}")



async def get_user_conversations(
    user_id: str,
    session: AsyncSession = Depends(get_async_db)
):
    async with session as async_session:
        result = await async_session.execute(
            select(ConversationModel)
            .where(ConversationModel.user_id == user_id)
        )
        conversations = result.scalars().all()
        # if not conversations:
        #     raise HTTPException(status_code=404, detail="No conversations found for this user")

        return [ConversationResponse(
            id=conv.id,
            name=conv.name,
            chat_type=conv.chat_type,
            create_time=conv.create_time
        ) for conv in conversations]



async def get_conversation_messages(
    conversation_id: str,
    session: AsyncSession = Depends(get_async_db)
):
    async with session as async_session:
        result = await async_session.execute(
            select(MessageModel)
            .where(MessageModel.conversation_id == conversation_id)
        )
        messages = result.scalars().all()
        # if not messages:
        #     raise HTTPException(status_code=404, detail="No messages found for this conversation")

        return [MessageResponse(
            id=msg.id,
            conversation_id=msg.conversation_id,
            chat_type=msg.chat_type,
            query=msg.query,
            response=msg.response,
            meta_data=msg.meta_data,
            feedback_score=msg.feedback_score,
            feedback_reason=msg.feedback_reason,
            create_time=msg.create_time
        ) for msg in messages]
