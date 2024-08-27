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
from sqlalchemy import delete
import uuid
from sqlalchemy.dialects.postgresql import UUID
from typing import List
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from fastapi import Response
from fastapi.responses import JSONResponse
from fastapi import Query
from sqlalchemy import desc


# 定义 Pydantic 模型，用于请求体验证
class CreateConversationRequest(BaseModel):
    user_id: str
    name: str = Field(default="new_chat", example="new_chat")
    chat_type: str


class ConversationResponse(BaseModel):
    id: str
    name: str
    chat_type: str
    create_time: datetime


class MessageResponse(BaseModel):
    id: str = Field(..., description="消息的唯一标识符")
    conversation_id: str = Field(..., description="关联的会话ID")
    chat_type: str = Field(..., description="对话类型（普通问答、知识库问答、AI搜索、推荐系统、Agent问答）")
    query: str = Field(..., description="用户的问题")
    response: str = Field(..., description="大模型的回答")
    meta_data: dict = Field(..., description="其他元数据")
    create_time: datetime = Field(..., description="消息创建时间")


class UpdateConversationRequest(BaseModel):
    name: str = Field(..., example="更新会话框的名称", description="新的会话名称")


async def create_conversation(
        request: CreateConversationRequest = Body(...),
        session: AsyncSession = Depends(get_async_db)
):
    """
    用来创建新的会话记录至Mysql数据库中
    """
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
            status_code=200,
            content={"status": 200, "id": new_conversation.id}
        )

    except Exception as e:
        # 处理数据库操作错误或其他异常
        await session.rollback()  # 回滚以避免部分提交
        raise HTTPException(status_code=400, detail=f"Error creating conversation: {str(e)}")


async def get_user_conversations(
        user_id: str,
        chat_types: str = Query(...),
        session: AsyncSession = Depends(get_async_db)
):
    """
    用来获取指定用户名的历史对话窗口
    """
    async with session as async_session:
        result = await async_session.execute(
            select(ConversationModel)
            .where(ConversationModel.user_id == user_id,
                   ConversationModel.chat_type == chat_types)
            .order_by(desc(ConversationModel.create_time))
        )
        conversations = result.scalars().all()

        if conversations == []:
            return {"status": 200, "msg": "success", "data": []}
        else:
            data = [ConversationResponse(
                id=conv.id,
                name=conv.name,
                chat_type=conv.chat_type,
                create_time=conv.create_time
            ) for conv in conversations]

            return {"status": 200, "msg": "success", "data": data}


async def get_conversation_messages(
        conversation_id: str,
        chat_types: List[str] = Query(None),
        session: AsyncSession = Depends(get_async_db)
):
    """
    使用 Query 参数接收可选的 chat_types 列表
    """
    async with (session as async_session):
        query = select(MessageModel).where(MessageModel.conversation_id == conversation_id)
        if chat_types:
            query = query.where(MessageModel.chat_type)

        result = await async_session.execute(query)
        messages = result.scalars().all()
        if not messages:
            return {"status": 200, "msg": "success", "data": []}
            # raise HTTPException(status_code=404,
            #                     detail="No messages found for this conversation with the specified types")

        else:
            data = [MessageResponse(
                id=msg.id,
                conversation_id=msg.conversation_id,
                chat_type=msg.chat_type,
                query=msg.query,
                response=msg.response,
                meta_data=msg.meta_data,
                create_time=msg.create_time
            ) for msg in messages]

            return {"status": 200, "msg": "success", "data": data}


async def delete_conversation_and_messages(
        conversation_id: str,
        session: AsyncSession = Depends(get_async_db)
):
    """
    删除指定的会话及其所有关联的消息记录。
    """
    async with session.begin():
        # 检查是否存在指定的会话
        conversation = await session.get(ConversationModel, conversation_id)
        if not conversation:
            # 如果会话不存在，返回404错误
            raise HTTPException(status_code=404, detail="Conversation not found")

        # 删除与会话关联的所有消息
        await session.execute(
            delete(MessageModel).where(MessageModel.conversation_id == conversation_id)
        )

        # 删除会话本身
        await session.delete(conversation)

        # 提交事务，确保所有操作都能一起完成
        await session.commit()

    # 返回204 No Content状态码，因为没有实体内容返回
    return JSONResponse(
            status_code=200,
            content={"status": 200}
        )


async def update_conversation_name(
        conversation_id: str,
        request: UpdateConversationRequest,
        session: AsyncSession = Depends(get_async_db)
):
    """
    更新会话框的显示名称
    """
    async with session.begin():
        # 查询对应的会话
        conversation = await session.get(ConversationModel, conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # 更新会话名称
        conversation.name = request.name
        session.add(conversation)
        await session.commit()

    # 返回200 OK状态码和确认消息
    return JSONResponse(status_code=200, content={"status": 200, "message": "Conversation name updated successfully"})
