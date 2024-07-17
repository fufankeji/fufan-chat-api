from server.db.models.knowledge_base_model import KnowledgeBaseModel
from sqlalchemy.future import select
from server.db.session import with_async_session, async_session_scope
from fastapi import HTTPException, Depends, Body
from server.db.session import get_async_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from configs import EMBEDDING_MODEL
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError
from server.db.models.knowledge_file_model import KnowledgeFileModel, FileDocModel



@with_async_session
async def add_kb_to_db(session, kb_name, kb_info, vs_type, embed_model, user_id):
    # 查询现有知识库实例
    kb = await session.execute(
        select(KnowledgeBaseModel)
        .where(KnowledgeBaseModel.kb_name.ilike(kb_name))
    )
    kb = kb.scalars().first()

    if not kb:
        # 创建新的知识库实例
        kb = KnowledgeBaseModel(kb_name=kb_name, kb_info=kb_info, vs_type=vs_type, embed_model=embed_model,
                                user_id=user_id)
        session.add(kb)
    else:
        # 更新现有知识库实例
        kb.kb_info = kb_info
        kb.vs_type = vs_type
        kb.embed_model = embed_model
        kb.user_id = user_id

    # 异步提交数据库事务
    await session.commit()
    return True


@with_async_session
async def list_kbs_from_db(session, min_file_count: int = -1):
    # 过滤条件，用于指定返回的知识库所包含的文件数量下限，默认值为 -1，意味着默认情况下会返回所有知识库的名称，不论其文件数量如何。
    result = await session.execute(
        select(KnowledgeBaseModel.kb_name)
        .where(KnowledgeBaseModel.file_count > min_file_count)
    )

    # 提取向量数据库的名称
    kbs = [kb[0] for kb in result.scalars().all()]
    return kbs


@with_async_session
async def kb_exists(session, kb_name):
    kb = session.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.kb_name.ilike(kb_name)).first()
    status = True if kb else False
    return status


@with_async_session
async def load_kb_from_db(session, kb_name):
    stmt = select(KnowledgeBaseModel).filter(KnowledgeBaseModel.kb_name.ilike(kb_name))
    result = await session.execute(stmt)
    kb = result.scalar_one_or_none()
    if kb:
        kb_name, vs_type, embed_model = kb.kb_name, kb.vs_type, kb.embed_model
    else:
        kb_name, vs_type, embed_model = None, None, None
    return kb_name, vs_type, embed_model


@with_async_session
async def delete_kb_from_db(session, kb_name):
    kb = session.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.kb_name.ilike(kb_name)).first()
    if kb:
        session.delete(kb)
    return True


@with_async_session
async def get_kb_detail(session, kb_name: str) -> dict:
    stmt = select(KnowledgeBaseModel).where(KnowledgeBaseModel.kb_name.ilike(kb_name))
    result = await session.execute(stmt)
    kb = result.scalars().first()

    if kb:
        return {
            "kb_name": kb.kb_name,
            "kb_info": kb.kb_info,
            "vs_type": kb.vs_type,
            "embed_model": kb.embed_model,
            "file_count": kb.file_count,
            "create_time": kb.create_time,
        }
    else:
        return {}


async def list_knowledge_bases(
        user_id: str,
        session: AsyncSession = Depends(get_async_db)
):
    """
    异步从数据库检索特定用户的所有知识库名称，并返回给前端。
    """
    try:
        # 在查询中加入 user_id 来筛选知识库
        result = await session.execute(
            select(KnowledgeBaseModel.kb_name).where(KnowledgeBaseModel.user_id == user_id)
        )
        kbs = result.scalars().all()
        # 使用 JSONResponse 返回筛选后的知识库名称列表
        return JSONResponse(
            status_code=200,
            content={"knowledge_bases": kbs}
        )
    except SQLAlchemyError as e:
        await session.rollback()
        raise HTTPException(status_code=500, detail=str(e))


class CreateKnowledgeBaseRequest(BaseModel):
    user_id: str
    knowledge_base_name: str
    knowledge_base_description: str = "描述信息（如果需要）"
    vector_store_type: str = "faiss"
    embed_model: str = EMBEDDING_MODEL


class DeleteKnowledgeBaseRequest(BaseModel):
    user_id: str
    knowledge_base_name: str

class KnowledgeBaseFilesRequest(BaseModel):
    user_id: str
    knowledge_base_name: str

async def create_knowledge_base(
        request: CreateKnowledgeBaseRequest,  # 使用 Pydantic 模型来接收请求数据
        session: AsyncSession = Depends(get_async_db)
):
    """
    异步创建新的知识库记录到 MySQL 数据库中。
    """
    user_id = request.user_id
    knowledge_base_name = request.knowledge_base_name
    knowledge_base_description = request.knowledge_base_description
    vector_store_type = request.vector_store_type
    embed_model = request.embed_model

    if knowledge_base_name.strip() == "":
        return JSONResponse(status_code=400, content={"msg": "知识库名称不能为空，请重新填写知识库名称"})

    existing_kb = await session.execute(
        select(KnowledgeBaseModel)
        .where(KnowledgeBaseModel.kb_name == knowledge_base_name, KnowledgeBaseModel.user_id == user_id)
    )
    if existing_kb.scalars().first() is not None:
        return JSONResponse(status_code=400, content={"msg": f"已存在同名知识库 {knowledge_base_name}"})

    new_kb = KnowledgeBaseModel(
        kb_name=knowledge_base_name,
        kb_info=knowledge_base_description,
        vs_type=vector_store_type,
        embed_model=embed_model,
        create_time=datetime.now(),
        user_id=user_id
    )

    try:
        session.add(new_kb)
        await session.commit()
        await session.refresh(new_kb)
        return JSONResponse(
            status_code=201,
            content={"id": new_kb.id, "msg": f"已新增知识库 {knowledge_base_name}"}
        )
    except SQLAlchemyError as e:
        await session.rollback()
        return JSONResponse(status_code=500, content={"msg": f"创建知识库出错： {e}"})


async def delete_knowledge_base(
        request: DeleteKnowledgeBaseRequest,
        session: AsyncSession = Depends(get_async_db)
):
    """
    异步从数据库中删除指定用户的知识库。
    """
    user_id = request.user_id
    knowledge_base_name = request.knowledge_base_name

    # 查找并删除知识库
    try:
        kb_to_delete = await session.execute(
            select(KnowledgeBaseModel)
            .where(KnowledgeBaseModel.kb_name == knowledge_base_name,
                   KnowledgeBaseModel.user_id == user_id)
        )
        kb_to_delete = kb_to_delete.scalars().first()
        if kb_to_delete is None:
            return JSONResponse(status_code=404, content={"msg": f"未找到知识库 {knowledge_base_name}"})

        await session.delete(kb_to_delete)
        await session.commit()
        return JSONResponse(
            status_code=200,
            content={"msg": f"成功删除知识库 {knowledge_base_name}"}
        )
    except SQLAlchemyError as e:
        await session.rollback()
        return JSONResponse(status_code=500, content={"msg": f"删除知识库时出现错误：{e}"})



async def list_knowledge_base_files(
        request: KnowledgeBaseFilesRequest,
        session: AsyncSession = Depends(get_async_db)
):
    """
    异步从数据库中获取指定知识库的文件列表。
    """
    user_id = request.user_id
    knowledge_base_name = request.knowledge_base_name

    # 查找知识库并获取文件列表
    try:
        kb = await session.execute(
            select(KnowledgeBaseModel)
            .where(KnowledgeBaseModel.kb_name == knowledge_base_name,
                   KnowledgeBaseModel.user_id == user_id)
        )
        kb = kb.scalars().first()
        if kb is None:
            return JSONResponse(status_code=404, content={"msg": f"未找到知识库 {knowledge_base_name}", "data": []})

        # 使用 select 构建查询
        stmt = select(KnowledgeFileModel.file_name).where(KnowledgeFileModel.kb_name.ilike(f"%{kb.kb_name}%"))
        result = await session.execute(stmt)
        all_doc_names = result.scalars().all()  # 获取所有匹配的文件名

        return JSONResponse(
            status_code=200,
            content={"data": all_doc_names}
        )
    except SQLAlchemyError as e:
        await session.rollback()
        return JSONResponse(status_code=500, content={"msg": f"获取文件列表时出现错误：{e}", "data": []})