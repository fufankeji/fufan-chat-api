from document_loaders.pdfloader import UnstructuredLightPipeline
import operator
from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from server.db.session import with_async_session
from sqlalchemy.future import select
from server.db.repository.user_repository import (
    register_user, UserRegistrationRequest
)
from server.knowledge_base.kb_service.faiss_kb_service import FaissKBService
from server.knowledge_base.utils import (
    get_kb_path, get_doc_path, KnowledgeFile,
    list_kbs_from_folder, list_files_from_folder,
)
from langchain.docstore.document import Document
import asyncio
from server.db.models.user_model import UserModel
from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.future import select
import bcrypt
import uuid


@with_async_session
async def get_user_by_username(session, username: str, ):
    """
    根据用户名获取用户
    """
    user = await session.execute(select(UserModel).filter_by(username=username))
    return user.scalars().first()


@with_async_session
async def register_user(session, username: str, password: str):
    """
    根据用户名注册新用户
    """
    # 密码加密
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

    # 创建新用户
    new_user = UserModel(
        id=str(uuid.uuid4()),
        username=username,
        password_hash=hashed_password.decode()  # 确保存储的是字符串
    )
    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)
    return {
        "id": new_user.id,
        "username": new_user.username
    }


async def process_and_add_document(file_path, faiss_service, user_id):
    from server.knowledge_base.kb_service.base import KBServiceFactory
    kb = await KBServiceFactory.get_service_by_name("private")

    # 如果想要使用的向量数据库的collecting name 不存在，则进行创建
    if kb is None:
        from server.db.repository.knowledge_base_repository import add_kb_to_db

        # 先在Mysql中创建向量数据库的基本信息
        await add_kb_to_db(kb_name="private",
                           kb_info="个人/公司私有知识库数据",
                           vs_type="faiss",
                           embed_model="bge-large-zh-v1.5",
                           user_id=user_id)

    processor = UnstructuredLightPipeline()
    docs = await processor.run_pipeline(file_path, ['unstructured'])

    # 创建 KnowledgeFile 对象
    kb_file = KnowledgeFile(Path(file_path).name, "private")

    # 添加文档到 FAISS 服务
    added_docs_info = await faiss_service.add_doc(kb_file, docs=docs)
    print(f"Added documents for {file_path}: {added_docs_info}")


async def private_main(user_id):
    # 文件夹路径，包含所有PDF文件
    folder_path = '/home/00_rag/fufan-chat-api/knowledge_base/private/content'
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]

    # 实例化 FaissKBService
    faiss_service = FaissKBService("private")

    # 处理每一个PDF文件
    for pdf_file in pdf_files:
        full_path = os.path.join(folder_path, pdf_file)
        await process_and_add_document(full_path, faiss_service, user_id)


async def wiki_main(user_id):
    from langchain.schema import Document
    file_path = "/home/00_rag/fufan-chat-api/knowledge_base/wiki/content/education.jsonl"
    # 创建一个空的 Document 列表
    docs = []
    # 打开文件并读取每一行
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    # 解析 JSON 数据
                    data = json.loads(line)
                    # 创建一个 Document 对象
                    document = Document(page_content=data['contents'],
                                        metadata={'source': file_path})
                    # 将 Document 对象添加到列表中
                    docs.append(document)
                except json.JSONDecodeError as e:
                    # 如果 JSON 数据有问题，打印错误信息并跳过
                    print(f"Error decoding JSON: {e}")
                except KeyError as e:
                    # 如果缺少预期的键
                    print(f"Missing key in JSON data: {e}")
    except Exception as e:
        print(f"Failed to read file: {e}")

    # 实例化 FaissKBService
    faiss_service = FaissKBService("wiki")

    from server.knowledge_base.kb_service.base import KBServiceFactory
    kb = await KBServiceFactory.get_service_by_name("wiki")

    # 如果想要使用的向量数据库的collecting name 不存在，则进行创建
    if kb is None:
        from server.db.repository.knowledge_base_repository import add_kb_to_db

        # 先在Mysql中创建向量数据库的基本信息
        await add_kb_to_db(kb_name="wiki",
                           kb_info="wiki公共数据信息",
                           vs_type="faiss",
                           embed_model="bge-large-zh-v1.5",
                           user_id=user_id)

    # print(faiss_service)
    # 创建 KnowledgeFile 对象，注意这里只传递文件名和知识库名称
    kb_file = KnowledgeFile("education.jsonl", "wiki")

    # 添加文档到 FAISS 服务
    added_docs_info = await faiss_service.add_doc(kb_file, docs=docs)

    print("Added documents:", added_docs_info)


async def sequential_execution():
    # 检查用户是否存在
    admin_user = await get_user_by_username(username="admin")

    if admin_user is None:
        # 用户不存在，创建用户
        admin_user = await register_user(username="admin", password="admin")
        user_id = admin_user["id"]
    else:
        user_id = admin_user.id
    await private_main(user_id)
    await wiki_main(user_id)


async def test_query():
    faissService = FaissKBService("private")
    search_ans = await faissService.search_docs(query="解释一下langchain")
    print(search_ans)


if __name__ == '__main__':
    asyncio.run(sequential_execution())
    # 测试
    asyncio.run(test_query())
