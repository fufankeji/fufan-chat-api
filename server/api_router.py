#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from fastapi.staticfiles import StaticFiles
from typing import List

from typing import Union
from fastapi import FastAPI
from server.chat.chat import chat
import uvicorn
import argparse
from fastapi.middleware.cors import CORSMiddleware


def create_app(run_mode: str = None):
    app = FastAPI(
        title="fufan-chat-api API Server",
    )
    # Add CORS middleware to allow all origins
    # 在config.py中设置OPEN_DOMAIN=True，允许跨域
    # set OPEN_DOMAIN=True in config.py to allow cross-domain

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 挂载路由
    mount_app_routes(app)

    # 挂载 Vue 构建的前端静态文件夹
    app.mount("/", StaticFiles(directory="static/dist"), name="static")
    return app


from server.chat.knowledge_base_chat import knowledge_base_chat
from server.chat.search_engine_chat import search_engine_chat
from server.chat.recommendation_chat import recommend_base_chat
from server.chat.agent_chat import agent_chat
from server.utils import BaseResponse


def mount_app_routes(app: FastAPI):
    """
    这里定义通用领域问答对话的接口
    """

    # 大模型对话接口
    app.post("/api/chat",
             tags=["Chat"],
             summary="大模型对话交互接口",
             )(chat)

    # # 通用知识库问答接口
    app.post("/api/chat/knowledge_base_chat",
             tags=["Chat"],
             summary="与知识库对话")(knowledge_base_chat)

    app.post("/api/chat/search_engine_chat",
             tags=["Chat"],
             summary="与搜索引擎对话",
             )(search_engine_chat)

    app.post("/api/chat/recommend_chat",
             tags=["Chat"],
             summary="推荐",
             )(recommend_base_chat)

    app.post("/api/chat/agent_chat",
             tags=["Chat"],
             summary="Agent对话能力",
             )(agent_chat)

    # 用户管理模块相关接口
    from server.db.repository.user_repository import (
        register_user, login_user
    )

    # 用户注册
    app.post("/api/users/register",
             tags=["Users"],
             summary="用户注册",
             )(register_user)

    # 用户登录
    app.post("/api/users/login",
             tags=["Users"],
             summary="用户登录",
             )(login_user)

    # 会话管理模块相关接口
    from server.db.repository.conversation_repository import (
        create_conversation, get_user_conversations, get_conversation_messages,
        ConversationResponse, MessageResponse, delete_conversation_and_messages,
        update_conversation_name
    )

    # 新建会话接口
    app.post("/api/conversations",
             tags=["Conversations"],
             summary="新建会话接口",
             )(create_conversation)

    # 获取用户会话列表接口
    app.get("/api/users/{user_id}/conversations",  # 确保路径正确表示用户ID的参数化
            # response_model=List[ConversationResponse],  # 使用正确的响应模型
            tags=["Conversations"],
            summary="获取指定用户的会话列表",
            )(get_user_conversations)

    # 删除用户会话列表及对应的消息接口
    app.delete("/api/conversations/{conversation_id}",
               tags=["Conversations"],
               summary="删除指定会话及其所有消息",
               status_code=204,  # 明确设置状态码为204
               )(delete_conversation_and_messages)

    # 更新会话名称接口
    app.put("/api/conversations/{conversation_id}/update_name",
            tags=["Conversations"],
            summary="更新指定会话的名称",
            status_code=200,  # 明确指定成功时返回的状态码
            )(update_conversation_name)

    # 获取会话消息列表接口
    app.get("/api/conversations/{conversation_id}/messages",
            # response_model=List[MessageResponse],  # 使用正确的响应模型
            tags=["Messages"],
            summary="获取指定会话的消息列表",
            )(get_conversation_messages)

    from server.db.repository.utils import list_running_models

    # 获取加载的模型名称接口
    app.get("/api/llm_model/list_running_models",
            tags=["Models"],
            summary="列出当前已加载的模型",
            # response_model=BaseResponse,
            )(list_running_models)

    from server.db.repository.knowledge_base_repository import (
        list_knowledge_bases, create_knowledge_base, CreateKnowledgeBaseRequest,
        delete_knowledge_base, DeleteKnowledgeBaseRequest,
        list_knowledge_base_files, KnowledgeBaseFilesRequest
    )

    # 获取指定用户可以访问的知识库列表
    app.get("/api/knowledge-bases/{user_id}",
            tags=["Knowledge Management"],
            summary="获取知识库列表")(list_knowledge_bases)

    # 新建一个知识库
    app.post("/api/knowledge_base/create_knowledge_base",
             tags=["Knowledge Management"],
             # response_model=List[CreateKnowledgeBaseRequest],
             summary="创建知识库"
             )(create_knowledge_base)

    # 删除数据库
    app.delete("/api/knowledge_base/delete_knowledge_base",
             tags=["Knowledge Management"],
             # response_model=List[DeleteKnowledgeBaseRequest],
             summary="删除知识库"
             )(delete_knowledge_base)

    # 获取知识库内文件列表
    app.get("/api/knowledge_base/list_files",
            tags=["Knowledge Management"],
            # response_model=List[KnowledgeBaseFilesRequest],
            summary="获取知识库内的文件列表"
            )(list_knowledge_base_files)


def run_api(host, port, **kwargs):
    if kwargs.get("ssl_keyfile") and kwargs.get("ssl_certfile"):
        uvicorn.run(app,
                    host=host,
                    port=port,
                    ssl_keyfile=kwargs.get("ssl_keyfile"),
                    ssl_certfile=kwargs.get("ssl_certfile"),
                    )
    else:
        uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="192.168.110.131")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl_keyfile", type=str)
    parser.add_argument("--ssl_certfile", type=str)
    # 初始化消息
    args = parser.parse_args()
    args_dict = vars(args)

    app = create_app()

    run_api(host=args.host,
            port=args.port,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            )
