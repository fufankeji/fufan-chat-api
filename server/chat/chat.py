#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from fastapi import Body, HTTPException
from typing import List, Union, Optional
from sse_starlette.sse import EventSourceResponse
from configs.model_config import LLM_MODELS, TEMPERATURE, MAX_TOKENS, STREAM
from server.utils import wrap_done, get_ChatOpenAI
from server.chat.utils import History
from typing import AsyncIterable
import json
from langchain.callbacks import AsyncIteratorCallbackHandler
from server.utils import get_prompt_template
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
import uuid
from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from server.db.session import get_async_db
from server.db.repository.message_repository import add_message_to_db
from server.memory.conversation_db_buffer_memory import ConversationBufferDBMemory
from server.callback_handler.conversation_callback_handler import ConversationCallbackHandler
from langchain.prompts import PromptTemplate


async def chat(query: str = Body(..., description="用户输入", examples=["你好"]),
               conversation_id: str = Body("", description="对话框ID"),
               model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
               prompt_name: str = Body("general_chat",
                                       description="使用的prompt模板名称(在configs/prompt_config.py中配置)")
               ):
    """
    :param query: 在对话框输入的问题
    :param user_id: 用户的id（经过登录校验的）
    :param conversation_id: 对话框的id
    :param model_name: 使用的大模型的名称
    :return:
    """

    #  官方Docs：https://github.com/sysid/sse-starlette
    #  Server-Sent Events (SSE) 相关联。SSE 是一种服务器推技术，允许服务器通过HTTP连接向浏览器或其他客户端推送信息。
    #  与WebSocket相比，SSE专门设计用于一对多的通信，其中服务器向多个客户端发送更新。
    async def chat_iterator() -> AsyncIterable[str]:

        callback = AsyncIteratorCallbackHandler()
        callbacks = [callback]
        memory = None

        # 构造一个新的Message_ID记录
        message_id = await add_message_to_db(query=query,
                                             conversation_id=conversation_id,
                                             prompt_name=prompt_name
                                             )

        conversation_callback = ConversationCallbackHandler(query=query,
                                                            conversation_id=conversation_id,
                                                            message_id=message_id,
                                                            chat_type=prompt_name,
                                                            )
        callbacks.append(conversation_callback)

        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            callbacks=callbacks,
        )

        if conversation_id:
            # 使用memory 时必须 prompt 必须含有memory.memory_key 对应的变量
            prompt = get_prompt_template(prompt_name, "chat_with_history")
            chat_prompt = PromptTemplate.from_template(prompt)
            # 根据conversation_id 获取message 列表进而拼凑 memory
            memory = ConversationBufferDBMemory(conversation_id=conversation_id,
                                                llm=model,
                                                chat_type=prompt_name,
                                                message_limit=10)

        chain = LLMChain(prompt=chat_prompt, llm=model, memory=memory)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"input": query}),
            callback.done),
        )

        if STREAM:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps(
                    {"text": token, "message_id": message_id},
                    ensure_ascii=False)
        else:

            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps(
                {"text": answer, "message_id": message_id},
                ensure_ascii=False)

        await task

    return EventSourceResponse(chat_iterator())
