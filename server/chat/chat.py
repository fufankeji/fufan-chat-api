#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from fastapi import Body, HTTPException
from typing import List, Union, Optional
from sse_starlette.sse import EventSourceResponse
from configs.model_config import LLM_MODELS, TEMPERATURE
from server.utils import wrap_done, get_ChatOpenAI
from server.chat.utils import History
from typing import AsyncIterable
# from server.callback_handler.conversation_callback_handler import ConversationCallbackHandler
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
from server.verify.check_user import check_user

async def chat(query: str = Body(..., description="用户输入", examples=["你好"]),
               user_id: str = Body("", description="用户ID"),
               conversation_id: str = Body("", description="对话框ID"),
               conversation_name: str = Body("", description="对话框名称"),
               history_len: int = Body(-1, description="从数据库中取历史消息的数量"),
               history: Union[int, List[History]] = Body([],
                                                         description="历史对话，设为一个整数可以从数据库中读取历史消息",
                                                         examples=[[
                                                             {"role": "user",
                                                              "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                             {"role": "assistant", "content": "虎头虎脑"}]]
                                                         ),
               stream: bool = Body(False, description="流式输出"),
               model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
               temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=2.0),
               max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
               prompt_name: str = Body("default", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
               ):
    """
    :param query: 在对话框输入的问题
    :param user_id: 用户的id（经过登录校验的）
    :param conversation_id: 对话框的id
    :param conversation_name: 对话框的名称
    :param history_len: 如果是1，拿前端传过来的历史对话，如果是-1，从数据库中查询对话历史
    :param history: 前端传来的当前会话的ID
    :param stream: 是否流式输出
    :param model_name: 大模型的名称
    :param temperature: 大模型的采样温度
    :param max_tokens: 大模型的最大输入限制
    :param prompt_name: 提示模板
    :return:
    """

    # 官方Docs：https://github.com/sysid/sse-starlette
    #  Server-Sent Events (SSE) 相关联。SSE 是一种服务器推技术，允许服务器通过HTTP连接向浏览器或其他客户端推送信息。
    #  与WebSocket相比，SSE专门设计用于一对多的通信，其中服务器向多个客户端发送更新。
    async def chat_iterator() -> AsyncIterable[str]:
        nonlocal history, max_tokens
        callback = AsyncIteratorCallbackHandler()
        callbacks = [callback]
        memory = None

        # 进行用户校验
        await check_user(user_id)

        # 构造一个新的Message_ID记录
        message_id = await add_message_to_db(user_id=user_id,
                                             conversation_id=conversation_id,
                                             conversation_name=conversation_name,
                                             prompt_name=prompt_name,
                                             query=query
                                             )

        conversation_callback = ConversationCallbackHandler(conversation_id=conversation_id,
                                                            message_id=message_id,
                                                            chat_type=prompt_name,
                                                            query=query)
        callbacks.append(conversation_callback)

        # 这段检查和处理代码的意义在于确保 max_tokens 的值在有效范围内。
        # 如果 max_tokens 是一个不合理的值（即小于或等于0），代码将其设置为 None，以防止后续操作中的错误或异常。
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None


        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
        )


        if history:  # 优先使用前端传入的历史消息
            pass
        elif conversation_id and history_len > 0:  # 前端要求从数据库取历史消息
            # 使用memory 时必须 prompt 必须含有memory.memory_key 对应的变量
            prompt = get_prompt_template("llm_chat", "with_history")
            chat_prompt = PromptTemplate.from_template(prompt)
            # 根据conversation_id 获取message 列表进而拼凑 memory
            memory = ConversationBufferDBMemory(conversation_id=conversation_id,
                                                llm=model,
                                                message_limit=history_len)

        else:
            pass



        chain = LLMChain(prompt=chat_prompt, llm=model, memory=memory)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"input": query}),
            callback.done),
        )

        if stream:
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


if __name__ == '__main__':
    chat(query="你好，请你详细的向我介绍一下什么是机器学习？", conversation_id="", model_name="chatglm3-6b",
         max_tokens=1024, temperature=0.8)

    # import openai
    # 
    # openai.api_key = "EMPTY"
    # openai.base_url = "http://192.168.110.131:20000/v1/"
    # 
    # model = "zhipu-api"
    # # create a chat completion
    # completion = openai.chat.completions.create(
    #     model=model,
    #     messages=[{"role": "user", "content": "Hello! What is your name?"}]
    # )
    # # print the completion
    # print(completion.choices[0].message.content)
