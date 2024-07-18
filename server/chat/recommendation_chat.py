#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from fastapi import Body, Request
from sse_starlette.sse import EventSourceResponse
from fastapi.concurrency import run_in_threadpool
from configs import (LLM_MODELS,
                     USE_RERANKER,
                     RERANKER_MODEL,
                     RERANKER_MAX_LENGTH,
                     VECTOR_SEARCH_TOP_K,
                     SCORE_THRESHOLD,
                     TEMPERATURE, MAX_TOKENS, STREAM, )
from server.utils import wrap_done, get_ChatOpenAI, get_model_path
from server.utils import BaseResponse, get_prompt_template
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional
from server.chat.utils import History
import asyncio, json
from langchain.prompts.chat import ChatPromptTemplate
from server.chat.utils import History
from server.knowledge_base.kb_service.base import KBServiceFactory
from urllib.parse import urlencode
from server.knowledge_base.kb_doc_api import search_docs
from server.reranker.reranker import LangchainReranker
from server.utils import embedding_device
from server.db.repository.message_repository import add_message_to_db
from langchain.callbacks import AsyncIteratorCallbackHandler
from server.memory.conversation_db_buffer_memory import ConversationBufferDBMemory
from server.callback_handler.conversation_callback_handler import ConversationCallbackHandler
from langchain.prompts import PromptTemplate
from server.knowledge_base.kb_doc_api import search_docs
from server.reranker.reranker import LangchainReranker
from server.chat.utils import generate_user_profile_and_extract_info
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.knowledge_base.kb_service.milvus_kb_service import MilvusKBService
from server.chat.utils import get_conversation_history


async def recommend_base_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                              conversation_id: str = Body("", description="对话框ID"),
                              knowledge_base_name: str = Body("recommend_system", description="知识库名称",
                                                              examples=["samples"]),
                              model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                              prompt_name: str = Body(
                                  "recommend_base_chat",
                                  description="使用的prompt模板名称(在configs/prompt_config.py中配置)"
                              ),
                              request: Request = None,
                              ):
    async def recommend_base_chat_iterator(
            query: str,
            model_name: str = model_name,
            prompt_name: str = prompt_name,
    ) -> AsyncIterable[str]:
        callback = AsyncIteratorCallbackHandler()

        callbacks = [callback]

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

        # 在同一个 model 实例上同时运行两个异步链（LLMChain）可能导致内部状态的混乱,所以为用户画像生成和聊天响应分别实例化模型
        # 该模型实例用于生成用户画像
        model_for_profile = get_ChatOpenAI(
            model_name=model_name,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        # 该模型实例用于生成聊天响应
        model_for_chat = get_ChatOpenAI(
            model_name=model_name,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            callbacks=callbacks,
        )

        # 获取当前conversation中历史的对话记录
        chat_history = await get_conversation_history(conversation_id=conversation_id, limit=5, prompt_name=prompt_name)

        # 至少超过5轮对话，才会有概率生成推荐
        if len(chat_history) >= 5:
            # 使用LLM对历史对话进行分析，生成用户画像
            from configs.prompt_config import user_profile_prompt

            try:
                user_profile_ans = await generate_user_profile_and_extract_info(chat_history, user_profile_prompt,
                                                                                model_for_profile)
            except Exception as e:
                print(f"Error during profile generation: {e}")

            # 这里如果量非常大，可以采用es做倒排索引
            # 提取向量数据库的实例,统一进入同一个默认向量库中检索
            milvusService = MilvusKBService(knowledge_base_name)
            recommend_retriever = await milvusService.search_docs(str(user_profile_ans), top_k=3)

            recommend_documents = []

            valid_recommendation_index = 0  # 用于记录有效推荐的序号

            for doc_tuple in recommend_retriever:
                doc, _ = doc_tuple  # 每个元组的结构是 (Document对象, 相似度得分)
                # 解析Document对象中的page_content JSON字符串
                if hasattr(doc, 'page_content') and isinstance(doc.page_content, str):
                    try:
                        content = json.loads(doc.page_content)  # 将JSON字符串转换为字典
                        course = content.get('Course', '未知课程')  # 获取课程名称，如果不存在则返回'未知课程'
                        module_name = content.get('ModuleName', '未知模块')  # 获取模块名称，如果不存在则返回'未知模块'

                        # 仅在成功解析JSON后递增有效推荐的序号
                        valid_recommendation_index += 1
                        # 构建推荐文本
                        text = f"推荐学习 [{valid_recommendation_index}] 【{course}】课程中：【{module_name}】"
                        recommend_documents.append(text)
                    except json.JSONDecodeError:
                        # 如果解析JSON失败，打印错误信息并跳过此条目
                        print(f"Error parsing JSON for document with content: {doc.page_content}")

        # 正常的模型推理流程
        prompt_template = get_prompt_template(prompt_name, "chat_with_recommend")

        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages([input_msg])
        # 根据conversation_id 获取message 列表进而拼凑 memory
        memory = ConversationBufferDBMemory(conversation_id=conversation_id,
                                            llm=model_for_chat,
                                            chat_type=prompt_name,
                                            message_limit=10)

        chain = LLMChain(prompt=chat_prompt, llm=model_for_chat, memory=memory)

        task = asyncio.create_task(wrap_done(
            chain.acall({"input": query}),
            callback.done),
        )

        recommend_documents = recommend_documents if 'recommend_documents' in locals() else []

        if STREAM:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps({"answer": token}, ensure_ascii=False)
                yield json.dumps({"recommend course": recommend_documents}, ensure_ascii=False)

        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps({"answer": answer, "recommend course": recommend_documents},
                             ensure_ascii=False)
        await task

    return EventSourceResponse(recommend_base_chat_iterator(query, model_name, prompt_name))
