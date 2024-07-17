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
                     TEMPERATURE)
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
from configs.model_config import LLM_MODELS, TEMPERATURE, MAX_TOKENS, STREAM, SCORE_THRESHOLD, VECTOR_SEARCH_TOP_K, RERANKER_TOP_K


async def knowledge_base_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                              conversation_id: str = Body("", description="对话框ID"),
                              knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
                              model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                              top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                              prompt_name: str = Body(
                                  "knowledge_base_chat",
                                  description="使用的prompt模板名称(在configs/prompt_config.py中配置)"
                              ),
                              request: Request = None,
                              ):
    # 提取向量数据库的实例
    kb = await KBServiceFactory.get_service_by_name(knowledge_base_name)  # default @ bge-large-zh-v1.5

    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    async def knowledge_base_chat_iterator(
            query: str,
            top_k: int,
            model_name: str = model_name,
            prompt_name: str = prompt_name,
    ) -> AsyncIterable[str]:
        callback = AsyncIteratorCallbackHandler()

        callbacks = [callback]

        # 构造一个新的Message_ID记录
        message_id = await add_message_to_db(
            conversation_id=conversation_id,
            prompt_name=prompt_name,
            query=query
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

        docs = await search_docs(query=query,
                                 knowledge_base_name=knowledge_base_name,
                                 top_k=top_k,
                                 score_threshold=SCORE_THRESHOLD)
        
        # 加入reranker
        if USE_RERANKER:
            reranker_model_path = get_model_path(RERANKER_MODEL)
            reranker_model = LangchainReranker(top_n=RERANKER_TOP_K,
                                               device=embedding_device(),
                                               max_length=RERANKER_MAX_LENGTH,
                                               model_name_or_path=reranker_model_path
                                               )

            docs = reranker_model.compress_documents(documents=docs,
                                                     query=query)

        context = "\n".join([doc.page_content for doc in docs])

        if len(docs) == 0:  # 如果没有找到相关文档，使用empty模板
            prompt_template = get_prompt_template(prompt_name, "empty")
        else:
            prompt_template = get_prompt_template(prompt_name, "chat_with_retrieval")

            # 这里需要根据会话ID中的对话类型，选择匹配的历史对话信息
            memory = ConversationBufferDBMemory(conversation_id=conversation_id,
                                                llm=model,
                                                chat_type=prompt_name,
                                                message_limit=10)

        system_msg = History(role="system", content="你是一位善于结合历史对话信息，以及相关文档分析并回答问题的高智商人才").to_msg_template(is_raw=False)

        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages([system_msg, input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model, memory=memory)

        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )

        source_documents = []
        for inum, doc in enumerate(docs):
            filename = os.path.basename(doc.metadata.get("source"))
            parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
            base_url = request.base_url
            url = f"{base_url}knowledge_base/download_doc?" + parameters

            text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
            source_documents.append(text)

        if len(source_documents) == 0:  # 没有找到相关文档
            source_documents.append(f"<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>")

        if STREAM:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps({"answer": token}, ensure_ascii=False)
            yield json.dumps({"docs": source_documents}, ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps({"answer": answer,
                              "docs": source_documents},
                             ensure_ascii=False)
        await task

    return EventSourceResponse(knowledge_base_chat_iterator(query, top_k, model_name, prompt_name))
