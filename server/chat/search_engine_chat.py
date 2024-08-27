from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler

from langchain.prompts.chat import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from fastapi import Body
from fastapi.concurrency import run_in_threadpool
from sse_starlette import EventSourceResponse
from server.utils import wrap_done, get_ChatOpenAI
from server.utils import BaseResponse, get_prompt_template
from server.chat.utils import History
from typing import AsyncIterable
import asyncio
import json
from typing import List, Optional, Dict
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
from markdownify import markdownify
from configs import (LLM_MODELS, SEARCH_ENGINE_TOP_K, TEMPERATURE, MAX_TOKENS, STREAM,
                     USE_RERANKER,
                     RERANKER_MODEL,
                     RERANKER_MAX_LENGTH,
                     VECTOR_SEARCH_TOP_K,
                     SCORE_THRESHOLD,
                     )
from server.utils import wrap_done, get_ChatOpenAI, get_model_path
from server.reranker.reranker import LangchainReranker
from server.utils import embedding_device

from server.chat.utils import search, fetch_details
from server.reranker.search_reranker import reranking
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.knowledge_base.kb_service.milvus_kb_service import MilvusKBService
from text_splitter.chinese_recursive_text_splitter import ChineseRecursiveTextSplitter
from server.db.repository.message_repository import add_message_to_db
from server.memory.conversation_db_buffer_memory import ConversationBufferDBMemory
from server.callback_handler.conversation_callback_handler import ConversationCallbackHandler
from langchain.prompts import PromptTemplate


async def search_engine_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                             conversation_id: str = Body("", description="对话框ID"),
                             knowledge_base_name: str = Body("real_time_search", description="知识库名称", examples=["samples"]),
                             retrival_top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                             search_top_k: int = Body(SEARCH_ENGINE_TOP_K, description="检索结果数量"),
                             model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                             prompt_name: str = Body("real_time_search",
                                                     description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                             ):
    async def search_engine_chat_iterator(query: str,
                                          search_top_k: int,
                                          model_name: str = LLM_MODELS[0],
                                          prompt_name: str = prompt_name,
                                          ) -> AsyncIterable[str]:

        callback = AsyncIteratorCallbackHandler()
        callbacks = [callback]

        # 构造一个新的Message_ID记录
        message_id = await add_message_to_db(query=query,
                                             conversation_id=conversation_id,
                                             prompt_name=prompt_name,
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

        # 根据用户输入的问题，调用SerperAPI执行联网检索，返回search_top_k个相关的链接
        search_results = await search(query, search_top_k)

        # 对检索到的网址链接，通过计算 query 和 每个网站的简介，进一步做 rerank，提取最相关的一个网址
        rerank_results = reranking(query, search_results)

        # 对经过rerank 的 网站，提取主体内容。
        detail_results = await fetch_details(rerank_results)
        context = " "
        if detail_results == []:
            prompt_template = get_prompt_template(prompt_name, "empty")
        else:
            # 提取向量数据库的实例,联网检索，统一进入同一个默认向量库中检索
            milvusService = MilvusKBService(knowledge_base_name)

            # 添加文档到 milvus 服务
            await milvusService.do_add_doc(docs=detail_results)

            search_retriever = await milvusService.search_docs(query, top_k=retrival_top_k)

            context = "\n".join([doc[0].page_content for doc in search_retriever])

            if len(search_retriever) == 0:  # 如果没有找到相关文档，使用empty模板
                prompt_template = get_prompt_template(prompt_name, "empty")
            else:
                prompt_template = get_prompt_template(prompt_name, "chat_with_search")

        # 这里需要根据会话ID中的对话类型，选择匹配的历史对话信息
        memory = ConversationBufferDBMemory(conversation_id=conversation_id,
                                            llm=model,
                                            chat_type=prompt_name,
                                            message_limit=10)

        system_msg = History(role="system", content="你现在得到的上下文是基于实时联网检索信息后提取得到的，你需要从中提取关键信息，并基于这些关键信息回答用户提出的问题。").to_msg_template(is_raw=False)

        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages([system_msg, input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model, memory=memory)


        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )

        if detail_results == []:
            retriever_documents = []
            search_documents = []
        else:
            retriever_documents = []
            search_documents = []
            for inum, doc_tuple in enumerate(search_retriever):
                doc, _ = doc_tuple  # 每个元组的结构是 (Document对象, 相似度得分)
                page_content = doc.page_content.replace('__', '').replace('__', '').replace('__', '').replace('__',
                                                                                                                 '').replace(
                    '__', '').replace('__', '')

                text = f"""向量检索 [{inum + 1}]\n\n{page_content}\n\n"""
                retriever_documents.append(text)

            for inum, doc in enumerate(search_results):
                url = doc['link']
                snippet = doc['snippet']
                # text = f"""联网检索 [{inum + 1}] ({url})\n\n{snippet}\n\n"""
                text = f"**实时联网检索 [{inum + 1}]** - [{snippet}]({url}) <sup>{inum + 1}</sup>"
                search_documents.append(text)

        if STREAM:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps({"text": token}, ensure_ascii=False)
            yield json.dumps({"docs": retriever_documents}, ensure_ascii=False)
            yield json.dumps({"search": search_documents}, ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps({"text": answer, "docs": retriever_documents, "search": search_documents},
                             ensure_ascii=False)
        await task

    return EventSourceResponse(search_engine_chat_iterator(query=query,
                                                           search_top_k=search_top_k,
                                                           model_name=model_name,
                                                           prompt_name=prompt_name),
                               )
