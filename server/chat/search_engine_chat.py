
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
from configs import (LLM_MODELS, SEARCH_ENGINE_TOP_K, TEMPERATURE,
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

async def search_engine_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                             top_k: int = Body(SEARCH_ENGINE_TOP_K, description="检索结果数量"),
                             history: List[History] = Body([],
                                                           description="历史对话",
                                                           examples=[[
                                                               {"role": "user",
                                                                "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                               {"role": "assistant",
                                                                "content": "虎头虎脑"}]]
                                                           ),
                             stream: bool = Body(False, description="流式输出"),
                             model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                             temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                             max_tokens: Optional[int] = Body(None,
                                                              description="限制LLM生成Token数量，默认None代表模型最大值"),
                             prompt_name: str = Body("default",
                                                     description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                             ):

    history = [History.from_data(h) for h in history]

    async def search_engine_chat_iterator(query: str,
                                          top_k: int,
                                          history: Optional[List[History]],
                                          model_name: str = LLM_MODELS[0],
                                          prompt_name: str = prompt_name,
                                          ) -> AsyncIterable[str]:
        nonlocal max_tokens
        callback = AsyncIteratorCallbackHandler()
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
        )

        # 根据用户输入的问题，执行联网检索，返回tok k 个相关的链接
        search_results = await search(query, top_k)

        # 对检索到的网址链接，通过计算 query 和 每个网站的简介，进一步做 rerank，提取最相关的一个网址
        rerank_results = reranking(query, search_results)

        # 对经过rerank 的 网站，提取主体内容。
        detail_results = await fetch_details(rerank_results)

        # 提取向量数据库的实例,联网检索，统一进入同一个默认向量库中检索
        milvusService = MilvusKBService("test")

        # 添加文档到 milvus 服务
        await milvusService.do_add_doc(docs=detail_results)
        
        search_retriever = await milvusService.search_docs(query, top_k=3)
        
        context = "\n".join([doc[0].page_content for doc in search_retriever])

        if len(search_retriever) == 0:  # 如果没有找到相关文档，使用empty模板
            prompt_template = get_prompt_template("knowledge_base_chat", "empty")
        else:
            prompt_template = get_prompt_template("knowledge_base_chat", prompt_name)

        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)

        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )

        retriever_documents = []
        for inum, doc_tuple in enumerate(search_retriever):
            doc, _ = doc_tuple  # 每个元组的结构是 (Document对象, 相似度得分)
            text = f"""出处 [{inum + 1}] \n\n{doc.page_content}\n\n"""
            retriever_documents.append(text)

        search_documents = []
        for inum, doc in enumerate(search_results):
            url = doc['link']
            snippet = doc['snippet']
            text = f"""出处 [{inum + 1}] ({url})\n\n{snippet}\n\n"""
            search_documents.append(text)


        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps({"answer": token}, ensure_ascii=False)
            yield json.dumps({"docs": retriever_documents}, ensure_ascii=False)
            yield json.dumps({"search": search_documents}, ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps({"answer": answer, "docs": retriever_documents, "search": search_documents}, ensure_ascii=False)
        await task

    return EventSourceResponse(search_engine_chat_iterator(query=query,
                                                           top_k=top_k,
                                                           history=history,
                                                           model_name=model_name,
                                                           prompt_name=prompt_name),
                               )
