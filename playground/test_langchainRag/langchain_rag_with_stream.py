#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
添加一个有关如何使用 FastAPI 实现流式传输的工作示例。

基于langchain_rag_with_history.py，我们继续深入研究将大模型问答无缝集成到流中的实用方法问答应用程序。

"""

# 在控制台先安装 pip install --upgrade langchain langchain-community langchainhub httpx httpx-sse PyJWT langchain-chroma bs4 python-dotenv

from dotenv import load_dotenv


# Step 1. 在“.env”的文件中写入 ZHIPUAI_API_KEY， 并在当前文件中读取zhipu API Keys的信息
load_dotenv()

# Step 2. 导入必要的库和模块
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.chat_models import ChatZhipuAI
from zhipuai import ZhipuAI
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 3 . WebBaseLoader 配置为专门从 Lilian Weng 的博客文章中抓取和加载内容。它仅针对网页的相关部分（例如帖子内容、标题和标头）进行处理。

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# Step 4. 使用 RecursiveCharacterTextSplitter 将内容分割成更小的块，这有助于通过将长文本分解为可管理的大小并有一些重叠来保留上下文来管理长文本。
"""
LangChain中关于文档切分（Text splitters）：
- 原生实现的：https://python.langchain.com/v0.2/docs/how_to/#text-splitters
- 文档链接：https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/
"""
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)


class EmbeddingGenerator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = ZhipuAI()

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(model=self.model_name, input=text)
            if hasattr(response, 'data') and response.data:
                embeddings.append(response.data[0].embedding)
            else:
                # 如果获取嵌入失败，返回一个零向量
                embeddings.append([0] * 1024)  # 假设嵌入向量维度为 1024
        return embeddings


    def embed_query(self, query):
        # 使用相同的处理逻辑，只是这次只为单个查询处理
        response = self.client.embeddings.create(model=self.model_name, input=query)
        if hasattr(response, 'data') and response.data:
            return response.data[0].embedding
        return [0] * 1024  # 如果获取嵌入失败，返回零向量

# GLM 4 官方调用说明：https://open.bigmodel.cn/dev/api#vector
embedding_generator = EmbeddingGenerator(model_name="embedding-2")

# 文本列表
texts = [content for document in splits for split_type, content in document if split_type == 'page_content']


# Step 5. 创建 Chroma VectorStore， 并存入向量。
# 源码：https://api.python.langchain.com/en/latest/_modules/langchain_chroma/vectorstores.html#Chroma
chroma_store = Chroma(
    collection_name="example_collection",
    embedding_function=embedding_generator,  # 使用定义的嵌入生成器实例
    create_collection_if_not_exists=True
)


# 添加文本到 Chroma VectorStore
IDs = chroma_store.add_texts(texts=texts)
# print("Added documents with IDs:", IDs)


# vectorstore 被转换为一个检索器，能够根据查询获取最相关的文本片段。
# 自定义函数 format_docs 用于适当地格式化这些片段。
# RAG 链集成了检索、提示工程（通过 hub.pull ）和初始化语言模型 ( llm ) 来处理查询并生成响应，最后使用 StrOutputParser 。

# 使用 Chroma VectorStore 创建检索器
# 官方文档：https://python.langchain.com/v0.2/docs/how_to/vectorstore_retriever/
retriever = chroma_store.as_retriever()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Step 6. 从 langChain hub 直接读取 Prompt
prompt = hub.pull("rlm/rag-prompt")

# 自定义函数 format_docs 用于适当地格式化这些片段。
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Step 7. 实例化一个 Chat Model
chat = ChatZhipuAI(
    model="glm-4",
    temperature=0.8,
)


# Step 8. RAG 链集成了检索、提示工程（通过 hub.pull ）和初始化语言模型 ( llm ) 来处理查询并生成响应，最后使用 StrOutputParser 。
"""
RunnableParallel 可以并发执行多个任务，而 RunnablePassthrough 用于需要顺序执行而不需修改的任务。

rag_chain_from_docs 和 rag_chain_with_source ：这些构造定义了使用 AI 模型检索文档和生成响应的数据流和执行流。
它们实现了上下文检索和响应生成的集成，确保在生成答案时有效地跟踪和使用源。
"""
# 此处用于将格式化字符串传递到下一个阶段。
rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | chat
    | StrOutputParser()
)

# RunnableParallel 实现上下文检索和问题处理的并行处理。
rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)


for chunk in rag_chain_with_source.stream("What is Task Decomposition"):
    print(chunk)


# 添加一些逻辑来编译返回的流：
output = {}
curr_key = None
for chunk in rag_chain_with_source.stream("What is Task Decomposition"):
    for key in chunk:
        if key not in output:
            output[key] = chunk[key]
        else:
            output[key] += chunk[key]

        # 当前键不等于前一个处理的键（curr_key），说明正在开始一个新的数据类型或部分的输出。这时，会在新的一行开始打印，并输出键和相应的值。
        if key != curr_key:
            print(f"\n\n{key}: {chunk[key]}", end="", flush=True)
        # 如果当前键与前一个键相同，表明还在处理同一类型的数据，继综输出当前块的内容，而不换行。
        else:
            print(chunk[key], end="", flush=True)
        curr_key = key



"""
接下来考虑一个场景：我们的目标不仅是流式传输过程的最终结果，还包括某些中间阶段。如何做？
以我们的聊天历史链为例：在这种情况下，我们首先重新表述用户的问题，然后再将其提交给检索器。
"""

# from operator import itemgetter
#
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.tracers.log_stream import LogEntry, LogStreamCallbackHandler
#
# # 创建一个用于重新制定问题的提示模板。该提示指示模型生成一个独立的问题，无需事先的聊天记录即可理解。
# contextualize_q_system_prompt = """Given a chat history and the latest user question \
# which might reference context in the chat history, formulate a standalone question \
# which can be understood without the chat history. Do NOT answer the question, \
# just reformulate it if needed and otherwise return it as is."""
#
# # 此方法通过标记来配置提示模板，用于跟踪或识别较大系统或日志中的配置。
# contextualize_q_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", contextualize_q_system_prompt),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("human", "{question}"),
#     ]
# ).with_config(tags=["contextualize_q_system_prompt"])
#
# # 将提示模板 ( contextualize_q_prompt )、语言模型 ( chat ) 和字符串输出解析器 ( StrOutputParser ) 链接在一起，创建数据流管道。
# # 与上面类似， .with_config(tags=["contextualize_q_chain"]) 标记整个链，用于后续流程的记录。
# contextualize_q_chain = (contextualize_q_prompt | chat | StrOutputParser()).with_config(
#     tags=["contextualize_q_chain"]
# )
#
# qa_system_prompt = """You are an assistant for question-answering tasks. \
# Use the following pieces of retrieved context to answer the question. \
# If you don't know the answer, just say that you don't know. \
# Use three sentences maximum and keep the answer concise.\
#
# {context}"""
# qa_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", qa_system_prompt),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("human", "{question}"),
#     ]
# )
#
#
# def contextualized_question(input: dict):
#     """
#     功能逻辑：根据聊天记录的存在，确定是通过重构链传递问题还是按原样返回问题。
#     """
#     if input.get("chat_history"):
#         return contextualize_q_chain
#     else:
#         return input["question"]
#
# # 构建具有上下文检索功能的完整问答系统：链的这一部分处理上下文，检索必要的文档，并在输入 QA 提示之前格式化它们。
# rag_chain = (
#     RunnablePassthrough.assign(context=contextualized_question | retriever | format_docs)
#     | qa_prompt
#     | chat
# )
#
# # 至此，该链就定义了如何使用 LangChain 将聊天历史集成到动态问答流程中，确保问题具有情境性，答案由相关数据提供。
#
#
# # 流式传输重新表述的问题（中间步骤）
# from langchain_core.messages import HumanMessage
#
# # 初始化一个空列表，用于跟踪对话的历史记录。
# chat_history = []
#
# question = "What is Task Decomposition?"
# ai_msg = rag_chain.invoke({"question": question, "chat_history": chat_history})
#
# # 将用户的问题和 AI 的响应添加到聊天记录中。这对于维持持续交互中的上下文非常重要。
# chat_history.extend([HumanMessage(content=question), ai_msg])
#
# # 初次交互后，进行的第二轮提问
# second_question = "What are common ways of doing it?"
#
# # 异步生成器函数流式传输在处理第二个问题期间生成的事件。它可以用于实时调试或监控系统。
#
#
# import asyncio
#
# async def stream_responses():
#     async for chunk in rag_chain.astream_events(
#         {"question": second_question, "chat_history": chat_history},
#         include_tags=["contextualize_q_system_prompt"],    # 将事件过滤为带有 "contextualize_q_system_prompt" 标记的事件。该标签与处理问题重新表述的链部分相关联，确保仅传输与该特定任务相关的事件。
#         include_names=["StrOutputParser"],                 # 过滤事件以包含来自名为 "StrOutputParser" 的组件的事件
#         include_types=["on_parser_end"],                   # 过滤解析结束时触发的事件（ "on_parser_end" ），表示数据解析步骤的完成。
#         version="v1",
#     ):
#         print(chunk)
#
# """
# 该代码的主要目的是以上下文感知的方式与基于人工智能的问答系统进行交互，利用对话历史记录来提高响应准确性。
# 此外，事件的流式传输和过滤可以深入了解系统的特定操作阶段，特别是关注如何在内部重新制定和处理问题。这对于开发人员和工程师调试或优化系统性能特别有价值。
# """
# asyncio.run(stream_responses())
#
