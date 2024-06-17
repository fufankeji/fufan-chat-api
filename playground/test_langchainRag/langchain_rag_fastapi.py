#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将数据流式传输到前端可以通过提供实时更新而无需不断轮询来增强用户体验。在本节中，我们将探讨如何在 FastAPI 应用程序中实现流式传输，并使用简单的 HTML 前端显示流式传输的数据。
具体来说：
-  首先，设置一个基本的 FastAPI 应用程序，将数据流式传输到前端。
-  使用langchain_rag_with_stream.py创建的 RAG 管道创建一个端点，该端点使用服务器发送事件 (SSE) 来流式传输事件。
"""

# Step 1. 在“.env”的文件中写入 ZHIPUAI_API_KEY， 并在当前文件中读取zhipu API Keys的信息
from dotenv import load_dotenv
load_dotenv()

# Step 2. 导入必要的库和模块
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessageChunk
# from logging import logging

from dotenv import load_dotenv
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatZhipuAI
from zhipuai import ZhipuAI


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
retriever = chroma_store.as_retriever().with_config(
    tags=["retriever"]
)

# Step 7. 实例化一个 Chat Model
chat = ChatZhipuAI(
    model="glm-4",
    temperature=0.8,
)


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
contextualize_q_chain = (contextualize_q_prompt | chat | StrOutputParser()).with_config(
    tags=["contextualize_q_chain"]
)

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    RunnablePassthrough.assign(context=contextualize_q_chain | retriever | format_docs)
    | qa_prompt
    | chat
).with_config(
    tags=["main_chain"]
)


# 定义 FastAPI 服务

app = FastAPI()

# 允许所有的入口访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return FileResponse("static/index.html")


def serialize_aimessagechunk(chunk):
    """
    AIMessageChunk对象的自定义序列化器。
    将AIMessageChunk对象转换为可序列化的格式。
    """
    if isinstance(chunk, AIMessageChunk):
        return chunk.content
    else:
        raise TypeError(
            f"Object of type {type(chunk).__name__} is not correctly formatted for serialization"
        )


async def generate_chat_events(message):
    """
    这段代码主要用于生成异步的聊天事件流，通过检查事件的标签和类型来筛选出需要的数据（答案、重构问题、背景文档），并将其序列化和格式化后输出。
    同时，对每个关键步骤进行了详细的注释，以便理解每部分代码的功能和目的。
    """
    try:
        # 异步迭代消息事件
        async for event in rag_chain.astream_events(message, version="v1"):
            # 仅获取答案的处理部分
            # 标识与获取最终答案相关的事件的标签。
            sources_tags = ['seq:step:3', 'main_chain']
            # 检查事件的标签是否包含所有指定的源标签，并且事件类型为聊天模型流
            if all(value in event["tags"] for value in sources_tags) and event["event"] == "on_chat_model_stream":
                # 序列化消息片段
                chunk_content = serialize_aimessagechunk(event["data"]["chunk"])

                # 确保消息内容非空
                if len(chunk_content) != 0:
                    # 创建数据字典并转换为JSON格式
                    data_dict = {"data": chunk_content}
                    data_json = json.dumps(data_dict)
                    # 生成数据流格式的输出
                    yield f"data: {data_json}\n\n"

            # 获取重构后的问题的处理部分
            # 标识与重新表述的问题相关的事件的标签。
            sources_tags = ['seq:step:2', 'main_chain', 'contextualize_q_chain']
            # 序列化消息片段
            if all(value in event["tags"] for value in sources_tags) and event["event"] == "on_chat_model_stream":
                chunk_content = serialize_aimessagechunk(event["data"]["chunk"])
                if len(chunk_content) != 0:
                    data_dict = {"reformulated": chunk_content}
                    data_json = json.dumps(data_dict)
                    yield f"data: {data_json}\n\n"

            # 获取背景信息的处理部分
            # 标识与检索上下文相关的事件的标签。
            sources_tags = ['main_chain', 'retriever']
            if all(value in event["tags"] for value in sources_tags) and event["event"] == "on_retriever_end":
                # 提取文档列表
                documents = event['data']['output']['documents']
                # 创建格式化后的文档列表
                formatted_documents = []

                for doc in documents:
                    # 检查 metadata 中是否有 'source'，如果没有则提供一个默认值
                    source = doc.metadata.get('source', '未知来源')

                    # 为每个文档创建新的字典，并保留所需格式
                    formatted_doc = {
                        'page_content': doc.page_content,
                        'metadata': {
                            'source': source,
                        },
                        'type': 'Document'
                    }
                    # 将格式化后的文桾示例添加到列表
                    formatted_documents.append(formatted_doc)

                # 创建最终输出字典
                final_output = {'context': formatted_documents}

                # 生成数据流格式的输出
                data_json = json.dumps(final_output)
                yield f"data: {data_json}\n\n"
            if event["event"] == "on_chat_model_end":
                print("Chat model has completed one response.")

    except Exception as e:
        print('error' + str(e))


@app.get("/chat_stream/{message}")
async def chat_stream_events(message: str):
    return StreamingResponse(generate_chat_events({"question": message, "chat_history": []}),
                             media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="192.168.110.131", port=8000)
