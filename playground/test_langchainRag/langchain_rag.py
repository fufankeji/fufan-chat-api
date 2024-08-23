#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在LangChain中使用 GLM-4 实现基于 Chroma 向量数据库的RAG完整过程。
"""

# 1. 在控制台先安装 pip install --upgrade langchain langchain-community
# langchainhub httpx httpx-sse PyJWT langchain-chroma bs4 python-dotenv

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatZhipuAI
from zhipuai import ZhipuAI
import os
from dotenv import load_dotenv


# Step 1. 在“.env”的文件中写入 ZHIPUAI_API_KEY， 并在当前文件中读取zhipu API Keys的信息
load_dotenv()

# Step 2  . 初始化模型, 该行初始化与 智谱 的 GLM - 4  模型进行连接，将其设置为处理和生成响应。
chat = ChatZhipuAI(
    model="glm-4",
    temperature=0.8,
)

# # Step 3 . WebBaseLoader 配置为专门从 Lilian Weng 的博客文章中抓取和加载内容。它仅针对网页的相关部分（例如帖子内容、标题和标头）进行处理。
# """
# LangChain中关于 文本加载器的集成：
#  - 原生实现的：https://python.langchain.com/v0.2/docs/how_to/#document-loaders
#  - 外部集成的：https://python.langchain.com/v0.2/docs/integrations/document_loaders/
#
# bs4：是 Beautiful Soup 4，一个用于从 HTML 和 XML 文件中提取数据的 Python 库。它可以在这种情况下用于解析作为源检索的网页。
# WebBaseLoader： langchain_community.document_loaders 中的一个组件，用于从基于 Web 的源加载文档。
# """
#
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)


#
# # LangChain 规范下统一的 Document 对象
# # API Docs：https://api.python.langchain.com/en/latest/documents/langchain_core.documents.base.Document.html#langchain_core.documents.base.Document
docs = loader.load()

#
#
# # Step 4. 使用 RecursiveCharacterTextSplitter 将内容分割成更小的块，这有助于通过将长文本分解为可管理的大小并有一些重叠来保留上下文来管理长文本。
# """
# LangChain中关于文档切分（Text splitters）：
# - 原生实现的：https://python.langchain.com/v0.2/docs/how_to/#text-splitters
# - 文档链接：https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/
# - 源码：https://api.python.langchain.com/en/latest/_modules/langchain_text_splitters/base.html#TextSplitter
# """
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
#
# # 直观展示文档切分的一个可视化页面：https://chunkviz.up.railway.app/
# # splits 是一个列表，其中每个元素也是一个列表，表示一个文档的分割结果
# # for doc_index, doc_splits in enumerate(splits):
# #     print(f"Document {doc_index + 1}:")  # 显示文档编号
# #     for split_index, split_text in enumerate(doc_splits):
# #         print(f"  Split {split_index + 1}: {split_text[:50]}...")  # 打印每个分段的前50个字符
# #     print("\n" + "-"*60 + "\n")  # 在每个文档之间加入分隔线，增加可读性
# # #
# # #
# # Step 4. 对每个块，构建Embeddings。 智谱的 GLM Embedding 在线模型：https://open.bigmodel.cn/dev/api#text_embedding
# # embeddings = []
# #
# # from zhipuai import ZhipuAI
# #
# # #GLM 4 官方调用说明：https://open.bigmodel.cn/dev/api#vector
# #
# # client = ZhipuAI()
# #
# # for doc_splits in splits:
# #     for split_type, split_content in doc_splits:
# #         if split_type == 'page_content' and split_content.strip():  # 确保处理的是 'page_content' 且内容不为空
# #             try:
# #                 response = client.embeddings.create(
# #                     model="embedding-2",
# #                     input=split_content
# #                 )
# #                 if hasattr(response, 'data'):
# #                     embeddings.append(response.data[0].embedding)
# #
# #                 else:
# #                     print("未能成功获取嵌入向量")
# #             except Exception as e:
# #                 print(f"请求失败，错误信息：{e}")
# #
# # # # 打印嵌入向量
# # for i, embedding in enumerate(embeddings):
# #     print(f"Embedding {i + 1}: {embedding[:3]}...")  # 仅展示前10个值以示例
# #
# # #
# # step 5. Chroma 使用 GLM 4 的 Embedding 模型 提供的嵌入从这些块创建向量存储，从而促进高效检索。
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
#

embedding_generator = EmbeddingGenerator(model_name="embedding-2")

# 文本列表
texts = [content for document in splits for split_type, content in document if split_type == 'page_content']


# Step 6. 创建 Chroma VectorStore， 并存入向量。
# 源码：https://api.python.langchain.com/en/latest/_modules/langchain_chroma/vectorstores.html#Chroma
chroma_store = Chroma(
    collection_name="example_collection",
    embedding_function=embedding_generator,  # 使用定义的嵌入生成器实例
    create_collection_if_not_exists=True
)
#
#
# # 添加文本到 Chroma VectorStore
IDs = chroma_store.add_texts(texts=texts)

print("Added documents with IDs:", IDs)

#
# # vectorstore 被转换为一个检索器，能够根据查询获取最相关的文本片段。
# # 自定义函数 format_docs 用于适当地格式化这些片段。
# # RAG 链集成了检索、提示工程（通过 hub.pull ）和初始化语言模型 ( llm ) 来处理查询并生成响应，最后使用 StrOutputParser 。
# # 使用 Chroma VectorStore 创建检索器
# # 官方文档：https://python.langchain.com/v0.2/docs/how_to/vectorstore_retriever/
# # 源码：https://api.python.langchain.com/en/latest/vectorstores/langchain_core.vectorstores.VectorStore.html#langchain_core.vectorstores.VectorStore.as_retriever
# retriever = chroma_store.as_retriever()
#
# # 这里从 'hub.pull' 是从某处获取提示的方法
# prompt = hub.pull("rlm/rag-prompt")
#
#
# # 自定义函数 format_docs 用于适当地格式化这些片段。
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)
#
#
# # vectorstore 被转换为一个检索器，能够根据查询获取最相关的文本片段。
#
# # Step 7. RAG 链集成了检索、提示工程（通过 hub.pull ）和初始化语言模型 ( llm ) 来处理查询并生成响应，最后使用 StrOutputParser 。
# """
# 其过程如下：
# 1. 查询输入：字符串“什么是任务分解？”作为输入传递给 rag_chain 。
# 2. 上下文检索：链的 retriever 组件链接到矢量存储，激活以从索引的博客内容中查找并获取最相关的文本片段。这些片段是根据与问题的语义相似性与查询最匹配的片段。
# 3. 格式化检索的内容：然后， format_docs 函数获取这些检索到的文档，并将它们格式化为单个字符串，每个文档内容由双换行符分隔。此格式化字符串提供了一个连贯的上下文，其中封装了回答查询所需的所有相关信息。
# 4. 生成答案：此格式化上下文字符串与查询一起被输入到 glm-4 模型中。该模型使用提供的上下文和查询的细节，根据检索到的信息生成上下文相关且准确的响应。
# 5. 输出解析：最后， ChatZhipu 模型生成的响应由 StrOutputParser 进行解析，将模型的输出转换为干净的、用户可读的格式。
#
#
# RunnableParallel 可以并发执行多个任务，而 RunnablePassthrough 用于需要顺序执行而不需修改的任务。
# """
#
# #
# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | chat
#     | StrOutputParser()
# )
#
#
# # Step 8. 进行提问
# """
# 1. 查询处理：该命令接受查询“什么是任务分解？”并将其传递给 retriever 组件。检索器本质上是系统中的搜索功能，设置为在预先索引的数据集中查找信息 - 这里是根据博客内容创建的矢量存储。
# 2. 语义搜索：检索器使用向量存储中存储的文本片段的嵌入（向量表示）来执行语义搜索。它将查询的向量表示与存储的片段的向量进行比较，以识别在语义上与查询最相似的片段。
# 3. 检索相关文本片段：根据相似度分数，检索器从博客中选择并返回与查询最匹配的文本片段。这些片段包含被认为与回答任务分解问题最相关的信息。
# """
#
# rag_res = rag_chain.invoke("What is Task Decomposition?")
# print(rag_res)
#
#
# # Step 9. 此命令指示 vectorstore 删除其保存的整个数据集合。这里的集合是指所有文档（文本片段）及其相应的已被索引并存储在向量存储中的向量表示的集合。
# chroma_store.delete_collection()