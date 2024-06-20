#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在许多问答应用程序中，促进用户和系统之间的动态、来回对话至关重要。这要求应用程序维护过去交互的“记忆”，从而允许它引用之前的交换并将其集成到当前的处理中。
本文重点介绍在问答框架内实现处理历史消息的逻辑，让大家了解如何将聊天历史记录合并到 RAG 模型中，以维护上下文并提高类似聊天的对话中的交互质量

具体来看，我们要在 langchain_rag.py的实现基础上：
- 修改应用程序的提示以包含历史消息作为输入。这一变化确保系统可以访问之前的交互，并使用它们来更有效地理解和响应新的查询。
- 引入一个新组件，通过将最新的用户问题放置在聊天历史记录的上下文中来处理该问题。
  eg：例如，如果用户询问“您能详细说明第二点吗？”，系统必须参考先前的上下文才能提供有意义的响应。如果没有这种历史背景，有效检索和生成相关答案将具有挑战性。
"""

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


# 在控制台先安装 pip install --upgrade  langchain langchain-community langchainhub httpx httpx-sse PyJWT langchain-chroma bs4 python-dotenv

# Step 1. 在“.env”的文件中写入 ZHIPUAI_API_KEY， 并在当前文件中读取zhipu API Keys的信息
load_dotenv()


# Step 2. 构建一个能够利用历史消息和最新的用户问题的子链
"""
为了使应用程序能够处理涉及先前交互的问题， 需要先建立一个流程（称为子链），
该子链旨在每当引用过去的讨论时重新表述问题，具体来看：
- 在提示结构中合并了一个名为“chat_history”的变量，它充当历史消息的占位符。通过使用“chat_history”输入键，我们可以将以前的消息列表无缝地注入到提示中。
- 这些消息战略性地放置在系统响应之后和用户提出的最新问题之前，以确保上下文得到维护。
"""


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# 此提示告诉模型接收聊天历史记录和用户的最新问题，然后重新表述问题，以便可以独立于聊天历史记录来理解问题。明确指示模型不要回答问题，而是在必要时重新表述问题。

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""


# Step 3. 创建提示模板来构建模型的交互
# 该模板包括带有说明的系统消息、聊天历史记录的占位符 ( MessagesPlaceholder ) 以及由 {input} 标记的最新用户问题。
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


# Step 4. 初始化模型, 该行初始化与 智谱 的 GLM - 4  模型进行连接，将其设置为处理和生成响应。
chat = ChatZhipuAI(
    model="glm-4",
    temperature=0.8,
)



# Step 5 . WebBaseLoader 配置为专门从 Lilian Weng 的博客文章中抓取和加载内容。它仅针对网页的相关部分（例如帖子内容、标题和标头）进行处理。
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()


# Step 6. 使用 RecursiveCharacterTextSplitter 将内容分割成更小的块，这有助于通过将长文本分解为可管理的大小并有一些重叠来保留上下文来管理长文本。
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)


# Step 7. Chroma 使用 GLM 4 的 Embedding 模型 提供的嵌入从这些块创建向量存储，从而促进高效检索。
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


# 创建嵌入生成器实例
embedding_generator = EmbeddingGenerator(model_name="embedding-2")


# 文本列表
texts = [content for document in splits for split_type, content in document if split_type == 'page_content']

# Step 8. 创建 Chroma VectorStore
chroma_store = Chroma(
    collection_name="example_collection",
    embedding_function=embedding_generator,  # 使用定义的嵌入生成器实例
    create_collection_if_not_exists=True
)

# 添加文本到 Chroma VectorStore
IDs = chroma_store.add_texts(texts=texts)



# Step 9. 使用 Chroma VectorStore 创建检索器
retriever = chroma_store.as_retriever()


# Step 10. 设置历史信息感知检索器：
# create_history_aware_retriever 函数旨在接受输入和“chat_history”的键，用于创建集成聊天历史记录以进行上下文感知处理的检索器。
# 官方文档：https://python.langchain.com/v0.1/docs/modules/chains/
from langchain.chains import create_history_aware_retriever
"""
如果历史记录存在，它会构建一个有效组合提示、大型语言模型 (LLM) 和结构化输出解析器 ( StrOutputParser ) 的序列，后跟检索器。此顺序可确保最新问题在累积的历史数据中得到体现。
"""
history_aware_retriever = create_history_aware_retriever(chat,
                                                         retriever,
                                                         contextualize_q_prompt
)

# 至此，子链构建结束


# Step 11. 构建提示
prompt = hub.pull("rlm/rag-prompt")


# # 自定义函数 format_docs 用于适当地格式化这些片段。
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



from langchain.chains import create_retrieval_chain # 此函数用于创建将检索功能与处理函数或模型集成的链。


# Step 12. 定义 QA 系统的提示模板，指定系统应如何根据检索到的上下文响应输入。
# 该字符串设置语言模型的指令，指示它使用提供的上下文来简洁地回答问题。如果答案未知，则指示模型明确说明这一点。
qa_system_prompt = """You are an assistant for question-answering tasks. \  
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
# 此函数用于创建一个将文档处理与其他流程相结合的链，通常涉及文档检索和在问答等任务中的使用。
from langchain.chains.combine_documents import create_stuff_documents_chain

# Step 13 构建问答链：question_answer_chain 是使用 create_stuff_documents_chain 函数创建的，该函数利用语言模型 ( llm ) 和定义的提示 ( qa_prompt )。
# 官方文档链接：https://python.langchain.com/v0.1/docs/modules/chains/
question_answer_chain = create_stuff_documents_chain(chat, qa_prompt)

# Step 14. 组装 RAG 链条：该链代表完整的工作流程，其中历史感知检索器首先处理查询以合并任何相关的历史上下文，然后由 question_answer_chain 处理处理后的查询以生成最终答案。
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# 以下代码演示了如何使用 RAG 链来处理一系列问题，并能够引用之前的交互。该代码模拟聊天交互，其中用户提出问题，收到答案，然后提出可以利用初始交流上下文的后续问题。以下是包含代码片段的详细说明：
from langchain_core.messages import HumanMessage

# 聊天历史记录被初始化为空列表。该列表将存储会话期间交换的消息以维护上下文。
chat_history = []

# 第一个问题和响应：定义一个问题，并使用该问题和当前（空）聊天历史记录调用 RAG 链。
question = "What is Task Decomposition?"
ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
# print("First ans: %s" % ai_msg_1["answer"])

# 然后，用户的问题和 AI 生成的答案分别作为 HumanMessage 实例和响应对象添加到聊天历史记录中。
chat_history.extend([HumanMessage(content=question), ai_msg_1["answer"]])

# 第二个问题和响应：利用现在包含第一次交流上下文的更新的聊天历史记录，提出后续问题。
second_question = "What are common ways of doing it?"
ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})

# RAG 链再次被调用，这次是第二个问题和更新的聊天历史记录，使其在生成响应时能够考虑之前的交互。
# print("Second ans: %s " % ai_msg_2["answer"])

# Step 14. 此命令指示 vectorstore 删除其保存的整个数据集合。这里的集合是指所有文档（文本片段）及其相应的已被索引并存储在向量存储中的向量表示的集合。
chroma_store.delete_collection()


######################## 分割线  ########################

"""
到目前为止，我们探索了如何将历史交互集成到应用程序逻辑中。然而，我们一直在手动处理聊天历史记录——为每次用户交互更新和插入它。在强大的问答应用程序中，自动化此过程对于效率和可扩展性至关重要。
接下来看两个组件：
    - BaseChatMessageHistory：该组件用于存储聊天历史记录。
    - RunnableWithMessageHistory：这是 LCEL 链与 BaseChatMessageHistory 结合的包装器。它自动执行将聊天历史记录注入输入的过程，并在每次调用后更新它。
    - 官方文档：https://python.langchain.com/v0.1/docs/expression_language/how_to/message_history/
"""

import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1. 初始化模型, 该行初始化与 智谱 的 GLM - 4  模型进行连接，将其设置为处理和生成响应。
chat = ChatZhipuAI(
    model="glm-4",
    temperature=0.5,
)


# Step 2 . WebBaseLoader 配置为专门从 Lilian Weng 的博客文章中抓取和加载内容。它仅针对网页的相关部分（例如帖子内容、标题和标头）进行处理。
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()


# Step 3. 使用 RecursiveCharacterTextSplitter 将内容分割成更小的块，这有助于通过将长文本分解为可管理的大小并有一些重叠来保留上下文来管理长文本。
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)


# Step 4. Chroma 使用 GLM 4 的 Embedding 模型 提供的嵌入从这些块创建向量存储，从而促进高效检索。
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


# 创建嵌入生成器实例
embedding_generator = EmbeddingGenerator(model_name="embedding-2")


# 文本列表
texts = [content for document in splits for split_type, content in document if split_type == 'page_content']

# Step 5. 创建 Chroma VectorStore
chroma_store = Chroma(
    collection_name="example_collection",
    embedding_function=embedding_generator,  # 使用定义的嵌入生成器实例
    create_collection_if_not_exists=True
)

# 添加文本到 Chroma VectorStore
IDs = chroma_store.add_texts(texts=texts)



# Step 6. 使用 Chroma VectorStore 创建检索器
retriever = chroma_store.as_retriever()



# Step 6. 此提示告诉模型接收聊天历史记录和用户的最新问题，然后重新表述问题，以便可以独立于聊天历史记录来理解问题。明确指示模型不要回答问题，而是在必要时重新表述问题。
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Step 7. 构建问答的子链
history_aware_retriever = create_history_aware_retriever(
    chat, retriever, contextualize_q_prompt
)


# Step 8. 构建系统的链路信息
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(chat, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# Step 9. 使用基本字典结构管理聊天历史记录
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 官方Docs：https://python.langchain.com/v0.2/docs/how_to/message_history/
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# 现在我们问第一个问题
first_ans = conversational_rag_chain.invoke(
    {"input": "What is Task Decomposition?"},
    config={
        "configurable": {"session_id": "abc123"}
    },
)["answer"]


secone_ans = conversational_rag_chain.invoke(
    {"input": "What are common ways of doing it?"},
    config={"configurable": {"session_id": "abc123"}},
)["answer"]


print(f"first_ans:{first_ans}")
print(f"secone_ans:{secone_ans}")


# 此命令指示 vectorstore 删除其保存的整个数据集合。这里的集合是指所有文档（文本片段）及其相应的已被索引并存储在向量存储中的向量表示的集合。
chroma_store.delete_collection()