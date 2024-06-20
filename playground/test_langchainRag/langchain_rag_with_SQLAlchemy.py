#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用 SQLAlchemy 在 SQLite 数据库中保存聊天历史记录，需要进行一些关键的添加和修改。以下是这些更改的分步说明：
"""

# Step 1. 首先，导入必要的 SQLAlchemy 模块来设置数据库和 ORM（对象关系映射）。这些模块将允许以 Python 方式与 SQLite 数据库进行交互。
import bs4
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import ChatZhipuAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from zhipuai import ZhipuAI
import os
from dotenv import load_dotenv


# 在控制台先安装 pip install --upgrade  langchain langchain-community langchainhub httpx httpx-sse PyJWT langchain-chroma bs4 python-dotenv sqlalchemy

# Step 2. 在“.env”的文件中写入 ZHIPUAI_API_KEY， 并在当前文件中读取zhipu API Keys的信息
load_dotenv()

Base = declarative_base()

# Step 3. 定义可以使用Base类管理的模型类
class Session(Base):
    """
    Session 类表示聊天会话
    """
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True, nullable=False)
    messages = relationship("Message", back_populates="session")


class Message(Base):
    """
    Message 类表示会话中的各个消息
    """
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    session = relationship("Session", back_populates="messages")


def get_db():
    """
    创建一个实用程序函数来管理数据库会话。该函数将确保每个数据库会话正确打开和关闭。
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_message(session_id: str, role: str, content: str):
    """
    定义一个函数将各个消息保存到数据库中。该函数检查会话是否存在；如果没有，它就会创建一个。然后它将消息保存到相应的会话中。
    """
    db = next(get_db())
    try:
        # Check if the session already exists
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if not session:
            # Create a new session if it doesn't exist
            session = Session(session_id=session_id)
            db.add(session)
            db.commit()
            db.refresh(session)

        # Add the message to the session
        db.add(Message(session_id=session.id, role=role, content=content))
        db.commit()
    except SQLAlchemyError:
        db.rollback()
    finally:
        db.close()

def load_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    定义一个函数来从数据库加载聊天历史记录。此函数检索与给定会话 ID 关联的所有消息并重建聊天历史记录。
    """
    db = next(get_db())
    chat_history = ChatMessageHistory()
    try:
        # Retrieve the session
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if session:
            # Add each message to the chat history
            for message in session.messages:
                chat_history.add_message({"role": message.role, "content": message.content})
    except SQLAlchemyError:
        pass
    finally:
        db.close()

    return chat_history


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    更新 get_session_history 函数以从数据库检索会话历史记录，而不是仅使用内存存储。
    """
    if session_id not in store:
        store[session_id] = load_session_history(session_id)
    return store[session_id]


def save_all_sessions():
    """
    添加退出应用程序之前保存所有会话的功能。此函数迭代内存中的所有会话并将其消息保存到数据库中。
    增加错误处理以确保程序稳定性。
    """
    from langchain_core.messages import HumanMessage
    from langchain_core.messages import AIMessage


    for session_id, chat_history in store.items():
        for message in chat_history.messages:
            # 检查 message 是否为字典类型
            if isinstance(message, dict):
                # 检查字典是否包含必要的键
                if "role" in message and "content" in message:
                    save_message(session_id, message["role"], message["content"])
                else:
                    print(f"Skipped a message due to missing keys: {message}")
            # 处理 HumanMessage 和 AIMessage 类型的消息
            elif isinstance(message, HumanMessage):
                save_message(session_id, "human", message.content)
            elif isinstance(message, AIMessage):
                save_message(session_id, "ai", message.content)
            else:
                print(f"Skipped an unsupported message type: {message}")



def invoke_and_save(session_id, input_text):
    """
    修改链式调用函数，同时保存用户问题和AI答案。这确保了每次交互都被记录下来。
    """
    # Save the user question with role "human"
    save_message(session_id, "human", input_text)

    # Get the AI response
    result = conversational_rag_chain.invoke(
        {"input": input_text},
        config={"configurable": {"session_id": session_id}}
    )["answer"]

    print(f"invoke_and_save:{result}")

    # Save the AI answer with role "ai"
    save_message(session_id, "ai", result)

    return result


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


if __name__ == '__main__':

    # Step 1. 定义模型示例
    chat = ChatZhipuAI(
        model="glm-4",
        temperature=0.5,
    )

    # Step 2. 定义 SQLite 数据库以及用于存储会话和消息的模型。
    DATABASE_URL = "sqlite:///chat_history.db"


    # Step 3. 创建模型类. 官网：https://docs.sqlalchemy.org/en/20/orm/quickstart.html
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)

    # Step 4. 构建Session 来管理会话。官方Docs：https://docs.sqlalchemy.org/en/20/orm/session_basics.html
    SessionLocal = sessionmaker(bind=engine)

    # 使用 atexit 模块来注册一个函数 save_all_sessions，这个函数将在Python程序即将正常终止时自动执行。目的是在程序退出前保存所有会话数据，以确保不会因程序突然终止而丢失数据。
    import atexit

    atexit.register(save_all_sessions)

    ### Construct retriever ###
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    # 创建嵌入生成器实例
    embedding_generator = EmbeddingGenerator(model_name="embedding-2")

    # 文本列表
    texts = [content for document in splits for split_type, content in document if split_type == 'page_content']

    # 创建 Chroma VectorStore
    chroma_store = Chroma(
        collection_name="example_collection",
        embedding_function=embedding_generator,  # 使用定义的嵌入生成器实例
        create_collection_if_not_exists=True
    )

    # 添加文本到 Chroma VectorStore
    IDs = chroma_store.add_texts(texts=texts)
    # print("Added documents with IDs:", IDs)

    # 使用 Chroma VectorStore 创建检索器
    retriever = chroma_store.as_retriever()

    ### Contextualize question ###
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
    history_aware_retriever = create_history_aware_retriever(
        chat, retriever, contextualize_q_prompt
    )

    ### Answer question ###
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

    ### Statefully manage chat history ###
    store = {}

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    result = invoke_and_save("abc123", "What is Task Decomposition?")
    print(result)

    # 此命令指示 vectorstore 删除其保存的整个数据集合。这里的集合是指所有文档（文本片段）及其相应的已被索引并存储在向量存储中的向量表示的集合。
    chroma_store.delete_collection()