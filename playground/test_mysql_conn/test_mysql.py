import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text

# 更新以下字段为你本地数据库的实际用户名、密码和数据库名
username = 'root'
hostname = '192.168.110.131'
database_name = 'fufan'

from urllib.parse import quote

# 使用 quote 函数对密码进行编码
password_encoded = quote('Snowball2019)&@(')

# 现在使用编码后的密码构建连接字符串
SQLALCHEMY_DATABASE_URI = f"mysql+asyncmy://{username}:{password_encoded}@{hostname}/{database_name}?charset=utf8mb4"

print(SQLALCHEMY_DATABASE_URI)


async_engine = create_async_engine(
    SQLALCHEMY_DATABASE_URI,
    echo=True,
)

AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)


from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String

Base = declarative_base()

# class TestTable(Base):
#     __tablename__ = 'test_table'
#     id = Column(Integer, primary_key=True)
#     name = Column(String(50), nullable=False)

from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, func
from sqlalchemy.orm import relationship
from server.db.base import Base


class UserModel(Base):
    __tablename__ = 'user'
    id = Column(String(32), primary_key=True, comment='用户ID')
    username = Column(String(255), unique=True, comment='用户名')
    password_hash = Column(String(255), comment='密码的哈希值')
    # 你可以添加更多用户相关的字段，如邮箱、电话等

    messages = relationship('MessageModel', back_populates='user')

    def __repr__(self):
        return f"<User(id='{self.id}', username='{self.username}')>"



class MessageModel(Base):
    """
    聊天记录模型
    """
    __tablename__ = 'message'
    id = Column(String(32), primary_key=True, comment='聊天记录ID')
    user_id = Column(String(32), ForeignKey('user.id'), comment='用户ID')
    user = relationship('UserModel', back_populates='messages')

    # chat/agent_chat等
    chat_type = Column(String(50), comment='聊天类型')
    query = Column(String(4096), comment='用户问题')
    response = Column(String(4096), comment='模型回答')

    # 记录知识库id等，以便后续扩展
    meta_data = Column(JSON, default={})

    # 满分100 越高表示评价越好
    feedback_score = Column(Integer, default=-1, comment='用户评分')
    feedback_reason = Column(String(255), default="", comment='用户评分理由')
    create_time = Column(DateTime, default=func.now(), comment='创建时间')

    def __repr__(self):
        return f"<Message(id='{self.id}', user_id='{self.user_id}', chat_type='{self.chat_type}', query='{self.query}', response='{self.response}', meta_data='{self.meta_data}', feedback_score='{self.feedback_score}', feedback_reason='{self.feedback_reason}', create_time='{self.create_time}')>"





async def create_tables(engine):
    async with engine.begin() as conn:
        # 使用 Base.metadata.create_all 创建所有继承 Base 的模型对应的表
        await conn.run_sync(Base.metadata.create_all)

if __name__ == "__main__":
    # asyncio.run(test_db_connection())
    asyncio.run(create_tables(async_engine))

