import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text




# 更新以下字段为你本地数据库的实际用户名、密码和数据库名
username = 'root'
hostname = '192.168.110.131'
database_name = 'test'

from urllib.parse import quote

# 使用 quote 函数对密码进行编码
password = "snowball950123"

# 现在使用编码后的密码构建连接字符串
SQLALCHEMY_DATABASE_URI = f"mysql+asyncmy://{username}:{password}@{hostname}/{database_name}?charset=utf8mb4"



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

class TestTable(Base):
    __tablename__ = 'test_table'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)


async def create_tables(engine):
    async with engine.begin() as conn:
        # 使用 Base.metadata.create_all 创建所有继承 Base 的模型对应的表
        await conn.run_sync(Base.metadata.create_all)

if __name__ == "__main__":

    asyncio.run(create_tables(async_engine))

