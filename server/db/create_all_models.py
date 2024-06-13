import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from configs import SQLALCHEMY_DATABASE_URI
from server.db.base import Base
import server.db.models  # 确保模型被导入以便创建表

from server.db.base import async_engine, AsyncSessionLocal



async def create_tables(engine):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

if __name__ == "__main__":
    asyncio.run(create_tables(async_engine))
