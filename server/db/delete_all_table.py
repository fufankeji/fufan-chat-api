import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from configs import SQLALCHEMY_DATABASE_URI
from server.db.base import Base

async_engine = create_async_engine(
    SQLALCHEMY_DATABASE_URI,
    echo=True,
)

async def drop_tables(engine):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

if __name__ == "__main__":
    asyncio.run(drop_tables(async_engine))