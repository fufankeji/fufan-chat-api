from sqlalchemy.orm import declarative_base, Mapped, mapped_column, relationship
from sqlalchemy.orm import sessionmaker

from typing import List
from typing import Optional
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

"""
官方文档：https://docs.sqlalchemy.org/en/20/dialects/mysql.html
"""

# 用于创建一个基类，该基类将为后续定义的所有模型类提供 SQLAlchemy ORM 功能的基础。
Base = declarative_base()


class UserInfo(Base):
    __tablename__ = "user_account"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(30))
    fullname: Mapped[Optional[str]] = mapped_column(String(255))
    addresses: Mapped[List["Address"]] = relationship(
        "Address", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"UserInfo(id={self.id!r}, name={self.name!r}, fullname={self.fullname!r})"


class Address(Base):
    __tablename__ = "address"
    id: Mapped[int] = mapped_column(primary_key=True)
    email_address: Mapped[str] = mapped_column(String(30))
    user_id: Mapped[int] = mapped_column(ForeignKey("user_account.id"))
    user: Mapped["UserInfo"] = relationship("UserInfo", back_populates="addresses")

    def __repr__(self) -> str:
        return f"Address(id={self.id!r}, email_address={self.email_address!r})"



# 更新以下字段为你本地数据库的实际用户名、密码和数据库名
username = 'root'
hostname = '192.168.110.131'
database_name = 'test'

from urllib.parse import quote

password = "snowball950123"

SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{username}:{password}@{hostname}/{database_name}?charset=utf8mb4"

from sqlalchemy import create_engine

engine = create_engine(
    SQLALCHEMY_DATABASE_URI,
    echo=True
)


# 创建所有表
Base.metadata.create_all(engine)

# 会话和操作
SessionLocal = sessionmaker(bind=engine)


def main():
    with SessionLocal() as session:
        spongebob = UserInfo(
            name="spongebob",
            fullname="Spongebob Squarepants",
            addresses=[Address(email_address="spongebob@sqlalchemy.org")]
        )
        sandy = UserInfo(
            name="sandy",
            fullname="Sandy Cheeks",
            addresses=[
                Address(email_address="sandy@sqlalchemy.org"),
                Address(email_address="sandy@squirrelpower.org")
            ]
        )
        patrick = UserInfo(name="patrick", fullname="Patrick Star")
        session.add_all([spongebob, sandy, patrick])
        session.commit()

if __name__ == '__main__':
    main()