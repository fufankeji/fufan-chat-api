from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, func
from sqlalchemy.orm import relationship
from server.db.base import Base

class UserModel(Base):
    __tablename__ = 'user'
    id = Column(String(32), primary_key=True, comment='用户ID')
    username = Column(String(255), unique=True, comment='用户名')
    password_hash = Column(String(255), comment='密码的哈希值')
    # 你可以添加更多用户相关的字段，如邮箱、电话等

    conversations = relationship('ConversationModel', back_populates='user')
    knowledge_bases = relationship('KnowledgeBaseModel', back_populates='user', cascade='all, delete-orphan')  # 更新关系

    def __repr__(self):
        return f"<User(id='{self.id}', username='{self.username}')>"
