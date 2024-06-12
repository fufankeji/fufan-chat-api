from sqlalchemy import Column, Integer, String, DateTime, JSON, func, ForeignKey

from sqlalchemy.orm import relationship
from server.db.base import Base


class ConversationModel(Base):
    """
    会话模型，表示用户的一个聊天会话
    """
    __tablename__ = 'conversation'
    id = Column(String(32), primary_key=True, comment='会话ID')
    user_id = Column(String(32), ForeignKey('user.id'), comment='用户ID')

    name = Column(String(50), comment='对话框名称')
    # chat/agent_chat等
    chat_type = Column(String(50), comment='聊天类型')
    create_time = Column(DateTime, default=func.now(), comment='创建时间')

    # 会话与用户的多对一关系
    user = relationship('UserModel', back_populates='conversations')
    # 会话与消息的一对多关系
    messages = relationship('MessageModel', back_populates='conversation')


    def __repr__(self):
        return f"<Conversation(id='{self.id}'>, name='{self.name}', chat_type='{self.chat_type}', create_time='{self.create_time}')>"
