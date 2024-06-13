from server.db.session import with_async_session, async_session_scope
from typing import Dict, List
import uuid
from server.db.models.knowledge_base_model import KnowledgeBaseModel


from sqlalchemy.future import select


@with_async_session
async def load_kb_from_db(session, kb_name: str):
    """
    加载知识库的详细信息
    """
    result = await session.execute(
        select(KnowledgeBaseModel)
        .filter(KnowledgeBaseModel.kb_name.ilike(kb_name))
    )
    kb = result.scalars().first()

    if kb:
        kb_name, vs_type, embed_model = kb.kb_name, kb.vs_type, kb.embed_model
    else:
        kb_name, vs_type, embed_model = None, None, None

    return kb_name, vs_type, embed_model
