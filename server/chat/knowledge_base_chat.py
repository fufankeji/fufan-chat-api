#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import Body, Request
from configs import (LLM_MODELS,
                     VECTOR_SEARCH_TOP_K,
                     SCORE_THRESHOLD,)

from server.knowledge_base.kb_service.base import KBServiceFactory


async def knowledge_base_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                              knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
                              # history: List[History] = Body(
                              #     [],
                              #     description="历史对话",
                              #     examples=[[
                              #         {"role": "user",
                              #          "content": "我们来玩成语接龙，我先来，生龙活虎"},
                              #         {"role": "assistant",
                              #          "content": "虎头虎脑"}]]
                              # ),
                              # stream: bool = Body(False, description="流式输出"),
                              # model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                              # temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                              # max_tokens: Optional[int] = Body(
                              #     None,
                              #     description="限制LLM生成Token数量，默认None代表模型最大值"
                              # ),
                              # prompt_name: str = Body(
                              #     "default",
                              #     description="使用的prompt模板名称(在configs/prompt_config.py中配置)"
                              # ),
                              # request: Request = None,
                              ):

    # 从数据库中读取知识库的相关配置
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    print(kb)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    # history = [History.from_data(h) for h in history]
    pass
