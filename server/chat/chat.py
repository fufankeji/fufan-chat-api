#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from fastapi import Body, HTTPException
from typing import List, Union, Optional

# 使用LangChain调用ChatGLM3-6B的依赖包
from langchain.chains.llm import LLMChain
from langchain_community.llms.chatglm3 import ChatGLM3
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate

# 日志包
from loguru import logger


def chat(query: str = Body("", description="用户的输入"),
         # model_name: str = Body("chatglm3-6b", description="基座模型的名称"),
         model_name: str = Body("glm4-9b-chat", description="基座模型的名称"),
         temperature: float = Body(0.8, description="大模型参数：采样温度", ge=0.0, le=2.0),
         max_tokens: Optional[int] = Body(None, description="大模型参数：最大输入Token限制"),
         ):
    """
    :param query: 用户输入的问题
    :param model_name: 使用哪个大模型作为后端服务
    :param temperature: 采样温度
    :param max_tokens: 最大输入Token限制
    :return:  大模型的回复结果
    """

    logger.info("Received query: {}", query)
    logger.info("Model name: {}", model_name)
    logger.info("Temperature: {}", temperature)
    logger.info("Max tokens: {}", max_tokens)

    # 使用LangChain调用glm4-9b-chat 或者 ChatGLM3-6B服务
    try:
        # 使用LangChain调用ChatGLM3-6B服务
        template = """{query}"""
        prompt = PromptTemplate.from_template(template)

        endpoint_url = "http://192.168.110.131:9091/v1/chat/completions"

        llm = ChatGLM3(
            endpoint_url=endpoint_url,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        llm_chain = prompt | llm
        response = llm_chain.invoke(query)

        if response is None:
            raise ValueError("Received null response from LLM")

        return {"LLM Response": response}

    except ValueError as ve:
        # 捕获值错误并返回400响应
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # 捕获所有其他异常并返回500响应
        raise HTTPException(status_code=500, detail="Internal Server Error: " + str(e))
