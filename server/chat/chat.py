#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-05-30 16:04

@author: @木羽Cheney
@description: 定义对话函数
"""

from fastapi import Body, HTTPException
from typing import List, Union, Optional

# 使用LangChain调用ChatGLM3-6B的依赖包
from langchain.chains.llm import LLMChain
from langchain_community.llms.chatglm3 import ChatGLM3
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate


def chat(query: str = Body("", description="用户输入"),
               history: Union[int, List] = Body([],description="历史对话信息",),
               stream: bool = Body(False, description="流式输出"),
               temperature: float = Body(0.8, description="LLM 采样温度", ge=0.0, le=2.0),
               max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
               ):
    """
    # 大模型对话交互接口的执行逻辑
    :param query: 用户输入的问题
    :param conversation_id: 对话框的id，用来标识当前会话记录
    :param history: 对话的历史信息
    :param stream: 是否在前端采用流式输出
    :param model_name: 使用哪个大模型作为后端服务
    :param temperature: 采样温度
    :param max_tokens: 最大输入Token限制
    :return:
    """


    # 使用LangChain调用ChatGLM3-6B服务
    try:
        # 使用LangChain调用ChatGLM3-6B服务
        template = """{query}"""
        prompt = PromptTemplate.from_template(template)

        endpoint_url = "http://192.168.110.131:9091/v1/chat/completions"

        llm = ChatGLM3(
            endpoint_url=endpoint_url,
            history=history,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
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

