#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# FastAPI教程地址：https://fastapi.tiangolo.com/tutorial/first-steps/

from typing import Union
from fastapi import FastAPI
from chat.chat import chat
import uvicorn

app = FastAPI(
    description="FuFan Chat Web API Server"
)

# 使用路由函数的方式定义，在动态添加路由或在运行时修改路由配置时更为灵活。
app.post("/api/chat",
         tags=["Chat"],
         summary="大模型对话交互接口",
         )(chat)

if __name__ == '__main__':

    uvicorn.run(app, host='192.168.110.131', port=8000)