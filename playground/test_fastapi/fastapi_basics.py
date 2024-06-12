#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024-05-30 14:04

@author: @木羽Cheney
@description: 测试 FastAPI 的基本功能。
"""


# 测试 Get 方法

# Union 是 Python 的 typing 模块中的一个类型提示工具，用于指示变量可以是多种类型之一
from typing import Union
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
# Union[str, None] 表示参数 q 可以是 str 类型或者 None 类型。
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


def document():
    return RedirectResponse(url="/docs")


if __name__ == '__main__':
    # 第一种启动方式：
    # 如需启动，在终端 输入命令 ：
    #    fastapi dev fastapi_basics.py  （命令 fastapi dev 读取您的 main.py 文件，检测其中的 FastAPI 应用程序，并使用 Uvicorn 启动服务器。）
    # - http://127.0.0.1:8000/docs
    
    # 第二种启动方式：
    # pip install uvicorn
    import uvicorn
    uvicorn.run(app, host='192.168.110.131', port=8000)
    