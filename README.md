# fufan-chat-api
<<<<<<< HEAD
fufan-chat项目的后端服务，负责处理业务逻辑、数据存储和API接口的提供。采用稳健的后端技术，确保服务的稳定性和可扩展性。

## 版本：v1.0

## 业务流程

![01](https://muyu001.oss-cn-beijing.aliyuncs.com/img/01.png)


## 介绍

基于 LangChain 加载 ChatGLM3-6b，通过 FastAPI 构建 ResfFul API，实现与本地开源大模型 ChatGLM3-6B 模型的对话交互。

## 安装

### 前提条件

- Python 3.11

### 安装步骤

1. 私有化启动 ChatGLM3-6B。
2. 克隆仓库并安装依赖：
    ```bash
    cd fufan-chat-api
    pip install -r requirements.txt
    ```
3. 启动后端服务：
    ```bash
    python server/api_router.py
    ```

## 使用示例

使用 Postman 或其他 HTTP 客户端工具访问 API 接口：

### POST 请求示例

```http
POST http://192.168.110.131:8000/api/chat
Content-Type: application/json
{
    "query": "你好，请你详细的向我介绍一下什么是机器学习？",
    "history": [],
    "stream": false,
    "temperature": 0.8,
    "max_tokens": 2048
}
```
=======
基于RAG的私有知识库问答系统
>>>>>>> f90d81addd27be9355239a43bf4662d4621f3963
