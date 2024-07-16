# fufan-chat-api
fufan-chat项目的后端服务，负责处理业务逻辑、数据存储和API接口的提供。

## 介绍

实现问答系统的系统架构搭建，目前支持：

架构方面：
1. 用户权限管理
2. 灵活接入在线开源大模型
3. 灵活接入在线API模型
4. 接入Mysql数据库
5. 接入向量数据库

功能方面：

# 一、用户模块
1. 用户注册、登录功能

# 二、模型接入
1. 接入glm4-9b-chat
   - 修复流式输出问题
   - 修复glm-4-9b-chat模型自问自答，停不下来的问题

修复：
1. GLM-4 API 无法实现流式输出
2. LangChain Memory的异步加载

## 安装

### 前提条件

- Python>=3.10
- Mysql
- Milvus

### 安装步骤

1. 克隆仓库并安装依赖：
    ```bash
    cd fufan-chat-api
    pip install -r requirements.txt
    ```
2. 本地部署Mysql服务并启动
3. 初始化关系型数据库表
    ```bash
    python /fufan-chat-api/server/db/create_all_model.py
    ```
4. 初始化Faiss向量数据库
    ```bash
    python /fufan-chat-api/server/knowledge_base/init_vs.py
    ```
5. 本地部署milvus向量数据库并启动
6. 启动后端服务：
    ```bash
    python startup.py
    ```
## 使用示例

使用 Postman 或其他 HTTP 客户端工具访问 API 接口：

### POST 请求示例

```http
http://192.168.110.131:8000/api/chat  # 替换为自己实际启动的服务 IP + 端口

{
    "query":"什么是快乐星球",
    "model_name":"zhipu-api",
    "conversation_id":"edcrfv33",
    "conversation_name":"new chat 1",
    "user_id":"admin",
    "prompt_name":"llm_chat",
    "history_len":3,
    "stream":true

    // "history":[]    

}
```