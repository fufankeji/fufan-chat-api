# fufan-chat-api
fufan-chat项目的后端服务，负责处理业务逻辑、数据存储和API接口的提供。采用稳健的后端技术，确保服务的稳定性和可扩展性。

## 版本：v1.0

## 业务流程

1. 本地RAG知识问答开发逻辑
![1](https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240704144032483.png)

2. 向量数据库集成逻辑
![2](https://muyu001.oss-cn-beijing.aliyuncs.com/img/%E5%90%91%E9%87%8F%E6%95%B0%E6%8D%AE%E5%BA%93%E9%9B%86%E6%88%90%E9%80%BB%E8%BE%91.png)

3. 通用领域知识问答开发逻辑
![3](https://muyu001.oss-cn-beijing.aliyuncs.com/img/3.png)

## 介绍

实现问答系统的系统架构搭建，目前支持：

架构方面：
1. 用户权限管理
2. 灵活接入在线开源大模型
3. 灵活接入在线API模型
4. 接入Mysql数据库
5. 接入Faiss向量数据库

功能方面：

1. 实现通用领域问答业务流
2. 历史对话信息支持从mysql数据库中实时查询

修复：
1. GLM-4 API 无法实现流式输出
2. LangChain Memory的异步加载

## 安装

### 前提条件

- Python>=3.10
- Mysql

### 安装步骤

1. 克隆仓库并安装依赖：
    ```bash
    git clone --branch v3.0.0 https://github.com/fufankeji/fufan-chat-api.git
    cd fufan-chat-api
    pip install -r requirements.txt
    ```
2. 本地部署Mysql服务并启动
3. 初始化关系型数据库表
    ```bash
    python /fufan-chat-api/server/dbinit_models.py
    ```

4. 启动后端服务：
    ```bash
    python startup.py
    ```

## 使用示例

使用 Postman 或其他 HTTP 客户端工具访问 API 接口：

### POST 请求示例

```http
http://192.168.110.131:8000/api/chat/knowledge_base_chat

{
    "query":"什么是GLM4 多角色对话",
    "user_id":"admin",
    "conversation_id": "df221b2f-ea52-4200-82f5-fcfc011e6786", 
    "conversation_name":"新对话",
    "knowledge_base_name":"private",
    "top_k":"3",
    "score_threshold":"0.5",
    "history":[],
    "history_len": 3,
    "stream": false,
    "model_name":"chatglm3-6b",
    "prompt_name":"default"
}
```