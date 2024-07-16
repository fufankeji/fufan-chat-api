# fufan-chat-api
fufan-chat项目的后端服务，负责处理业务逻辑、数据存储和API接口的提供。采用稳健的后端技术，确保服务的稳定性和可扩展性。

## 版本：v4.0

## 业务流程

1. 实时联网 + RAG 检索开发逻辑
![6](https://muyu001.oss-cn-beijing.aliyuncs.com/img/%E8%81%94%E7%BD%91%E6%A3%80%E7%B4%A2.png)


## 介绍

实现问答系统的系统架构搭建，目前支持：

架构方面：
1. 用户权限管理
2. 灵活接入在线开源大模型
3. 灵活接入在线API模型
4. 接入Mysql数据库
5. 接入向量数据库
6. 接入SerperAPI做联网实时检索

功能方面：

1. 实现通用领域问答业务流程
2. 历史对话信息支持从mysql数据库中实时查询
3. 实现本地RAG检索问答业务流程
4. 实现实时联网 + 检索问答业务流程

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
    git clone --branch v4.0.0 https://github.com/fufankeji/fufan-chat-api.git
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
http://192.168.110.131:8000/api/chat/search_engine_chat  # 替换为自己实际启动的服务 IP + 端口

{
    "query":"保罗乔治加盟了哪一支NBA球队？",
    "search_top_k":3,
    "model_name":"chatglm3-6b",
    "prompt_name":"default",
    "user_id":"admin",
    "conversation_id":"df221b2f-ea52-4200-82f5-fcfc011e6786", 
    "retrival_top_k":3,
    "knowledge_base_name":"test"
}
```