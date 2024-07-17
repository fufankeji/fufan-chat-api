# fufan-chat-api

后端服务为 `fufan-chat-api` 项目提供支持，前端服务为`fufan-chat-web` vue框架提供交互式应用页面。整体项目专注于处理复杂的业务逻辑、数据存储和 API 接口提供。

## 介绍

`fufan-chat-api` 设计为一个高度模块化和可扩展的问答系统架构。该系统具备以下核心架构和功能特性：

### 架构特点：
- **用户权限管理**：实现细粒度的用户访问控制，保障数据安全与隐私。
- **模型接入灵活性**：支持接入各类在线开源大模型及 API 模型，确保系统的适应性和前瞻性。
- **数据库整合**：集成 Mysql 关系型数据库和向量数据库，优化数据存取效率和查询响应时间。（目前支持Faiss和Milvus）

### 功能亮点：

#### 一、用户模块
- 提供完善的用户注册和登录机制，支持用户身份验证与会话管理。

#### 二、模型接入
- 接入并优化 `glm4-9b-chat` 模型，包括：
  - **流式输出修复**：确保连续对话的流畅性和稳定性。
  - **自问自答循环问题修复**：解决了模型在对话中陷入自循环的技术难题。

#### 三、核心问答功能
##### 3.1 通用问答流程
- 支持多轮对话与会话历史记忆，增强用户体验，提升对话的上下文关联性。

##### 3.2 基于 RAG 的问答
- 实现基于 RAG 技术的问答系统，具备以下特点：
  - **多轮对话支持**：允许在多个回合内保持对话连贯性。
  - **历史记忆功能**：通过历史会话记录增强对话的个性化和相关性。
  - **系统提示角色**：增添系统提示角色以引导用户对话，提供更为人性化的交互体验。
  - **实时 Faiss 向量数据检索召回**：利用 Faiss 向量数据库进行快速高效的数据检索，优化答案的精准度。

#### 修复项
- **GLM-4 API 流式输出**：针对 GLM-4 API 流式输出功能的缺陷进行了修复，增强了模型的实时交互能力。
- **LangChain Memory 异步加载**：优化了内存管理，支持异步数据处理，提升了系统的整体性能。

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
   
## API接口示例

![1](https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240717192132838.png)

## 使用示例

使用 Postman 或其他 HTTP 客户端工具访问 API 接口：

### POST 请求示例

```http
http://192.168.110.131:8000/api/chat  # 替换为自己实际启动的服务 IP + 端口

{
    "query":"什么是机器学习",
    "conversation_id":"18b352a0-42de-419c-ada1-a0fa44dbee1d",
    "model_name":"chatglm3-6b"
}
```