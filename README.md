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

#### 3.3 联网实时检索 + RAG 问答
- **基于 Serper API 的 Google Search 信息检索**：利用 Serper API 构建的搜索能力，可以即便在国内网络环境下，也能高效进行信息检索。此功能的实现流程如下：
  1. **实时搜索**：根据用户提出的查询 (`Query`)，实时调用 Serper API 搜索指定数量的网页信息。
  2. **初步重排**：对搜索结果进行初步重排，筛选出与查询最相关的 Top N 网页信息。
  3. **信息索引**：对这些 Top N 网页内容进行索引处理，并存入 Milvus 向量数据库，以便进行高效的数据检索。
  4. **向量检索**：在 Milvus 向量数据库中，根据用户的查询执行检索操作，获取最相关的信息块（Chunks）。
  5. **回答生成**：结合检索到的信息块构建完整的提示（Prompt），生成并提供用户所需的精确回答。



#### 修复项
- **GLM-4 API 流式输出**：针对 GLM-4 API 流式输出功能的缺陷进行了修复，增强了模型的实时交互能力。
- **LangChain Memory 异步加载**：优化了内存管理，支持异步数据处理，提升了系统的整体性能。
- **Milvus添加索引时报错问题：TypeError: 'NoneType' object is not subscriptable**
  - 官方新版本的BUG：https://github.com/langchain-ai/langchain/issues/24116
  - 解决方法：强制安装 pip install langchain-core==0.2.5, 可暂时忽略版本依赖冲突的警告，等待langchain官方修复BUG

## 安装

### 前提条件

确保以下软件或服务已安装并配置好：

- Python (版本 3.10 或更高)
- Mysql (版本 5.7 或更高)
- Milvus (版本 2.3.7 或更高)

### 安装步骤

1. 克隆仓库并安装依赖：
    ```bash
    git clone https://github.com/fufankeji/fufan-chat-api.git
    cd fufan-chat-api
    pip install -r requirements.txt
    ```
2. 本地部署Mysql服务并启动
   - [**Ubuntu系统上安装Mysql**](/docs/01_Ubuntu系统上安装Mysql.md)
   
3. 初始化关系型数据库表
    ```bash
    python /fufan-chat-api/server/db/create_all_model.py
    ```
4. 初始化Faiss向量数据库
    ```bash
    python /fufan-chat-api/server/knowledge_base/init_vs.py
    ```
5. 本地部署milvus向量数据库并启动（如需使用）
   - [**Ubuntu系统安装部署Milvus向量数据库**](/docs/02_Ubuntu 系统安装部署Milvus向量数据库.md) 
6. 启动后端服务：
    ```bash
    python startup.py
    ```
   
## API接口示例
<div align="center">
<img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240717192132838.png" alt="image-20240713010710534" width="1000"/>
</div>


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
## 技术交流
**fufan_chat_api目前上线了v1.0版本，将持续迭代更新，感兴趣的小伙伴欢迎加入技术交流群。如有其他任何问题，<span style="color:red;">扫码添加小可爱(微信：littlelion_1215)，回复“RAG”详询哦👇</span>**

<div align="center">
<img src="https://ml2022.oss-cn-hangzhou.aliyuncs.com/img/image-20240713010710534.png" alt="image-20240713010710534" width="200"/>
</div>

#### [课程详情页👉](https://appze9inzwc2314.h5.xiaoeknow.com/v1/goods/goods_detail/p_66540df1e4b0694c9816e922?entry=2&entry_type=2002&share_type=5&share_user_id=u_6219ad9e8014c_HOpokbsTov&type=3)：该项目提供详细的源码讲解，可进入课程目录详细了解。
#### [BiliBili公开课视频 @木羽Cheney👉](https://space.bilibili.com/3537113897241540?spm_id_from=333.337.0.0)：实时追踪大模型前言发展与应用。

