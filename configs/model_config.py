import os

MODEL_ROOT_PATH = ""

TEMPERATURE = 1.0
MAX_TOKENS = 4096
# 默认让大模型采用流式输出
STREAM = True

# 默认启动的模型，如果使用的是glm3-6b，请替换模型名称
# LLM_MODELS = ["glm4-9b-chat", "zhipu-api"]
LLM_MODELS = ["chatglm3-6b", "zhipu-api"]

RERANKER_MODEL = "bge-reranker-large"
RERANKER_MAX_LENGTH = 1024
# 是否启用reranker模型
USE_RERANKER = True

# 知识库匹配向量数量
VECTOR_SEARCH_TOP_K = 5
# 知识库匹配的距离阈值，一般取值范围在0-1之间，SCORE越小，距离越小从而相关度越高。
SCORE_THRESHOLD = 1.0
# 如果使用ReRank模型
RERANKER_TOP_K = 3

# 搜索引擎匹配结题数量
SEARCH_ENGINE_TOP_K = 3

MODEL_PATH = {
    # 这里定义 本机服务器上存储的大模型权重存储路径
    "local_model": {
        # 默认使用glm4-9b-chat
        "glm4-9b-chat": "/home/00_rag/model/ZhipuAI/chatglm4-9b-chat",

        "chatglm3-6b": "/home/00_rag/model/ZhipuAI/chatglm3-6b/",

        # 可扩展其他的开源大模型

    },

    # 这里定义 本机服务器上存储的Embedding模型权重存储路径
    "embed_model": {
        "bge-large-zh-v1.5": "/home/00_rag/model/AI-ModelScope/bge-large-zh-v1___5",

        "m3e-base": "/home/00_rag/model/m3e-base",
        # 可扩展其他的Embedding模型
    },

    "reranker": {
        "bge-reranker-large": "/home/00_rag/model/Xorbits/bge-reranker-large",

    }
}

ONLINE_LLM_MODEL = {

    # 智谱清言的在线API服务
    "zhipu-api": {
        "api_key": "8e1482448c5924ea98bc483b4846009d.bViRUeDbZShksjPO",
        "version": "glm-4",
        "provider": "ChatGLMWorker",
    },

    # OpenAI GPT模型的在线服务
    "openai-api": {
        "model_name": "gpt-4",
        "api_base_url": "https://api.openai.com/v1",
        "api_key": "",
        "openai_proxy": "",
    },

    # 可扩展其他的模型在线模型

}

SUPPORT_AGENT_MODEL = [
    "openai-api",  # GPT4 模型
    "qwen-api",  # Qwen Max模型
    "zhipu-api",  # 智谱AI GLM4模型
    "Qwen",  # 所有Qwen系列本地模型
    "chatglm3-6b",
    "internlm2-chat-20b",
    "Orion-14B-Chat-Plugin",
]

# 选用的 Embedding 名称
EMBEDDING_MODEL = "bge-large-zh-v1.5"

# Embedding 模型运行设备。设为 "auto" 会自动检测(会有警告)，也可手动设定为 "cuda","mps","cpu","xpu" 其中之一。
EMBEDDING_DEVICE = "auto"

# 搜索引擎匹配结题数量
SEARCH_ENGINE_TOP_K = 3

# 原始网页搜索结果筛选后保留的有效数量
SEARCH_RERANK_TOP_K = 3

# 历史对话窗口长度
HISTORY_LEN = 5

URL = "https://google.serper.dev/search"
SERPER_API_KEY = "c4a95d2be3eb2337a373a308c8403f3dc82a1aee"  # 这里替换为自己实际的Serper API Key
