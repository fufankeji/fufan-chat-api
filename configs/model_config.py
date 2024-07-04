MODEL_ROOT_PATH = ""

TEMPERATURE = 0.8

# LLM_MODELS = ["zhipu-api"]

LLM_MODELS = ["chatglm3-6b", "zhipu-api"]
# LLM_MODELS = ["chatglm3-6b"]

RERANKER_MODEL = "bge-reranker-large"
RERANKER_MAX_LENGTH = 1024
# 是否启用reranker模型
USE_RERANKER = True

Agent_MODEL = None
HISTORY_LEN = 3
# 知识库匹配向量数量
VECTOR_SEARCH_TOP_K = 3
# 知识库匹配的距离阈值，一般取值范围在0-1之间，SCORE越小，距离越小从而相关度越高。
# 但有用户报告遇到过匹配分值超过1的情况，为了兼容性默认设为1，在WEBUI中调整范围为0-2
SCORE_THRESHOLD = 1.0
# 搜索引擎匹配结题数量
SEARCH_ENGINE_TOP_K = 3

MODEL_PATH = {
    # 这里定义 本机服务器上存储的大模型权重存储路径
    "local_model": {
        "chatglm3-6b": "/home/00_rag/model/ZhipuAI/chatglm3-6b",

        # 可扩展其他的开源大模型

    },

    # 这里定义 本机服务器上存储的Embedding模型权重存储路径
    "embed_model": {
        "bge-large-zh-v1.5": "/home/00_rag/model/AI-ModelScope/bge-large-zh-v1___5",

        # 可扩展其他的Embedding模型
    },

    "reranker": {
        "bge-reranker-large": "/home/00_rag/model/Xorbits/bge-reranker-large",
    }
}

ONLINE_LLM_MODEL = {

    # 智谱清言的在线API服务
    "zhipu-api": {
        "api_key": "",
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

URL = "https://google.serper.dev/search"
SERPER_API_KEY="c8da0155b57b159becfdbf4767aef63306ae4619"  # 这里替换为自己实际的Serper API Key

ZILLIZ_URI = "https://in03-053b56bef5db18e.api.gcp-us-west1.zillizcloud.com"
ZILLIZ_TOKEN = ""
ZILLIZ_DIM = 1024
ZILLIZ_COLLECTION = "muyu"
