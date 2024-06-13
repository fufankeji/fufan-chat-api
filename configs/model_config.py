


TEMPERATURE = 0.8
#
LLM_MODELS = ["zhipu-api"]

# LLM_MODELS = ["chatglm3-6b", "zhipu-api"]

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
}


ONLINE_LLM_MODEL = {

    # 智谱清言的在线API服务
    "zhipu-api": {
        "api_key": "086a38e9141410d76e393ec52105c83b.7vBwRS4srgxpMRXU",
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



# 选用的 Embedding 名称
EMBEDDING_MODEL = "bge-large-zh-v1.5"
