import sys

# LLM 模型运行设备。设为"auto"会自动检测
LLM_DEVICE = "cuda"

# httpx 请求默认超时时间（秒）。如果加载模型或对话较慢，出现超时错误，可以适当加大该值。
HTTPX_DEFAULT_TIMEOUT = 300.0

# 各服务器默认绑定host。如改为"0.0.0.0"需要修改下方所有XX_SERVER的host
DEFAULT_BIND_HOST = "0.0.0.0" if sys.platform != "win32" else "127.0.0.1"

# api.py server
API_SERVER = {
    "host": DEFAULT_BIND_HOST,
    "port": 8000,
}

FSCHAT_MODEL_WORKERS = {
    # 默认的Model Worker 启动配置
    "default": {
        "host": DEFAULT_BIND_HOST,
        "port": 20002,
        "device": LLM_DEVICE,
    },

    # 本地模型 需要配置启动的设备，cpu还是gpu
    "glm4-9b-chat": {
        "device": "cuda",
    },

    "chatglm3-6b": {
        "device": "cuda",
    },

    # 在线模型只需要配置Model Worker的端口即可
    "zhipu-api": {
        "port": 21001,
    },
}

FSCHAT_CONTROLLER = {
    "host": DEFAULT_BIND_HOST,
    "port": 20001,
    "dispatch_method": "shortest_queue",
}

FSCHAT_OPENAI_API = {
    "host": DEFAULT_BIND_HOST,
    "port": 20000,
}
