import sys
import uuid
# 启动FastChat的Controller
from fastchat.serve.controller import app, Controller
import uvicorn
"""
代码启动FastChat controller , ModelWorker和openai_api_server三个服务
测试本地开源大模型使用借助FastChat 托管 提供 OpenAI API 服务
"""

def start_main_server():
    # 1. 创建controller
    controller = Controller(dispatch_method="shortest_queue")
    sys.modules["fastchat.serve.controller"].controller = controller
    app.title = "FastChat Controller"
    app._controller = controller
    uvicorn.run(app, host="192.168.110.131", port=20001)

    # 2. 创建Model_Worker
    worker_id = str(uuid.uuid4())[:8]
    # 启动本地模型
    from fastchat.serve.model_worker import app, ModelWorker
    worker = ModelWorker(
        controller_addr="http://192.168.110.131:21001",
        worker_addr="http://192.168.110.131:20002",
        worker_id=worker_id,
        limit_worker_concurrency=5,
        no_register=False,
        model_path="/home/00_rag/model/ZhipuAI/chatglm3-6b",
        num_gpus=4,
        model_names="chatglm3-6b",
        device="cuda",
        max_gpu_memory="22GiB",
    )

    app.title = f"FastChat LLM Server ChaGLM3-6b"
    app._worker = worker

    uvicorn.run(app, host="192.168.110.131", port=20002)

    # 3. 创建 OpenAI API Server
    from fastchat.serve.openai_api_server import app, app_settings, CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app_settings.controller_address = "http://192.168.110.131:21001"
    app_settings.api_keys = []

    app.title = "FastChat OpeanAI API Server"

    uvicorn.run(app, host="192.168.110.131", port=8000)


if __name__ == '__main__':
    start_main_server()