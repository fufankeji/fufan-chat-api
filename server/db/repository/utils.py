from fastapi import Body
from configs import logger, log_verbose, LLM_MODELS, HTTPX_DEFAULT_TIMEOUT
from server.utils import (BaseResponse, fschat_controller_address, list_config_llm_models,
                          get_httpx_client, get_model_worker_config)
from typing import List, Dict


def list_running_models(
        controller_address: str = Body(None, description="Fastchat controller服务器地址",
                                       examples=[fschat_controller_address()]),
):
    '''
    从fastchat controller获取已加载模型列表
    '''
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(controller_address + "/list_models")
            models = r.json()["models"]

            return {"status": 200, "msg": "success", "data": {"models": models}}
            # return BaseResponse(data={"models": models})  # 直接返回模型列表
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(
            code=500,
            data={},
            msg=f"Failed to get available models from controller: {controller_address}. Error: {e}")
