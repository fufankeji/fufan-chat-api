from pydantic import BaseModel, Field
from langchain.prompts.chat import ChatMessagePromptTemplate
from configs import logger
from typing import List, Tuple, Dict, Union
import aiohttp
import os
import hashlib
from configs import SERPER_API_KEY, URL


class History(BaseModel):
    """
    对话历史
    可从dict生成，如
    h = History(**{"role":"user","content":"你好"})
    也可转换为tuple，如
    h.to_msy_tuple = ("human", "你好")
    """
    role: str = Field(...)
    content: str = Field(...)

    def to_msg_tuple(self):
        return "ai" if self.role == "assistant" else "human", self.content

    def to_msg_template(self, is_raw=True) -> ChatMessagePromptTemplate:
        role_maps = {
            "ai": "assistant",
            "human": "user",
        }
        role = role_maps.get(self.role, self.role)
        if is_raw:  # 当前默认历史消息都是没有input_variable的文本。
            content = "{% raw %}" + self.content + "{% endraw %}"
        else:
            content = self.content

        return ChatMessagePromptTemplate.from_template(
            content,
            "jinja2",
            role=role,
        )

    @classmethod
    def from_data(cls, h: Union[List, Tuple, Dict]) -> "History":
        if isinstance(h, (list, tuple)) and len(h) >= 2:
            h = cls(role=h[0], content=h[1])
        elif isinstance(h, dict):
            h = cls(**h)

        return h


async def search(query, num, locale=''):
    params = {
        "q": query,
        "num": num
    }

    if locale:
        params["hl"] = locale

    try:
        # 确保get_search_results是异步函数
        search_results = await get_search_results(params=params)
        return search_results
    except Exception as e:
        print(f"search failed: {e}")
        raise e


async def get_search_results(params):
    try:
        # Serper API 的 URL
        url = URL
        # 从环境变量中获取 API 密钥
        params['api_key'] = SERPER_API_KEY

        # 使用 aiohttp 创建一个异步 HTTP 客户端会话
        async with aiohttp.ClientSession() as session:
            # 发送 GET 请求到 Serper API，并等待响应
            async with session.get(url, params=params) as response:
                # 解析 JSON 响应数据
                data = await response.json()
                # 提取有效的搜索结果
                items = data.get("organic", [])
                results = []
                for item in items:
                    # 为每个搜索结果生成 UUID（MD5 哈希）
                    item["uuid"] = hashlib.md5(item["link"].encode()).hexdigest()
                    # 初始化搜索结果的得分
                    item["score"] = 0.00
                    results.append(item)

        return results
    except Exception as e:
        # 记录错误信息
        print("get search results failed:", e)
        raise e
