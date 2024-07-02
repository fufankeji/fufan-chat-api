from langchain.tools import Tool
from server.agent.tools import *

tools = [
    Tool.from_function(
        func=search_knowledgebase_complex,
        name="search_knowledgebase_complex",
        description="读取本地知识库，获取GLM4 的相关信息",
        args_schema=KnowledgeSearchInput,
    ),

]

tool_names = [tool.name for tool in tools]
