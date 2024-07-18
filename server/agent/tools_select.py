from langchain.tools import Tool
from server.agent.tools import *

tools = [
    Tool.from_function(
        func=duckgosearch,
        name="DuckDuckGoSearch",
        description="useful for when you need to search the internet for information",
        args_schema=DuckGoSearchInput,
    ),

]

tool_names = [tool.name for tool in tools]