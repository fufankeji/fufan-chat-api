# LangChain 的 ArxivQueryRun 工具
from pydantic import BaseModel, Field
from langchain_community.utilities import ArxivAPIWrapper

"""
langChain Docs:https://python.langchain.com/v0.2/docs/integrations/tools/arxiv/
"""

from pydantic import BaseModel, Field
from langchain_community.tools import DuckDuckGoSearchRun


def duckgosearch(query: str):
    search = DuckDuckGoSearchRun()
    return search.run(query)


class DuckGoSearchInput(BaseModel):
    query: str = Field(description="The search query title")


if __name__ == "__main__":
    docs = duckgosearch("Obama's first name?")
    print(docs)