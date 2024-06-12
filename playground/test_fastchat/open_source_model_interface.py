import openai

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


def get_ChatOpenAI() -> ChatOpenAI:
    model = ChatOpenAI(
        openai_api_key="EMPTY",
        openai_api_base="http://localhost:8000/v1/",
        model_name="chatglm3-6b",
    )
    return model


def test_openai_api():
    model = get_ChatOpenAI()
    template = """{question}"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | model

    print(chain.invoke("请你介绍一下你自己"))

if __name__ == '__main__':
    test_openai_api()