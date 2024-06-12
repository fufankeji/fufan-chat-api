from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


chat = ChatZhipuAI(
    api_key="086a38e9141410d76e393ec52105c83b.7vBwRS4srgxpMRXU",
    model="glm-4",
    temperature=0.5,
)


messages = [
    SystemMessage(content="你是一位诗人"),
    AIMessage(content="你好"),
    HumanMessage(content="写一个关于AI的诗词"),
]


response = chat.invoke(messages)
print(response.content)  # Displays the AI-generated poem