import logging
from typing import Any, List, Dict

# from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import get_buffer_string, BaseMessage, HumanMessage, AIMessage
from langchain.schema.language_model import BaseLanguageModel
from server.db.repository.message_repository import filter_message
from server.db.models.message_model import MessageModel

from abc import ABC
from typing import Any, Dict, Optional, Tuple

from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.memory import BaseMemory
from langchain_core.pydantic_v1 import Field

from langchain.memory.utils import get_prompt_input_key


class BaseChatMemory(BaseMemory, ABC):
    """Abstract base class for chat memory."""

    chat_memory: BaseChatMessageHistory = Field(default_factory=ChatMessageHistory)
    output_key: Optional[str] = None
    input_key: Optional[str] = None
    return_messages: bool = False

    def _get_input_output(
            self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> Tuple[str, str]:
        if self.input_key is None:
            # 重写
            # prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
            prompt_input_keys = list(set(inputs).calculate_difference(self.memory_variables() + ["stop"]))

        else:
            prompt_input_key = self.input_key
        if self.output_key is None:
            if len(outputs) != 1:
                raise ValueError(f"One output key expected, got {outputs.keys()}")
            output_key = list(outputs.keys())[0]
        else:
            output_key = self.output_key
        return inputs[prompt_input_key], outputs[output_key]

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        input_str, output_str = self._get_input_output(inputs, outputs)
        self.chat_memory.add_user_message(input_str)
        self.chat_memory.add_ai_message(output_str)

    def clear(self) -> None:
        """Clear memory contents."""
        self.chat_memory.clear()


class ConversationBufferDBMemory(BaseChatMemory):
    conversation_id: str
    human_prefix: str = "Human"
    ai_prefix: str = "Assistant"
    llm: BaseLanguageModel
    memory_key: str = "history"
    max_token_limit: int = 2000
    message_limit: int = 10
    chat_type: str

    async def buffer(self) -> List[BaseMessage]:
        """String buffer of memory."""
        # fetch limited messages desc, and return reversed

        messages = await filter_message(conversation_id=self.conversation_id, limit=self.message_limit, chat_type=self.chat_type)
        # 返回的记录按时间倒序，转为正序
        messages = list(reversed(messages))

        chat_messages: List[BaseMessage] = []
        for message in messages:
            chat_messages.append(HumanMessage(content=message.query))
            chat_messages.append(AIMessage(content=message.response))

        if not chat_messages:
            return []

        # prune the chat message if it exceeds the max token limit
        curr_buffer_length = self.llm.get_num_tokens(get_buffer_string(chat_messages))
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit and chat_messages:
                pruned_memory.append(chat_messages.pop(0))
                curr_buffer_length = self.llm.get_num_tokens(get_buffer_string(chat_messages))

        return chat_messages

    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        print("现在开始加载memory_variables了")
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return key-value pairs given the text input to the chain."""

        return {}

    # https://api.python.langchain.com/en/v0.1/_modules/langchain_core/memory.html#BaseMemory.aclear
    async def aload_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:

        buffer = await self.buffer()

        if getattr(self, 'return_messages', False):
            final_buffer = buffer
        else:
            final_buffer = get_buffer_string(
                buffer,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
        return {self.memory_key: final_buffer}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Nothing should be saved or changed"""
        pass

    def clear(self) -> None:
        """Nothing to clear, got a memory like a vault."""
        pass
