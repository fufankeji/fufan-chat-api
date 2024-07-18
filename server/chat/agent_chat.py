import json
import asyncio

from fastapi import Body
from sse_starlette.sse import EventSourceResponse
from configs import LLM_MODELS, TEMPERATURE

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import LLMSingleActionAgent, AgentExecutor
from typing import AsyncIterable, Optional, List
from configs import LLM_MODELS, SEARCH_ENGINE_TOP_K, TEMPERATURE, MAX_TOKENS, STREAM, HISTORY_LEN
from server.utils import wrap_done, get_ChatOpenAI, get_prompt_template
from server.knowledge_base.kb_service.base import get_kb_details
from server.agent.custom_agent.ChatGLM3Agent import initialize_glm3_agent
from server.agent.tools_select import tools, tool_names
from server.agent.callbacks import CustomAsyncIteratorCallbackHandler, Status
from server.chat.utils import History
from server.agent import model_container
from server.agent.custom_template import CustomOutputParser, CustomPromptTemplate
from server.db.repository.message_repository import add_message_to_db
from server.callback_handler.conversation_callback_handler import ConversationCallbackHandler


async def agent_chat(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
                     conversation_id: str = Body("", description="对话框ID"),
                     model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                     prompt_name: str = Body("agent_chat",
                                             description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                     ):
    async def agent_chat_iterator(
            query: str,
            model_name: str = LLM_MODELS[0],
            prompt_name: str = prompt_name,
    ) -> AsyncIterable[str]:

        # 这里使用自定义的回调函数
        callback = CustomAsyncIteratorCallbackHandler()
        callbacks = [callback]

        # 构造一个新的Message_ID记录
        message_id = await add_message_to_db(
            conversation_id=conversation_id,
            prompt_name=prompt_name,
            query=query
        )

        conversation_callback = ConversationCallbackHandler(query=query,
                                                            conversation_id=conversation_id,
                                                            message_id=message_id,
                                                            chat_type=prompt_name,
                                                            )
        callbacks.append(conversation_callback)

        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            callbacks=callbacks,
        )

        model_container.MODEL = model

        # 获取 AI Agents的提示模板
        prompt_template = get_prompt_template(prompt_name, "ChatGLM3")

        # langChain Docs:https://python.langchain.com/v0.1/docs/modules/memory/types/buffer_window/
        # 保存一段时间内对话交互的列表。它仅使用最后K个交互。这对于保持最近交互的滑动窗口非常有用，因此缓冲区不会变得太大。
        memory = ConversationBufferWindowMemory(k=HISTORY_LEN)

        agent_executor = initialize_glm3_agent(
            llm=model,
            tools=tools,
            callback_manager=None,
            prompt=prompt_template,
            input_variables=["input", "intermediate_steps", "history"],
            memory=memory,
            verbose=True,
        )

        while True:
            try:
                task = asyncio.create_task(wrap_done(
                    agent_executor.acall(query, callbacks=[callback], include_run_info=True),
                    callback.done))
                break
            except:
                pass

        if STREAM:
            async for chunk in callback.aiter():
                tools_use = []
                # Use server-sent-events to stream the response
                data = json.loads(chunk)
                if data["status"] == Status.start or data["status"] == Status.complete:
                    continue
                elif data["status"] == Status.error:
                    tools_use.append("\n```\n")
                    tools_use.append("工具名称: " + data["tool_name"])
                    tools_use.append("工具状态: " + "调用失败")
                    tools_use.append("错误信息: " + data["error"])
                    tools_use.append("重新开始尝试")
                    tools_use.append("\n```\n")
                    yield json.dumps({"tools": tools_use}, ensure_ascii=False)
                elif data["status"] == Status.tool_finish:
                    tools_use.append("\n```\n")
                    tools_use.append("工具名称: " + data["tool_name"])
                    tools_use.append("工具状态: " + "调用成功")
                    tools_use.append("工具输入: " + data["input_str"])
                    tools_use.append("工具输出: " + data["output_str"])
                    tools_use.append("\n```\n")
                    yield json.dumps({"tools": tools_use}, ensure_ascii=False)
                elif data["status"] == Status.agent_finish:
                    yield json.dumps({"final_answer": data["final_answer"]}, ensure_ascii=False)
                else:
                    yield json.dumps({"answer": data["llm_token"]}, ensure_ascii=False)


        else:
            answer = ""
            final_answer = ""
            async for chunk in callback.aiter():
                data = json.loads(chunk)
                if data["status"] == Status.start or data["status"] == Status.complete:
                    continue
                if data["status"] == Status.error:
                    answer += "\n```\n"
                    answer += "工具名称: " + data["tool_name"] + "\n"
                    answer += "工具状态: " + "调用失败" + "\n"
                    answer += "错误信息: " + data["error"] + "\n"
                    answer += "\n```\n"
                if data["status"] == Status.tool_finish:
                    answer += "\n```\n"
                    answer += "工具名称: " + data["tool_name"] + "\n"
                    answer += "工具状态: " + "调用成功" + "\n"
                    answer += "工具输入: " + data["input_str"] + "\n"
                    answer += "工具输出: " + data["output_str"] + "\n"
                    answer += "\n```\n"
                if data["status"] == Status.agent_finish:
                    final_answer = data["final_answer"]
                else:
                    answer += data["llm_token"]

            yield json.dumps({"answer": answer, "final_answer": final_answer}, ensure_ascii=False)
        await task

    return EventSourceResponse(agent_chat_iterator(query=query,
                                                   model_name=model_name,
                                                   prompt_name=prompt_name),
                               )
