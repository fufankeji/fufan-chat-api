# pip install zhipuai

from zhipuai import ZhipuAI

client = ZhipuAI(api_key="")  # 请填写您自己的APIKey



if __name__ == '__main__':

    response = client.chat.completions.create(
        model="glm-4",  # 填写需要调用的模型名称
        messages=[
            {"role": "user", "content": "你好！你叫什么名字"},
        ],
        stream=False,
    )

    # 流式输出, 需要把stream 设置为True
    # for chunk in response:
    #     print(chunk.choices[0].delta)

    # 直接输出结果
    print(response.choices[0].message)