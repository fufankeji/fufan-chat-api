from transformers import AutoTokenizer, AutoConfig


class PromptTemplate:
    """
    用于处理与生成模型输入有关的任务，如格式化系统提示和用户提示，并结合检索结果。
    """

    # 定义模板中的占位符
    placeholders = ['reference', 'question']
    # 定义基本的系统提示和用户提示模板
    base_system_prompt = "Answer the question based on the given document." \
                         "Only give me the answer and do not output any other words." \
                         "\nThe following are given documents.\n\n{reference}"

    base_user_prompt = "Question: {question}"

    # 初始化函数，接收配置参数等
    def __init__(self,
                 config,
                 system_prompt="",
                 user_prompt="",
                 enable_chat=True
                 ):

        self.config = config

        # 检查模型配置，以确定是否为 OpenAI 模型或其他
        self.is_openai = config['framework'] == 'openai'
        if not self.is_openai:
            self.generator_path = config['generator_model_path']
            model_config = AutoConfig.from_pretrained(self.generator_path, trust_remote_code=True)
            model_name = model_config._name_or_path.lower()
            self.is_chat = False
            if 'chat' in model_name or 'instruct' in model_name:
                self.is_chat = True
                self.tokenizer = AutoTokenizer.from_pretrained(self.generator_path, trust_remote_code=True)
        else:
            self.is_chat = True
            self.enable_chat = True

        # 设置系统和用户提示，如果提供则使用提供的，否则使用默认的
        if len(system_prompt) == 0 and len(user_prompt) == 0:
            system_prompt = self.base_system_prompt
            user_prompt = self.base_user_prompt
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.enable_chat = enable_chat

        # 检查提示模板中是否包含所有必要的占位符
        self._check_placeholder()

    def _check_placeholder(self):
        # 检查模板中的占位符是否都被正确使用
        for holder in self.placeholders:
            flag = False
            for prompt in [self.system_prompt, self.user_prompt]:
                if f'{holder}' in prompt:
                    print(f"Find `{holder}` in template")
                    flag = True
                    break
            if not flag and holder != 'reference':
                assert False

    # 生成最终的输入字符串，用于模型生成
    def get_string(self,
                   question,
                   retrieval_result=None,
                   formatted_reference=None,
                   previous_gen=None,
                   **params
                   ):

        if formatted_reference is None:
            if retrieval_result is not None:
                formatted_reference = self.format_reference(retrieval_result)
            else:
                formatted_reference = ""

        input_params = {
            "question": question,
            "reference": formatted_reference
        }
        input_params.update(**params)

        system_prompt = self.system_prompt.format(**input_params)
        user_prompt = self.user_prompt.format(**input_params)

        # 根据是否启用聊天功能来决定如何构建输入
        if self.is_chat and self.enable_chat:
            input = []
            if system_prompt != "":
                input.append({"role": "system", "content": system_prompt})
            if user_prompt != "":
                input.append({"role": "user", "content": user_prompt})
            if self.is_openai:
                for item in input:
                    if item['role'] == 'system':
                        item['role'] == 'assistant'
            else:
                input = self.tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
        else:
            input = "\n\n".join([prompt for prompt in [system_prompt, user_prompt] if prompt != ""])

        if previous_gen is not None and self.is_openai is False:
            input += previous_gen

        return input

    # 格式化检索结果为引用文本
    def format_reference(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx + 1}(Title: {title}) {text}\n"

        return format_reference
