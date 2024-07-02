from fufanrag.evaluator import Evaluator
from fufanrag.dataset.utils import split_dataset, merge_dataset
from fufanrag.utils import get_retriever, get_generator, get_refiner, get_judger
from fufanrag.prompt import PromptTemplate


class BasicPipeline:
    """
    基类。一个管道包括了整个 RAG 流程。
    如果想实现一个管道，需要继承这个类。
    """

    def __init__(self, config, prompt_template=None):
        """初始化函数，设置管道配置和默认组件。

        参数:
        - config: 包含配置信息的字典，如设备和缓存设置。
        - prompt_template: 提示模板对象，如果未提供，则创建默认提示模板。
        """
        self.config = config
        self.device = config['device']
        self.retriever = None  # 检索器
        self.evaluator = Evaluator(config)  # 评估器
        self.save_retrieval_cache = config['save_retrieval_cache']

        # 提示模板
        if prompt_template is None:
            prompt_template = PromptTemplate(config)
        self.prompt_template = prompt_template

    def run(self, dataset):
        """执行 RAG 框架的整体推理过程。

        参数:
        - dataset: 要处理的数据集对象。
        """
        pass

    def evaluate(self, dataset, do_eval=True, pred_process_fun=None):
        """模型完成推合后的评估过程。

        参数:
        - dataset: 要评估的数据集对象。
        - do_eval: 是否执行评估。
        - pred_process_fun: 预测数据的处理函数。
        """

        if pred_process_fun is not None:
            raw_pred = dataset.pred  # 获取原始预测数据
            processed_pred = [pred_process_fun(pred) for pred in raw_pred]  # 处理预测数据
            dataset.update_output('raw_pred', raw_pred)  # 更新数据集的原始预测输出
            dataset.update_output('pred', processed_pred)  # 更新数据集的处理后的预测输出

        if do_eval:
            # 执行评估并保存结果
            eval_result = self.evaluator.evaluate(dataset)
            print(eval_result)

        # 保存检索缓存
        if self.save_retrieval_cache:
            self.retriever._save_cache()

        return dataset


class SequentialPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None):
        """
        检索过程：
            query -> pre-retrieval -> retriever -> post-retrieval -> generator
        """

        super().__init__(config, prompt_template)
        # 获取检索器实例
        self.retriever = get_retriever(config)
        # 获取生成器实例
        self.generator = get_generator(config)

    def naive_run(self, dataset, do_eval=True, pred_process_fun=None):
        # 不采用RAG流程，直接让模型生成结果
        input_prompts = [self.prompt_template.get_string(question=q) for q in dataset.question]
        dataset.update_output('prompt', input_prompts)

        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset

    def run(self, dataset, do_eval=True, pred_process_fun=None):

        # 获取到传入数据集的问题
        input_query = dataset.question

        # 加载实例化好的检索器，执行相似性文本匹配
        retrieval_results = self.retriever.batch_search(input_query)
        dataset.update_output('retrieval_result', retrieval_results)

        input_prompts = [
            self.prompt_template.get_string(question=q, retrieval_result=r)
            for q, r in zip(dataset.question, dataset.retrieval_result)
        ]

        dataset.update_output('prompt', input_prompts)

        # 加载实例化好的生成器，针对测试集中的问题，完成推理回复
        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred", pred_answer_list)

        # 进入评估流程，根据指定的评估指标，计算 实际的推理回复和 golden_answers 的差异
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset


class ConditionalPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None):
        """
        检索过程:
            query -> judger -> sequential pipeline or naive generate
        """

        super().__init__(config, prompt_template)
        self.judger = get_judger(config)

        self.sequential_pipeline = SequentialPipeline(config, prompt_template)
        from flashrag.prompt import PromptTemplate
        self.zero_shot_templete = PromptTemplate(
            config=config,
            system_prompt="Answer the question based on your own knowledge. \
                            Only give me the answer and do not output any other words.",
            user_prompt="Question: {question}"
        )

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        # judge_result: list of bool element, representing whether to use retrieval
        judge_result = self.judger.judge(dataset)
        dataset.update_output('judge_result', judge_result)

        # split dataset based on judge_result
        pos_dataset, neg_dataset = split_dataset(dataset, judge_result)

        pos_dataset = self.sequential_pipeline.run(pos_dataset, do_eval=False)
        self.sequential_pipeline.prompt_template = self.zero_shot_templete
        neg_dataset = self.sequential_pipeline.naive_run(neg_dataset, do_eval=False)

        # merge datasets into original format
        dataset = merge_dataset(pos_dataset, neg_dataset, judge_result)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset


if __name__ == '__main__':
    import argparse
    from fufanrag.config import Config
    from fufanrag.utils import get_dataset
    from fufanrag.prompt import PromptTemplate

    config = {

        # 以下是必须结合自己实际情况修改的参数
        'dataset_path': '/home/00_rag/fufan-chat-api/fufanrag/data/test',   # 评估数据集的存放路径
        "index_path": "/home/00_rag/fufan-chat-api/knowledge_base/wiki/vector_store/bge-large-zh-v1.5/index.faiss",  # 向量索引存放路径
        "corpus_path": "/home/00_rag/fufan-chat-api/fufanrag/data/indexes/sample_data.jsonl",  # 原始文本存放路径
        "retrieval_model_path": "/home/00_rag/model/AI-ModelScope/bge-large-zh-v1___5",  # 用于检索的模型存放路径
        "retrieval_cache_path": "/home/00_rag/fufan-chat-api/fufanrag/data/cache",    # 检索缓存存放目录
        "generator_model": "chatglm3-6b",  # 生成模型名称
        "generator_model_path": "/home/00_rag/model/ZhipuAI/chatglm3-6b",  # 生成模型存放路径
        "save_dir": "/home/00_rag/fufan-chat-api/fufanrag/data/result",    #


        # 如下参数可灵活修改
        'dataset_name': "test",
        'split': ['test'],
        'test_sample_num': 10,
        'random_sample': False,
        'retrieval_method': 'bge-large-zh-v1',
        "retrieval_topk": 3,
        "save_retrieval_cache": False,
        "use_retrieval_cache": False,
        "use_reranker": False,
        # "use_reranker": True,
        # "rerank_model_path":"/home/00_rag/model/Xorbits/bge-reranker-large",
        # "rerank_model_name":"bge-reranker-large",
        # "rerank_topk": 3,
        # "rerank_max_length": 512,
        # "rerank_batch_size": 2,
        # "rerank_use_fp16": True,
        "faiss_gpu": False,
        "use_sentence_transformer": False,
        "retrieval_pooling_method": "mean",
        "retrieval_batch_size": 12,
        "retrieval_query_max_length": 128,
        "retrieval_use_fp16": True,
        "generator_max_input_len": 2048,
        "generator_batch_size": 2,
        "generation_params": {"do_sample": False, "max_tokens": 1024},
        "device": "cuda",
        "save_metric_score": True,
        "save_intermediate_data": True,
        "metrics": ['em', 'sub_em', 'f1', 'precision', 'recall'],
        "framework": "fschat"
    }

    # 读取数据集
    all_split = get_dataset(config)

    # # 查看加载的数据信息
    # for split, dataset in all_split.items():
    #     if dataset is not None and len(dataset) > 0:
    #         print(f"First question in {split} dataset: {dataset.question[0]}")
    #         print(f"First golden answer in {split} dataset: {dataset.golden_answers[0]}")

    # 切分测试集
    test_data = all_split['test']
    # # 查看用于测试评估的加载内容
    # if test_data is not None and len(test_data) > 0:
    #     print("All questions in the test dataset:")
    #     print(test_data.question)  # 使用 @property 定义的方法获取所有问题
    #     print("\nAll golden answers in the test dataset:")
    #     print(test_data.golden_answers)  # 获取所有正确答案
    # else:
    #     print("Test dataset is not loaded or does not exist.")

    # 构建基于RAG评估的提示模板
    prompt_templete = PromptTemplate(
        config,
        system_prompt="根据给定文档回答问题。只给出答案，不要输出其他任何词语。 \
                        \n下面是给定的文档。\n\n{reference}",
        user_prompt="问题: {question}\n答案："
    )

    pipeline = SequentialPipeline(config, prompt_template=prompt_templete)

    output_dataset = pipeline.run(test_data, do_eval=True)
    print("---generation output---")
    for single_reponse in output_dataset.pred:
        print(single_reponse)
