import os
import importlib
from transformers import AutoConfig
from fufanrag.dataset.dataset import Dataset


def get_dataset(config):
    """从配置加载数据集。"""

    # 从配置中获取数据集的路径
    dataset_path = config['dataset_path']
    # 获取要加载的数据集的分割名称，如'train'、'test'等
    all_split = config['split']

    # 创建一个字典，用于存储每个分割的数据集对象
    split_dict = {split: None for split in all_split}

    # 遍历每个分割
    for split in all_split:
        # 构建该分割对应的文件路径
        split_path = os.path.join(dataset_path, f'{split}.jsonl')
        # 检查文件是否存在
        if not os.path.exists(split_path):
            print(f"{split} file not exists!")  # 如果文件不存在，打印提示信息
            continue
        # 如果分割是'test', 'val', 'dev'之一，初始化Dataset类的实例
        if split in ['test', 'val', 'dev']:
            split_dict[split] = Dataset(config,
                                        split_path,
                                        sample_num=config['test_sample_num'],
                                        random_sample=config['random_sample'])
        else:
            # 对于其它分割，直接初始化Dataset类的实例
            split_dict[split] = Dataset(config, split_path)

    # 返回包含所有分割数据集的字典
    return split_dict



def get_generator(config, **params):
    """Automatically select generator class based on config."""
    if config['framework'] == 'vllm':
        return getattr(
            importlib.import_module("flashrag.generator"),
            "VLLMGenerator"
        )(config, **params)
    elif config['framework'] == 'fschat':
        return getattr(
            importlib.import_module("fufanrag.generator"),
            "FastChatGenerator"
        )(config, **params)
    elif config['framework'] == 'hf':
        if "t5" in config['generator_model'] or "bart" in config['generator_model']:
            return getattr(
                importlib.import_module("flashrag.generator"),
                "EncoderDecoderGenerator"
            )(config, **params)
        else:
            return getattr(
                importlib.import_module("flashrag.generator"),
                "HFCausalLMGenerator"
            )(config, **params)
    elif config['framework'] == 'openai':
        return getattr(
            importlib.import_module("flashrag.generator"),
            "OpenaiGenerator"
        )(config, **params)
    else:
        raise NotImplementedError


def get_retriever(config):
    r"""根据配置自动选择检索器类

    参数:
        config (dict): 包含 'retrieval_method' 键的配置

    返回:
        Retriever: 检索器实例
    """
    if config['retrieval_method'] == "bm25":
        return getattr(
            importlib.import_module("fufanrag.retriever"),
            "BM25Retriever"
        )(config)
    else:
        return getattr(
            importlib.import_module("fufanrag.retriever"),
            "DenseRetriever"
        )(config)



def get_reranker(config):
    model_path = config['rerank_model_path']
    # get model config
    model_config = AutoConfig.from_pretrained(model_path)
    arch = model_config.architectures[0]
    if 'forsequenceclassification' in arch.lower():
        return getattr(
            importlib.import_module("fufanrag.retriever"),
            "CrossReranker"
        )(config)
    else:
        return getattr(
            importlib.import_module("fufanrag.retriever"),
            "BiReranker"
        )(config)


def get_judger(config):
    judger_name = config['judger_name']
    if 'skr' in judger_name.lower():
        return getattr(
            importlib.import_module("flashrag.judger"),
            "SKRJudger"
        )(config)
    else:
        assert False, "No implementation!"


def get_refiner(config):
    refiner_name = config['refiner_name']
    refiner_path = config['refiner_model_path']

    default_path_dict = {
        'recomp_abstractive_nq': 'fangyuan/nq_abstractive_compressor',
        'recomp:abstractive_tqa': 'fangyuan/tqa_abstractive_compressor',
        'recomp:abstractive_hotpotqa': 'fangyuan/hotpotqa_abstractive',
    }

    if refiner_path is None:
        if refiner_name in default_path_dict:
            refiner_path = default_path_dict[refiner_name]
        else:
            assert False, "refiner_model_path is empty!"

    model_config = AutoConfig.from_pretrained(refiner_path)
    arch = model_config.architectures[0].lower()
    if "recomp" in refiner_name.lower() or \
            "recomp" in refiner_path or \
            'bert' in arch:
        if model_config.model_type == "t5":
            return getattr(
                importlib.import_module("flashrag.refiner"),
                "AbstractiveRecompRefiner"
            )(config)
        else:
            return getattr(
                importlib.import_module("flashrag.refiner"),
                "ExtractiveRefiner"
            )(config)
    elif "lingua" in refiner_name.lower():
        return getattr(
            importlib.import_module("flashrag.refiner"),
            "LLMLinguaRefiner"
        )(config)
    elif "selective-context" in refiner_name.lower() or "sc" in refiner_name.lower():
        return getattr(
            importlib.import_module("flashrag.refiner"),
            "SelectiveContextRefiner"
        )(config)
    else:
        assert False, "No implementation!"
