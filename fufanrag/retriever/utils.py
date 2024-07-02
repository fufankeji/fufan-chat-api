import json
import datasets
from transformers import AutoTokenizer, AutoModel, AutoConfig


def load_model(
        model_path: str,
        use_fp16: bool = False
    ):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)

    return model, tokenizer


def pooling(
        pooler_output,
        last_hidden_state,
        attention_mask = None,
        pooling_method = "mean"
    ):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")

# def load_corpus(corpus_path: str):
#     corpus = datasets.load_dataset(
#             'json',
#             data_files=corpus_path,
#             split="train",
#             num_proc=4)
#     return corpus
import pickle
import json
from datasets import load_dataset

def convert_pkl_to_jsonl(pkl_path, jsonl_path):
    # 加载 .pkl 文件
    with open(pkl_path, 'rb') as file:
        data = pickle.load(file)

    # 写入到 .jsonl 文件
    with open(jsonl_path, 'w') as file:
        for item in data:
            # 假设每个元素都是一个可以直接转为 JSON 的字典
            json.dump(item, file)
            file.write('\n')

def load_corpus(corpus_path: str):
    # 检查文件扩展名
    if corpus_path.endswith('.pkl'):
        # 定义新的 JSONL 文件路径
        jsonl_path = corpus_path.replace('.pkl', '.jsonl')
        # 转换 .pkl 到 .jsonl
        convert_pkl_to_jsonl(corpus_path, jsonl_path)
        # 更新 corpus_path 为新的 JSONL 文件路径
        corpus_path = jsonl_path

    # 加载数据集
    corpus = load_dataset(
        'json',
        data_files=corpus_path,
        split='train',
        num_proc=4
    )
    return corpus

def read_jsonl(file_path):
    with open(file_path, "r") as f:
        while True:
            new_line = f.readline()
            if not new_line:
                return
            new_item = json.loads(new_line)

            yield new_item


def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]

    return results