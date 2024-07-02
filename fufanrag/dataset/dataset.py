import os
import json
import random
import numpy as np


class Item:
    """
     一个容器类，用于存储和操作数据集中的单个样本。
     在训练/推理过程中与此样本相关的信息将存储在 `self.output` 中。
     此类的每个属性可以像字典键一样使用（也适用于 `self.output` 中的键）。
     """

    def __init__(self, item_dict):
        # 初始化各个属性，如果字典中没有对应的键，则使用默认值
        self.id = item_dict.get("id", None)
        self.question = item_dict.get("question", None)
        self.golden_answers = item_dict.get("golden_answers", [])
        self.metadata = item_dict.get("metadata", {})
        self.output = item_dict.get("output", {})

    def update_output(self, key, value):
        """
          更新输出字典，并保持 self.output 中的键可以被当做属性使用。
          不允许更改特定的键（如'id'、'question'等）。
          """
        if key in ['id', 'question', 'golden_answers', 'output']:
            raise AttributeError(f'{key} should not be changed')
        else:
            self.output[key] = value

    def update_evaluation_score(self, metric_name, metric_score):
        """
        更新此样本的评估分数。
        """
        if 'metric_score' not in self.output:
            self.output['metric_score'] = {}
        self.output['metric_score'][metric_name] = metric_score

    def __getattr__(self, attr_name):
        """
        允许通过属性方式访问 output 字典中的内容。
        如果属性不存在，抛出 AttributeError。
        """
        if attr_name in ['id', 'question', 'golden_answers', 'metadata', 'output']:
            return super().__getattribute__(attr_name)
        else:
            output = super().__getattribute__('output')
            if attr_name in output:
                return output[attr_name]
            else:
                raise AttributeError(f"Attribute `{attr_name}` not found")

    def to_dict(self):
        """
        将数据样本的所有信息转换为字典格式。在推理过程中生成的信息将保存到 output 字段。
        """
        for k, v in self.output.items():
            if isinstance(k, np.ndarray):
                self.output[k] = v.tolist()
        output = {
            "id": self.id,
            "question": self.question,
            "golden_answers": self.golden_answers,
            "output": self.output
        }
        if self.metadata != {}:
            output['metadata'] = self.metadata

        return output


class Dataset:
    """
    用于存储整个数据集的容器类。类内部，每个数据样本都存储在 `Item` 类中。
    数据集的属性代表了数据集中每个项的属性列表。
    """

    def __init__(self, config=None, dataset_path=None, data=None, sample_num=None, random_sample=False):
        # 初始化数据集配置和属性
        self.config = config
        self.dataset_name = config['dataset_name']
        self.dataset_path = dataset_path

        self.sample_num = sample_num
        self.random_sample = random_sample

        # 根据是否直接提供数据来加载数据
        if data is None:
            self.data = self._load_data(self.dataset_name, self.dataset_path)
        else:
            self.data = data

    def _load_data(self, dataset_name, dataset_path):
        """
        从指定的路径加载数据，或者（未来）直接下载文件。
        """
        data = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                item_dict = json.loads(line)
                item = Item(item_dict)
                data.append(item)
        if self.sample_num is not None:
            if self.random_sample:
                print(f"Random sample {self.sample_num} items in test set.")
                data = random.sample(data, self.sample_num)
            else:
                data = data[:self.sample_num]

        return data

    def update_output(self, key, value_list):
        """
        更新数据集中每个样本的整体输出字段。
        """

        assert len(self.data) == len(value_list)
        for item, value in zip(self.data, value_list):
            item.update_output(key, value)

    @property
    def question(self):
        # 返回所有样本的标准答案列表
        return [item.question for item in self.data]

    @property
    def golden_answers(self):
        # 返回所有样本的 ID 列表
        return [item.golden_answers for item in self.data]

    @property
    def id(self):
        return [item.id for item in self.data]

    @property
    def output(self):
        return [item.output for item in self.data]

    def get_batch_data(self, attr_name: str, batch_size: int):
        """
        批量获取数据集中的某个属性。
        """

        for i in range(0, len(self.data), batch_size):
            batch_items = self.data[i:i + batch_size]
            yield [item[attr_name] for item in batch_items]

    def __getattr__(self, attr_name):
        """
        对于后期构造的属性（不使用 property 实现的），获取整个数据集中此属性的列表。
        """
        return [item.__getattr__(attr_name) for item in self.data]

    def get_attr_data(self, attr_name):
        """For the attributes constructed later (not implemented using property),
        obtain a list of this attribute in the entire dataset.
        """
        return [item[attr_name] for item in self.data]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def save(self, save_path):
        """
        将数据集保存到原始格式。
        """
        save_data = [item.to_dict() for item in self.data]
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=4)







