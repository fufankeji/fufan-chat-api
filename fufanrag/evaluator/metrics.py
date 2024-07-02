import re
import warnings
from collections import Counter
from fufanrag.evaluator.utils import normalize_answer


class BaseMetric:
    """`BaseMetric` serves as the base object of all metrics. Implemented metric should
    inherit this class.
    """

    metric_name = "base"

    def __init__(self, config):
        self.config = config
        self.dataset_name = config['dataset_name']

    def calculate_metric(self, data):
        """Get the total score of this metric and score for each sample.

        Args:
            data object: it contains basic information and generated information.

        Returns:
            (metric_score: dict, metric_score_list: list)
            metric_score: such as ``{'em': 0.53}``.
            metric_score_list: score for each sample.

        """
        return {}, []


class F1_Score(BaseMetric):
    """
    计算预测和真实答案之间的F1分数，这是精确度和召回率的调和平均数。
    """

    metric_name = "f1"

    def __init__(self, config):
        super().__init__(config)

    def token_level_scores(self, prediction: str, ground_truths: str):
        """
        计算单个预测与一个或多个真实答案之间的token级F1分数。

        Args:
            prediction (str): 预测的文本。
            ground_truths (str or list): 真实的答案，可以是单个字符串或字符串列表。

        Returns:
            dict: 包含 'f1', 'precision', 和 'recall' 的字典。
        """
        final_metric = {'f1': 0, 'precision': 0, 'recall': 0}
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]
        for ground_truth in ground_truths:
            normalized_prediction = normalize_answer(prediction)
            normalized_ground_truth = normalize_answer(ground_truth)

            # 如果预测或真实答案为特定单词（yes, no, noanswer），且不匹配，则跳过
            if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            if normalized_ground_truth in ['yes', 'no',
                                           'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue

            # 将预测的答案和真实的答案分割成单词（tokens）
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()

            # 使用Counter对象计算两组tokens的计数，然后使用 & 操作符得到两者的交集，即预测答案和真实答案中共同出现的单词及其最小出现次数。
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue

            # 计算公式为共同单词总数除以预测答案中的单词总数。这衡量了预测中的哪些单词实际上是正确的。
            precision = 1.0 * num_same / len(prediction_tokens)

            # 计算公式为共同单词总数除以真实答案中的单词总数。这衡量了所有应该被预测的正确单词中有多少被实际预测出来了。
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ['f1', 'precision', 'recall']:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric

    def calculate_metric(self, data):
        """
        计算数据集上的F1分数。

        Args:
            data: 包含预测答案和真实答案的数据集。

        Returns:
            tuple: 包含总体F1分数和每个样本的F1分数列表。

        """
        pred_list = data.pred
        golden_answers_list = data.golden_answers
        metric_score_list = [self.token_level_scores(pred, golden_answers)['f1'] for pred, golden_answers in
                             zip(pred_list, golden_answers_list)]
        f1 = sum(metric_score_list) / len(metric_score_list)
        return {"f1": f1}, metric_score_list


class Recall_Score(F1_Score):
    """基于分词的召回率得分"""

    # 定义度量名称为召回率
    metric_name = "recall"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        """
        计算数据集上的召回率得分。

        Args:
            data: 包含预测答案和真实答案的数据集。

        Returns:
            tuple: 包含总体召回率得分和每个样本的召回率得分列表。
        """

        # 从数据集中获取所有预测答案
        pred_list = data.pred
        print(f"pred_list: {pred_list}")
        # 从数据集中获取所有真实答案
        golden_answers_list = data.golden_answers
        print(f"golden_answers_list: {golden_answers_list}")
        # 计算每个样本的召回率
        metric_score_list = [self.token_level_scores(pred, golden_answers)['recall'] for pred, golden_answers in
                             zip(pred_list, golden_answers_list)]

        # 计算平均召回率
        precision = sum(metric_score_list) / len(metric_score_list)

        # 返回召回率结果
        return {"recall": precision}, metric_score_list


class Precision_Score(F1_Score):
    """Token-level Precision score"""

    metric_name = "precision"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        pred_list = data.pred
        golden_answers_list = data.golden_answers
        metric_score_list = [self.token_level_scores(pred, golden_answers)['precision'] for pred, golden_answers in
                             zip(pred_list, golden_answers_list)]
        precision = sum(metric_score_list) / len(metric_score_list)
        return {"precision": precision}, metric_score_list


class ExactMatch(BaseMetric):
    """
    Exact match (EM) 指标用于测量预测答案与标准答案是否完全一致。

    Attributes:
        metric_name (str): 度量指标的名称，这里是"em"，代表精确匹配。


    精确匹配（Exact Match, EM）是一种常用于评估自然语言处理任务，特别是问答系统中的指标。
    它测量的是预测答案是否与真实答案在文本上完全一致。
    这种评估方式非常严格，即只有当预测的答案与参考答案在字面上完全相同，包括所有的单词和标点符号，才会被视为正确。
    """
    metric_name = "em"

    def __init__(self, config):
        """
        初始化ExactMatch实例。

        Args:
            config (dict): 配置字典，包含评估所需的各种参数。
        """
        super().__init__(config)
        self.is_regex = self.dataset_name == 'curatedtrec'

    def calculate_em(self, prediction: str, golden_answers: list) -> float:
        """
        计算单个预测的精确匹配得分。

        Args:
            prediction (str): 模型生成的预测文本。
            golden_answers (list): 可能的正确答案列表。

        Returns:
            float: 预测的精确匹配得分，1.0 表示完全匹配，0.0 表示不匹配。
        """
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        normalized_prediction = normalize_answer(prediction)
        score = 0.0
        for golden_answer in golden_answers:
            # 如果答案应视为正则表达式，则以正则表达式方式匹配
            if self.is_regex:
                print("Consider answer as regex!")
                golden_answer = re.compile(golden_answer, re.IGNORECASE)
                match = re.fullmatch(golden_answer, normalized_prediction)
                if match is not None:
                    score = 1.0
                    break
            else:
                golden_answer = normalize_answer(golden_answer)
                if golden_answer == normalized_prediction:
                    score = 1.0
                    break
        return score

    def calculate_metric(self, data):
        golden_answers_list = data.golden_answers
        pred_list = data.pred

        metric_score_list = [self.calculate_em(pred, golden_answers) for pred, golden_answers in
                             zip(pred_list, golden_answers_list)]
        em_score = sum(metric_score_list) / len(metric_score_list)

        return {"em": em_score}, metric_score_list


class Sub_ExactMatch(BaseMetric):
    """
    是基于基类 BaseMetric 实现的一个评估指标。
    这个指标用于衡量预测答案是否包含了标准答案，即使不完全相同，也认为是部分正确。
    这种评估方法比完全精确匹配（Exact Match）要宽松，适合于那些允许答案有部分对应即可的场景。

    """
    metric_name = "sub_em"

    def __init__(self, config):
        super().__init__(config)
        self.is_regex = self.dataset_name == 'curatedtrec'

    def calculate_sub_em(self, prediction: str, golden_answers: list) -> float:
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
        normalized_prediction = normalize_answer(prediction)
        score = 0.0
        for golden_answer in golden_answers:
            if self.is_regex:
                print("Consider answer as regex!")
                golden_answer = re.compile(golden_answer, re.IGNORECASE)
                match = re.search(golden_answer, normalized_prediction)
                if match is not None:
                    score = 1.0
                    break
            else:
                golden_answer = normalize_answer(golden_answer)
                if golden_answer in normalized_prediction:
                    score = 1.0
                    break
        return score

    def calculate_metric(self, data):
        golden_answers_list = data.golden_answers
        pred_list = data.pred

        metric_score_list = [self.calculate_sub_em(pred, golden_answers) for pred, golden_answers in
                             zip(pred_list, golden_answers_list)]
        sub_em_score = sum(metric_score_list) / len(metric_score_list)

        return {"sub_em": sub_em_score}, metric_score_list


class Retrieval_Recall(BaseMetric):
    r"""The recall of the top-k retreived passages, we measure if any of the passage contain the answer string. """
    metric_name = "retrieval_recall"

    def __init__(self, config):
        super().__init__(config)
        self.topk = config['metric_setting']['retrieval_recall_topk']

    def calculate_metric(self, data):
        golden_answers_list = data.golden_answers
        retrieve_docs = data.retrieval_result
        recall_score_list = []
        for doc_list, golden_answers in zip(retrieve_docs, golden_answers_list):
            if len(doc_list) < self.topk:
                warnings.warn(f"Length of retrieved docs is smaller than topk ({self.topk})")
            doc_list = [doc['contents'] for doc in doc_list[:self.topk]]
            hit_list = []
            for doc in doc_list:
                for golden_answer in golden_answers:
                    if normalize_answer(golden_answer) in normalize_answer(doc):
                        hit_list.append(True)
                        break
                else:
                    hit_list.append(False)
            score = 1 if any(hit_list) else 0
            recall_score_list.append(score)
        recall_score = sum(recall_score_list) / len(recall_score_list)

        return {f"retrieval_recall_top{self.topk}": recall_score}, recall_score_list


class Retrieval_Precision(BaseMetric):
    r"""The precision of the top-k retreived passages, we measure if any of the passage contain the answer string. """
    metric_name = "retrieval_precision"

    def __init__(self, config):
        super().__init__(config)
        self.topk = config['metric_setting']['retrieval_recall_topk']

    def calculate_metric(self, data):
        golden_answers_list = data.golden_answers
        retrieve_docs = data.retrieval_result
        precision_score_list = []
        for doc_list, golden_answers in zip(retrieve_docs, golden_answers_list):
            if len(doc_list) < self.topk:
                warnings.warn(f"Length of retrieved docs is smaller than topk ({self.topk})")
            doc_list = [doc['contents'] for doc in doc_list[:self.topk]]
            hit_list = []
            for doc in doc_list:
                for golden_answer in golden_answers:
                    if normalize_answer(golden_answer) in normalize_answer(doc):
                        hit_list.append(True)
                        break
                else:
                    hit_list.append(False)
            score = sum(hit_list) / len(hit_list)
            precision_score_list.append(score)
        precision_score = sum(precision_score_list) / len(precision_score_list)

        return {f"retrieval_precision_top{self.topk}": precision_score}, precision_score_list


class Rouge_Score(BaseMetric):
    metric_name = "rouge_score"

    def __init__(self, config):
        super().__init__(config)
        from rouge import Rouge
        self.scorer = Rouge()

    def calculate_rouge(self, pred, golden_answers):
        output = {}
        for answer in golden_answers:
            scores = self.scorer.get_scores(pred, answer)
            for key in ['rouge-1', 'rouge-2', 'rouge-l']:
                if key not in output:
                    output[key] = []
                output[key].append(scores[0][key]['f'])
        for k, v in output.items():
            output[k] = max(v)

        return output


class Rouge_1(Rouge_Score):
    metric_name = "rouge-1"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        golden_answers_list = data.golden_answers
        pred_list = data.pred

        metric_score_list = [self.calculate_rouge(pred, golden_answers)['rouge-1'] for pred, golden_answers in
                             zip(pred_list, golden_answers_list)]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"rouge-1": score}, metric_score_list


class Rouge_2(Rouge_Score):
    metric_name = "rouge-2"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        golden_answers_list = data.golden_answers
        pred_list = data.pred

        metric_score_list = [self.calculate_rouge(pred, golden_answers)['rouge-2'] for pred, golden_answers in
                             zip(pred_list, golden_answers_list)]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"rouge-2": score}, metric_score_list


class Rouge_L(Rouge_Score):
    metric_name = "rouge-l"

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, data):
        golden_answers_list = data.golden_answers
        pred_list = data.pred

        metric_score_list = [self.calculate_rouge(pred, golden_answers)['rouge-l'] for pred, golden_answers in
                             zip(pred_list, golden_answers_list)]
        score = sum(metric_score_list) / len(metric_score_list)

        return {"rouge-l": score}, metric_score_list


class BLEU(BaseMetric):
    metric_name = "bleu"

    def __init__(self, config):
        super().__init__(config)
        from ._bleu import Tokenizer13a
        self.tokenizer = Tokenizer13a()
        self.max_order = config['metric_setting'].get('bleu_max_order', 4)
        self.smooth = config['metric_setting'].get('bleu_smooth', False)

    def calculate_metric(self, data):
        from ._bleu import compute_bleu
        golden_answers_list = data.golden_answers
        pred_list = data.pred

        pred_list = [self.tokenizer(pred) for pred in pred_list]
        golden_answers_list = [[self.tokenizer(ans) for ans in golden_answers] for golden_answers in
                               golden_answers_list]
        score = compute_bleu(
            reference_corpus=golden_answers_list,
            translation_corpus=pred_list,
            max_order=self.max_order,
            smooth=self.smooth
        )
        (total_bleu, precisions, bp, ratio, translation_length, reference_length) = score

        score_list = []
        for pred, golden_answers in zip(pred_list, golden_answers_list):
            pred = [pred]
            golden_answers = [golden_answers]
            score = compute_bleu(
                reference_corpus=golden_answers_list,
                translation_corpus=pred_list,
                max_order=self.max_order,
                smooth=self.smooth
            )
            (bleu, precisions, bp, ratio, translation_length, reference_length) = score
            score_list.append(bleu)

        return {"bleu": total_bleu}, score_list


class CountToken(BaseMetric):
    metric_name = "input_tokens"

    def __init__(self, config):
        super().__init__(config)
        tokenizer_name = config['metric_setting'].get('tokenizer_name', None)
        is_hf_tokenizer = True
        from flashrag.utils.constants import OPENAI_MODEL_DICT
        if tokenizer_name is None or tokenizer_name in OPENAI_MODEL_DICT:
            # use gpt4 tokenizer
            import tiktoken
            if tokenizer_name is None:
                tokenizer_name = 'gpt-4'
            tokenizer = tiktoken.encoding_for_model(tokenizer_name)
            is_hf_tokenizer = False
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.tokenizer = tokenizer
        self.is_hf_tokenizer = is_hf_tokenizer

    def calculate_metric(self, data):
        input_prompts = data.prompt
        if self.is_hf_tokenizer:
            token_counts = [len(self.tokenizer.tokenize(text)) for text in input_prompts]
        else:
            token_counts = [len(self.tokenizer.encode(text)) for text in input_prompts]
        avg_tokens = sum(token_counts) / len(token_counts)

        return {"avg_input_tokens": avg_tokens}, token_counts