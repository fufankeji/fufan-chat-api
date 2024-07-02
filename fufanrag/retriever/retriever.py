import json
import os
import warnings
from typing import List, Dict
import functools
from tqdm import tqdm
import faiss

from fufanrag.utils import get_reranker
from fufanrag.retriever.utils import load_corpus, load_docs
from fufanrag.retriever.encoder import Encoder, STEncoder


def cache_manager(func):
    """
    Decorator used for retrieving document cache.
    With the decorator, The retriever can store each retrieved document as a file and reuse it.
    """

    @functools.wraps(func)
    def wrapper(self, query_list, num=None, return_score=False):
        if num is None:
            num = self.topk
        if self.use_cache:
            if isinstance(query_list, str):
                new_query_list = [query_list]
            else:
                new_query_list = query_list

            no_cache_query = []
            cache_results = []
            for query in new_query_list:
                if query in self.cache:
                    cache_res = self.cache[query]
                    if len(cache_res) < num:
                        warnings.warn(f"The number of cached retrieval results is less than topk ({num})")
                    cache_res = cache_res[:num]
                    # separate the doc score
                    doc_scores = [item.pop('score') for item in cache_res]
                    cache_results.append((cache_res, doc_scores))
                else:
                    cache_results.append(None)
                    no_cache_query.append(query)

            if no_cache_query != []:
                # use batch search without decorator
                no_cache_results, no_cache_scores = self._batch_search_with_rerank(no_cache_query, num, True)
                no_cache_idx = 0
                for idx, res in enumerate(cache_results):
                    if res is None:
                        assert new_query_list[idx] == no_cache_query[no_cache_idx]
                        cache_results = (no_cache_results[no_cache_idx], no_cache_scores[no_cache_scores])
                        no_cache_idx += 1

            results, scores = ([t[0] for t in cache_results], [t[1] for t in cache_results])

        else:
            results, scores = func(self, query_list, num, True)

        if self.save_cache:
            # merge result and score
            if isinstance(query_list, str):
                query_list = [query_list]
                if 'batch' not in func.__name__:
                    results = [results]
                    scores = [scores]
            for query, doc_items, doc_scores in zip(query_list, results, scores):
                for item, score in zip(doc_items, doc_scores):
                    item['score'] = score
                self.cache[query] = doc_items

        if return_score:
            return results, scores
        else:
            return results

    return wrapper


def rerank_manager(func):
    """
    Decorator used for reranking retrieved documents.
    """

    @functools.wraps(func)
    def wrapper(self, query_list, num=None, return_score=False):
        results, scores = func(self, query_list, num, True)
        if self.use_reranker:
            results, scores = self.reranker.rerank(query_list, results)
            if 'batch' not in func.__name__:
                results = results[0]
                scores = scores[0]
        if return_score:
            return results, scores
        else:
            return results

    return wrapper


class BaseRetriever:
    """构建检索器的基类"""

    def __init__(self, config):
        self.config = config
        self.retrieval_method = config['retrieval_method']
        self.topk = config['retrieval_topk']

        self.index_path = config['index_path']
        self.corpus_path = config['corpus_path']

        self.save_cache = config['save_retrieval_cache']
        self.use_cache = config['use_retrieval_cache']
        self.cache_path = config['retrieval_cache_path']

        self.use_reranker = config['use_reranker']
        if self.use_reranker:
            self.reranker = get_reranker(config)

        if self.save_cache:
            self.cache_save_path = os.path.join(config['save_dir'], 'retrieval_cache.json')
            self.cache = {}
        if self.use_cache:
            assert self.cache_path is not None
            with open(self.cache_path, "r") as f:
                self.cache = json.load(f)

    def _save_cache(self):
        with open(self.cache_save_path, "w") as f:
            json.dump(self.cache, f, indent=4)

    def _search(self, query: str, num: int, return_score: bool) -> List[Dict[str, str]]:
        r"""在语料库中检索最相关的前k个文档。

        返回:
            list: 包含与文档相关的信息，包括：
                contents: 用于构建索引的内容
                title: 文档的标题（如果提供）
                text: 文档的正文（如果提供）
        """

        pass

    def _batch_search(self, query_list, num, return_score):
        pass

    @cache_manager
    @rerank_manager
    def search(self, *args, **kwargs):
        return self._search(*args, **kwargs)

    @cache_manager
    @rerank_manager
    def batch_search(self, *args, **kwargs):
        return self._batch_search(*args, **kwargs)

    @rerank_manager
    def _batch_search_with_rerank(self, *args, **kwargs):
        return self._batch_search(*args, **kwargs)

    @rerank_manager
    def _search_with_rerank(self, *args, **kwargs):
        return self._search(*args, **kwargs)


class BM25Retriever(BaseRetriever):
    r"""基于预构建的pyserini索引的BM25检索器。"""

    def __init__(self, config):
        super().__init__(config)  # 调用基类的构造函数
        from pyserini.search.lucene import LuceneSearcher
        self.searcher = LuceneSearcher(self.index_path)  # 创建一个Lucene搜索器实例
        self.contain_doc = self._check_contain_doc()  # 检查索引中是否包含文档内容
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)  # 如果索引中不包含文档内容，则加载语料库
        self.max_process_num = 8   # 设置最大处理数

    def _check_contain_doc(self):
        r"""检查索引是否包含文档内容
        """
        return self.searcher.doc(0).raw() is not None  # 返回索引中是否有原始文档内容

    def _search(self, query: str, num: int = None, return_score=False) -> List[Dict[str, str]]:
        if num is None:
            num = self.topk   # 如果未指定文档数，则使用默认值
        hits = self.searcher.search(query, num)  # 使用搜索器查询
        if len(hits) < 1:
            if return_score:  # 获取文档的分数
                return [], []
            else:
                return []

        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn('Not enough documents retrieved!')
        else:
            hits = hits[:num]

        if self.contain_doc:
            all_contents = [json.loads(self.searcher.doc(hit.docid).raw())['contents'] for hit in hits]
            results = [{'title': content.split("\n")[0].strip("\""),
                        'text': "\n".join(content.split("\n")[1:]),
                        'contents': content} for content in all_contents]  # 解析标题和文本
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list, num: int = None, return_score=False):
        # TODO: modify batch method
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)

        if return_score:
            return results, scores
        else:
            return results


class DenseRetriever(BaseRetriever):
    r"""基于预建faiss索引的密集检索器。"""
 
    def __init__(self, config: dict):
        super().__init__(config)
        # 读取faiss索引
        self.index = faiss.read_index(self.index_path)

        # 如果配置为使用GPU，将索引复制到所有GPU
        if config['faiss_gpu']:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True   # 使用16位浮点数，减少内存使用
            co.shard = True        # 分片索引以适应多GPU
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

        # 加载语料库
        self.corpus = load_corpus(self.corpus_path)

        # 根据配置选择句子转换器或标准编码器
        if config['use_sentence_transformer']:
            self.encoder = STEncoder(
                model_name=self.retrieval_method,
                model_path=config['retrieval_model_path'],
                max_length=config['retrieval_query_max_length'],
                use_fp16=config['retrieval_use_fp16']
            )
        else:
            self.encoder = Encoder(
                model_name=self.retrieval_method,
                model_path=config['retrieval_model_path'],
                pooling_method=config['retrieval_pooling_method'],
                max_length=config['retrieval_query_max_length'],
                use_fp16=config['retrieval_use_fp16']
            )

        # 设置返回的最多结果数
        self.topk = config['retrieval_topk']
        # 批量查询的批次大小
        self.batch_size = self.config['retrieval_batch_size']

    def _search(self, query: str, num: int = None, return_score=False):
        if num is None:
            num = self.topk

        # 编码查询
        query_emb = self.encoder.encode(query)
        # 在索引中搜索
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]

        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score=False):
        # 如果传入的是单个字符串，将其转换为列表
        if isinstance(query_list, str):
            query_list = [query_list]

        # 如果没有指定返回的最大结果数量，使用默认的topk值
        if num is None:
            num = self.topk

        # 使用配置中定义的批量大小
        batch_size = self.batch_size

        results = []  # 用于存储检索结果的列表
        scores = []   # 用于存储相似度得分的列表

        # tqdm是一个进度条库，这里用于显示检索过程的进度
        for start_idx in tqdm(range(0, len(query_list), batch_size), desc='Retrieval process: '):
            # 获取当前批次的查询列表
            query_batch = query_list[start_idx:start_idx + batch_size]
            # 将查询批次编码成向量
            batch_emb = self.encoder.encode(query_batch)
            # 使用faiss索引执行批量检索
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            # 将得分数组转换为列表
            batch_scores = batch_scores.tolist()
            # 将索引数组转换为列表
            batch_idxs = batch_idxs.tolist()
            # 展平索引列表，以便一次性加载所有文档
            flat_idxs = sum(batch_idxs, [])
            # 加载对应索引的文档
            batch_results = load_docs(self.corpus, flat_idxs)
            # 将检索到的文档按照每个查询进行分组
            batch_results = [batch_results[i * num: (i + 1) * num] for i in range(len(batch_idxs))]

            # 添加当前批次的得分到总得分列表中
            scores.extend(batch_scores)
            # 添加当前批次的结果到总结果列表中
            results.extend(batch_results)
            
        if return_score:
            return results, scores
        else:
            return results


