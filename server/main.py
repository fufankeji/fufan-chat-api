import json
import os
import warnings
from typing import List, Dict
import functools
from tqdm import tqdm
import faiss

# from flashrag.utils import get_reranker
# from flashrag.retriever.utils import load_corpus, load_docs
# from flashrag.retriever.encoder import Encoder, STEncoder

def cache_manager(func):
    """
    Decorator used for retrieving document cache.
    With the decorator, The retriever can store each retrieved document as a file and reuse it.
    """

    @functools.wraps(func)
    def wrapper(self, query_list, num = None, return_score = False):
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
                no_cache_results, no_cache_scores = self._batch_search_with_rerank(no_cache_query, num ,True)
                no_cache_idx = 0
                for idx,res in enumerate(cache_results):
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
    def wrapper(self, query_list, num = None, return_score = False):
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
    """Base object for all retrievers."""

    def __init__(self, config):
        self.config = config
        self.retrieval_method = config['retrieval_method']
        self.topk = config['retrieval_topk']

        self.index_path = config['index_path']
        self.corpus_path = config['corpus_path']


        self.use_reranker = config['use_reranker']
        if self.use_reranker:
            self.reranker = get_reranker(config)


    def _save_cache(self):
        with open(self.cache_save_path, "w") as f:
            json.dump(self.cache, f, indent=4)

    def _search(self, query: str, num: int, return_score: bool) -> List[Dict[str, str]]:
        r"""Retrieve topk relevant documents in corpus.

        Return:
            list: contains information related to the document, including:
                contents: used for building index
                title: (if provided)
                text: (if provided)

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


def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
            'json',
            data_files=corpus_path,
            split="train",
            num_proc=4)
    return corpus

class BM25Retriever(BaseRetriever):
    r"""BM25 retriever based on pre-built pyserini index."""

    def __init__(self, config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher
        self.searcher = LuceneSearcher("/home/00_rag/temp/fufan-chat-api/server/indexes")
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8

    def _check_contain_doc(self):
        r"""Check if the index contains document content
        """
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score = False) -> List[Dict[str, str]]:
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if return_score:
                return [],[]
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
                        'contents': content} for content in all_contents]
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
        else:
            return results


    def _batch_search(self, query_list, num: int = None, return_score = False):
        # TODO: modify batch method
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num,True)
            results.append(item_result)
            scores.append(item_score)

        if return_score:
            return results, scores
        else:
            return results

# 定义检索器配置
retriever_config = {
    'data_dir': '/home/00_rag/temp/fufan-chat-api/server/dataset/',
    'index_path': '/home/00_rag/temp/fufan-chat-api/server/indexes/e5_flat_sample.index',
    'corpus_path': '/home/00_rag/temp/fufan-chat-api/server/indexes/sample_data.jsonl',
    'device': 'cpu',
    'save_retrieval_cache': True,
    'retrieval_method': 'bm25',
    'retrieval_topk': 3,
    "use_reranker": False,
}

if __name__ == '__main__':
    input_query = "你好，请你介绍一下你自己"
    retriever = BM25Retriever(retriever_config)
    print(retriever)