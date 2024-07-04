from llama_index.legacy import Document, VectorStoreIndex
from llama_index.legacy.node_parser import SimpleNodeParser
import os
from llama_index.legacy.vector_stores import MilvusVectorStore
from llama_index.legacy.storage import StorageContext
from configs import SERPER_API_KEY, URL, ZILLIZ_URI, ZILLIZ_TOKEN, ZILLIZ_DIM, ZILLIZ_COLLECTION
import hashlib
import uuid
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
from configs.model_config import SEARCH_RERANK_TOP_K


def md5(data: str):
    _md5 = hashlib.md5()
    _md5.update(data.encode("utf-8"))
    _hash = _md5.hexdigest()

    return _hash


def build_document(search_results):
    """
    构建Document对象
    """
    documents = []

    for result in search_results:

        if "uuid" in result:
            uuid = result["uuid"]
        else:
            uuid = md5(result["link"])

        text = result["snippet"]

        document = Document(
            page_content=text,
            metadata={
                "uuid": uuid,
                "title": result["title"],
                "snippet": result["snippet"],
                "link": result["link"],
            },
        )

        documents.append(document)

    return documents


def reranking(query, search_results, top_k=SEARCH_RERANK_TOP_K):
    # 将第一轮联网检索得到的网页信息构建成Document对象
    documents = build_document(search_results=search_results)

    # 计算query 与 每一个检索到的网页的snippet的文本相似性，判断其网页是否与当前的query高度相关
    normal = NormalizedLevenshtein()
    for x in documents:
        x.metadata["score"] = normal.similarity(query, x.page_content)

    # 降序排序
    documents.sort(key=lambda x: x.metadata["score"], reverse=True)

    # 返回最相关的 top_k 个网页信息数据
    return documents[:SEARCH_RERANK_TOP_K]
