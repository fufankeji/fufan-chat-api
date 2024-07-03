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
        if "content" in result and len(result["content"]) > len(result["snippet"]):
            text = result["content"]

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


def reranking(query, search_results, top_k=1):

    results = []
    for item in search_results:
        # 为每个搜索结果生成 UUID（MD5 哈希）
        item["uuid"] = hashlib.md5(item["link"].encode()).hexdigest()
        # 初始化搜索结果的得分
        item["score"] = 0.00
        results.append(item)

    documents = build_document(search_results=results)

    normal = NormalizedLevenshtein()
    for x in documents:
        x.metadata["score"] = normal.similarity(query, x.page_content)
    documents.sort(key=lambda x: x.metadata["score"], reverse=True)

    return documents[:top_k]