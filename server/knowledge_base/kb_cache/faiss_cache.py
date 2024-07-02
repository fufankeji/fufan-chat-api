from configs import CACHED_VS_NUM, CACHED_MEMO_VS_NUM
from server.knowledge_base.kb_cache.base import *
from server.knowledge_base.kb_service.base import EmbeddingsFunAdapter
from server.utils import load_local_embeddings
from server.knowledge_base.utils import get_vs_path
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
import os
from langchain.schema import Document


# 修补 FAISS 以在 Document.metadata 中包含文档 ID
def _new_ds_search(self, search: str) -> Union[str, Document]:
    if search not in self._dict:
        return f"ID {search} not found."
    else:
        doc = self._dict[search]
        if isinstance(doc, Document):
            doc.metadata["id"] = search
        return doc
InMemoryDocstore.search = _new_ds_search


class ThreadSafeFaiss(ThreadSafeObject):
    """
    线程安全的 FAISS 类，用于管理和操作 FAISS 向量存储。
    """
    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"<{cls}: key: {self.key}, obj: {self._obj}, docs_count: {self.docs_count()}>"


    def docs_count(self) -> int:
        """
        获取文档数量。
        """
        return len(self._obj.docstore._dict)

    def save(self, path: str, create_path: bool = True):
        """
        保存向量存储到指定路径。

        参数：
        path (str): 保存路径。
        create_path (bool): 如果路径不存在，是否创建路径。默认值为 True。

        返回：
        保存结果。
        """
        with self.acquire():
            if not os.path.isdir(path) and create_path:
                os.makedirs(path)
            ret = self._obj.save_local(path)
            logger.info(f"已将向量库 {self.key} 保存到磁盘")
        return ret

    def clear(self):
        """
        清空向量存储中的所有文档。

        返回：
        被删除的文档 ID 列表。
        """
        ret = []
        with self.acquire():
            ids = list(self._obj.docstore._dict.keys())
            if ids:
                ret = self._obj.delete(ids)
                assert len(self._obj.docstore._dict) == 0
            logger.info(f"已将向量库 {self.key} 清空")
        return ret


class _FaissPool(CachePool):
    """
    FAISS 池，用于管理 FAISS 向量存储的创建、保存和卸载。
    """
    def new_vector_store(
        self,
        embed_model: str = EMBEDDING_MODEL,
        embed_device: str = embedding_device(),
    ) -> FAISS:
        """
        创建一个新的 FAISS 向量存储。

        参数：
        embed_model (str): 嵌入模型名称，默认值为 EMBEDDING_MODEL。
        embed_device (str): 嵌入设备，默认值为 embedding_device。

        返回：
        新的 FAISS 向量存储。
        """
        embeddings = EmbeddingsFunAdapter(embed_model)
        doc = Document(page_content="init", metadata={})
        vector_store = FAISS.from_documents([doc], embeddings, distance_strategy="METRIC_INNER_PRODUCT")
        ids = list(vector_store.docstore._dict.keys())
        vector_store.delete(ids)
        return vector_store

    def save_vector_store(self, kb_name: str, path: str=None):
        """
        保存指定知识库的向量存储。

        参数：
        kb_name (str): 知识库名称。
        path (str): 保存路径。

        返回：
        保存结果。
        """
        if cache := self.get(kb_name):
            return cache.save(path)

    def unload_vector_store(self, kb_name: str):
        """
        卸载指定知识库的向量存储。

        参数：
        kb_name (str): 知识库名称。
        """
        if cache := self.get(kb_name):
            self.pop(kb_name)
            logger.info(f"成功释放向量库：{kb_name}")


class KBFaissPool(_FaissPool):
    """
    知识库 FAISS 池，用于管理知识库的 FAISS 向量存储。
    """
    async def load_vector_store(
            self,
            kb_name: str,
            vector_name: str = None,
            create: bool = True,
            embed_model: str = EMBEDDING_MODEL,
            embed_device: str = embedding_device(),
    ) -> ThreadSafeFaiss:
        """
          加载或创建知识库的 FAISS 向量存储。

          参数：
          kb_name (str): 知识库名称。
          vector_name (str): 向量存储名称。
          create (bool): 如果向量存储不存在，是否创建新的。默认值为 True。
          embed_model (str): 嵌入模型名称，默认值为 EMBEDDING_MODEL。
          embed_device (str): 嵌入设备，默认值为 embedding_device。

          返回：
          ThreadSafeFaiss: 线程安全的 FAISS 向量存储。
          """
        self.atomic.acquire()
        vector_name = vector_name or embed_model
        cache = self.get((kb_name, vector_name)) # 用元组比拼接字符串好一些
        if cache is None:
            item = ThreadSafeFaiss((kb_name, vector_name), pool=self)
            self.set((kb_name, vector_name), item)
            with item.acquire(msg="初始化"):
                self.atomic.release()
                logger.info(f"loading vector store in '{kb_name}/vector_store/{vector_name}' from disk.")
                vs_path = get_vs_path(kb_name, vector_name)

                # index.faiss：这个文件包含了FAISS库创建的向量索引，存储了所有向量的数据结构，
                # 用于快速最近邻搜索。它是一个二进制文件，因为FAISS索引无法通过标准的pickle序列化。
                if os.path.isfile(os.path.join(vs_path, "index.faiss")):

                # index.pkl：这个文件通常包含了与向量相关的元数据（如文档存储 docstore 和索引到文档存储ID的映射 index_to_docstore_id）。
                # 这些数据可以通过pickle进行序列化和反序列化。
                # docstore 存储了实际的文本或文档，而 index_to_docstore_id 维护了FAISS索引中向量与这些文档或文本之间的映射关系。
                    # 这里要加载Embedding 模型
                    embeddings = await self.load_kb_embeddings(kb_name=kb_name, embed_device=embed_device, default_embed_model=embed_model)

                    # https://api.python.langchain.com/en/latest/_modules/langchain_community/vectorstores/faiss.html#FAISS
                    # load_local 方法会检查本地是否存在指定的文件，如果存在，它会将这些文件加载到内存中，并返回一个加载好的 FAISS 实例
                    vector_store = FAISS.load_local(vs_path, embeddings, distance_strategy="METRIC_INNER_PRODUCT", allow_dangerous_deserialization=True)
                elif create:
                    # create an empty vector store
                    if not os.path.exists(vs_path):
                        os.makedirs(vs_path)
                    vector_store = self.new_vector_store(embed_model=embed_model, embed_device=embed_device)
                    vector_store.save_local(vs_path)
                else:
                    raise RuntimeError(f"knowledge base {kb_name} not exist.")
                item.obj = vector_store
                item.finish_loading()
        else:
            self.atomic.release()
        return self.get((kb_name, vector_name))


class MemoFaissPool(_FaissPool):
    """
    内存 FAISS 池，用于管理内存中的 FAISS 向量存储。
    """
    def load_vector_store(
        self,
        kb_name: str,
        embed_model: str = EMBEDDING_MODEL,
        embed_device: str = embedding_device(),
    ) -> ThreadSafeFaiss:
        """
        加载或创建内存中的 FAISS 向量存储。

        参数：
        kb_name (str): 知识库名称。
        embed_model (str): 嵌入模型名称，默认值为 EMBEDDING_MODEL。
        embed_device (str): 嵌入设备，默认值为 embedding_device。

        返回：
        ThreadSafeFaiss: 线程安全的 FAISS 向量存储。
        """
        self.atomic.acquire()
        cache = self.get(kb_name)
        if cache is None:
            item = ThreadSafeFaiss(kb_name, pool=self)
            self.set(kb_name, item)
            with item.acquire(msg="初始化"):
                self.atomic.release()
                logger.info(f"loading vector store in '{kb_name}' to memory.")
                # create an empty vector store
                vector_store = self.new_vector_store(embed_model=embed_model, embed_device=embed_device)
                item.obj = vector_store
                item.finish_loading()
        else:
            self.atomic.release()
        return self.get(kb_name)


kb_faiss_pool = KBFaissPool(cache_num=CACHED_VS_NUM)
memo_faiss_pool = MemoFaissPool(cache_num=CACHED_MEMO_VS_NUM)


if __name__ == "__main__":
    import time, random
    from pprint import pprint

    kb_names = ["vs1", "vs2", "vs3"]
    # for name in kb_names:
    #     memo_faiss_pool.load_vector_store(name)

    def worker(vs_name: str, name: str):
        vs_name = "samples"
        time.sleep(random.randint(1, 5))
        embeddings = load_local_embeddings()
        r = random.randint(1, 3)

        with kb_faiss_pool.load_vector_store(vs_name).acquire(name) as vs:
            if r == 1: # add docs
                ids = vs.add_texts([f"text added by {name}"], embeddings=embeddings)
                pprint(ids)
            elif r == 2: # search docs
                docs = vs.similarity_search_with_score(f"{name}", k=3, score_threshold=1.0)
                pprint(docs)
        if r == 3: # delete docs
            logger.warning(f"清除 {vs_name} by {name}")
            kb_faiss_pool.get(vs_name).clear()

    threads = []
    for n in range(1, 30):
        t = threading.Thread(target=worker,
                             kwargs={"vs_name": random.choice(kb_names), "name": f"worker {n}"},
                             daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
