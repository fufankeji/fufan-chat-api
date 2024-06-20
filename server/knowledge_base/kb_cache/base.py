from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
import threading
from configs import (EMBEDDING_MODEL, CHUNK_SIZE,
                     logger, log_verbose)
from server.utils import embedding_device, get_model_path, list_online_embed_models
from contextlib import contextmanager
from collections import OrderedDict
from typing import List, Any, Union, Tuple


class ThreadSafeObject:
    """
    线程安全对象类，用于保证多线程环境下对象的安全访问。
    """
    def __init__(self, key: Union[str, Tuple], obj: Any = None, pool: "CachePool" = None):
        self._obj = obj
        self._key = key
        self._pool = pool
        self._lock = threading.RLock()
        self._loaded = threading.Event()

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"<{cls}: key: {self.key}, obj: {self._obj}>"

    @property
    def key(self):
        return self._key

    @contextmanager
    def acquire(self, owner: str = "", msg: str = "") -> FAISS:
        """
        获取对象的上下文管理器，用于在多线程环境下安全地访问对象。

        参数：
        owner (str): 操作的所有者，默认为当前线程。
        msg (str): 日志消息。

        返回：
        FAISS: 向量存储对象。
        """
        owner = owner or f"thread {threading.get_native_id()}"
        try:
            self._lock.acquire()
            if self._pool is not None:
                self._pool._cache.move_to_end(self.key)
            if log_verbose:
                logger.info(f"{owner} 开始操作：{self.key}。{msg}")
            yield self._obj
        finally:
            if log_verbose:
                logger.info(f"{owner} 结束操作：{self.key}。{msg}")
            self._lock.release()

    def start_loading(self):
        """
        开始加载对象。
        """
        self._loaded.clear()

    def finish_loading(self):
        """
        完成加载对象。
        """
        self._loaded.set()

    def wait_for_loading(self):
        """
        等待对象加载完成。
        """
        self._loaded.wait()

    @property
    def obj(self):
        return self._obj

    @obj.setter
    def obj(self, val: Any):
        self._obj = val


class CachePool:
    """
    缓存池类，用于管理线程安全对象的缓存。
    """
    def __init__(self, cache_num: int = -1):
        self._cache_num = cache_num
        self._cache = OrderedDict()
        self.atomic = threading.RLock()

    def keys(self) -> List[str]:
        return list(self._cache.keys())

    def _check_count(self):
        """
        检查缓存数量，并在超过限制时移除最旧的缓存。
        """
        if isinstance(self._cache_num, int) and self._cache_num > 0:
            while len(self._cache) > self._cache_num:
                self._cache.popitem(last=False)

    def get(self, key: str) -> ThreadSafeObject:
        """
        根据键获取缓存对象，并等待对象加载完成。

        参数：
        key (str): 缓存对象的键。

        返回：
        ThreadSafeObject: 线程安全对象。
        """
        if cache := self._cache.get(key):
            cache.wait_for_loading()
            return cache

    def set(self, key: str, obj: ThreadSafeObject) -> ThreadSafeObject:
        """
         设置缓存对象。

         参数：
         key (str): 缓存对象的键。
         obj (ThreadSafeObject): 线程安全对象。

         返回：
         ThreadSafeObject: 设置的线程安全对象。
        """
        self._cache[key] = obj
        self._check_count()
        return obj

    def pop(self, key: str = None) -> ThreadSafeObject:
        """
        移除并返回缓存对象。

        参数：
        key (str): 缓存对象的键，如果为空则移除最旧的缓存。

        返回：
        ThreadSafeObject: 移除的线程安全对象。
        """
        if key is None:
            return self._cache.popitem(last=False)
        else:
            return self._cache.pop(key, None)

    def acquire(self, key: Union[str, Tuple], owner: str = "", msg: str = ""):
        """
        获取缓存对象的上下文管理器，用于在多线程环境下安全地访问对象。

        参数：
        key (Union[str, Tuple]): 缓存对象的键。
        owner (str): 操作的所有者，默认为当前线程。
        msg (str): 日志消息。

        返回：
        上下文管理器: 缓存对象的上下文管理器。
        """
        cache = self.get(key)
        if cache is None:
            raise RuntimeError(f"请求的资源 {key} 不存在")
        elif isinstance(cache, ThreadSafeObject):
            self._cache.move_to_end(key)
            return cache.acquire(owner=owner, msg=msg)
        else:
            return cache

    async def load_kb_embeddings(
            self,
            kb_name: str,
            embed_device: str = embedding_device(),
            default_embed_model: str = EMBEDDING_MODEL,
    ) -> Embeddings:
        """
        加载知识库嵌入模型。

        参数：
        kb_name (str): 知识库名称。
        embed_device (str): 嵌入设备。
        default_embed_model (str): 默认嵌入模型。

        返回：
        Embeddings: 嵌入模型对象。
        """
        from server.db.repository.knowledge_base_repository import get_kb_detail
        from server.knowledge_base.kb_service.base import EmbeddingsFunAdapter

        kb_detail = await get_kb_detail(kb_name)
        embed_model = kb_detail.get("embed_model", default_embed_model)

        print(f"这是查询到的load_kb_embeddings的参数：{kb_detail}")
        if embed_model in list_online_embed_models():
            return EmbeddingsFunAdapter(embed_model)
        else:
            return embeddings_pool.load_embeddings(model=embed_model, device=embed_device)


class EmbeddingsPool(CachePool):
    """
    嵌入池类，用于管理和加载嵌入模型。
    """
    def load_embeddings(self, model: str = None, device: str = None) -> Embeddings:
        """
        加载嵌入模型。

        参数：
        model (str): 模型名称，默认为 EMBEDDING_MODEL。
        device (str): 设备，默认为 embedding_device。

        返回：
        Embeddings: 嵌入模型对象。
        """
        self.atomic.acquire()
        model = model or EMBEDDING_MODEL
        device = embedding_device()
        key = (model, device)
        if not self.get(key):
            item = ThreadSafeObject(key, pool=self)
            self.set(key, item)
            with item.acquire(msg="初始化"):
                self.atomic.release()
                if model == "text-embedding-ada-002":  # openai text-embedding-ada-002
                    from langchain.embeddings.openai import OpenAIEmbeddings
                    embeddings = OpenAIEmbeddings(model=model,
                                                  openai_api_key=get_model_path(model),
                                                  chunk_size=CHUNK_SIZE)
                elif 'bge-' in model:
                    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
                    if 'zh' in model:
                        # for chinese model
                        query_instruction = "为这个句子生成表示以用于检索相关文章："
                    elif 'en' in model:
                        # for english model
                        query_instruction = "Represent this sentence for searching relevant passages:"
                    else:
                        # maybe ReRanker or else, just use empty string instead
                        query_instruction = ""


                    embeddings = HuggingFaceBgeEmbeddings(model_name=get_model_path(model),
                                                          model_kwargs={'device': device},
                                                          query_instruction=query_instruction)
                    if model == "bge-large-zh-noinstruct":  # bge large -noinstruct embedding
                        embeddings.query_instruction = ""
                else:
                    from langchain.embeddings import HuggingFaceBgeEmbeddings
                    embeddings = HuggingFaceEmbeddings(model_name=get_model_path(model),
                                                       model_kwargs={'device': device})
                item.obj = embeddings
                item.finish_loading()
        else:
            self.atomic.release()
        return self.get(key).obj


embeddings_pool = EmbeddingsPool(cache_num=1)
