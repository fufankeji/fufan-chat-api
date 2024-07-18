from typing import List, Dict, Optional

from langchain.schema import Document
from langchain_community.vectorstores import Milvus
import os
import uuid
from configs import kbs_config
from server.db.repository.knowledge_file_repository import list_file_num_docs_id_by_kb_name_and_file_name

from server.knowledge_base.kb_service.base import KBService, SupportedVSType, EmbeddingsFunAdapter, \
    score_threshold_process
from server.knowledge_base.utils import KnowledgeFile


class MilvusKBService(KBService):
    """
    该类继承自KBService, 用于提供Milvus向量数据库服务
    """
    # 用于存储Milvus数据库连接实例的属性。
    # 使用LangChain 的 milvus 的实例：https://api.python.langchain.com/en/v0.1/_modules/langchain_community/vectorstores/milvus.html#Milvus
    milvus: Milvus

    # 静态方法，用于从Milvus数据库中获取一个集合。
    @staticmethod
    def get_collection(milvus_name):
        from pymilvus import Collection
        # 根据提供的集合名称，返回Milvus数据库中的集合对象。
        return Collection(milvus_name)

    def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
        # 初始化一个空列表来存储查询结果
        result = []
        # 检查是否已经连接到Milvus的集合
        if self.milvus.col:
            # 如果需要，可以将字符串ID转换为整数ID（注释掉的代码，视Milvus版本需求）
            # ids = [int(id) for id in ids]  # for milvus if needed #pr 2725
            # 使用Milvus的查询功能，根据ID列表获取对应的文档
            data_list = self.milvus.col.query(expr=f'pk in {[int(_id) for _id in ids]}', output_fields=["*"])
            # 遍历查询结果
            for data in data_list:
                # 从数据中移除并获取文本内容
                text = data.pop("text")
                # 创建Document对象，并添加到结果列表中
                result.append(Document(page_content=text, metadata=data))
        # 返回包含Document对象的列表
        return result

    def del_doc_by_ids(self, ids: List[str]) -> bool:
        # 使用Milvus的删除功能，根据ID列表删除对应的文档
        self.milvus.col.delete(expr=f'pk in {ids}')

    @staticmethod
    def search(milvus_name, content, limit=3):
        # 定义Milvus搜索的参数
        search_params = {
            "metric_type": "L2",  # 使用L2距离作为度量类型
            "params": {"nprobe": 10},  # 设置搜索参数nprobe
        }
        # 从Milvus获取指定名称的集合
        c = MilvusKBService.get_collection(milvus_name)
        # 在集合中进行搜索，返回搜索结果
        return c.search(content, "embeddings", search_params, limit=limit, output_fields=["content"])

    def do_create_kb(self):
        # 此方法用于创建知识库，目前为空，需要实现具体逻辑
        pass

    def vs_type(self) -> str:
        # 返回此服务支持的知识库类型，这里是Milvus
        return SupportedVSType.MILVUS

    def _load_milvus(self):
        # 加载Milvus配置，并创建Milvus实例
        self.milvus = Milvus(embedding_function=EmbeddingsFunAdapter(self.embed_model),
                             collection_name=self.kb_name,
                             connection_args=kbs_config.get("milvus"),
                             index_params=kbs_config.get("milvus_kwargs")["index_params"],
                             search_params=kbs_config.get("milvus_kwargs")["search_params"],

                             auto_id=True  # 由Milvus自动为每个文档生成唯一的ID, 新版本LangChain的更改
                             )

    def do_init(self):
        self._load_milvus()

    def do_drop_kb(self):
        if self.milvus.col:
            self.milvus.col.release()
            self.milvus.col.drop()

    async def do_search(self, query: str, top_k: int, score_threshold: float):
        # 加载milvus 实例
        self._load_milvus()

        # 实例化Embedding模型
        embed_func = EmbeddingsFunAdapter(self.embed_model)

        # 将用户传入的问题，通过Embedding 模型转化为 Embedding向量
        embeddings = embed_func.embed_query(query)

        # 执行相似性搜索
        docs = self.milvus.similarity_search_with_score_by_vector(embeddings, top_k)
        return score_threshold_process(score_threshold, top_k, docs)

    async def do_add_doc(self, docs: List[Document], **kwargs) -> List[Dict]:

        for doc in docs:
            # 遍历文档的元数据字典，将每个值转换为字符串类型
            for k, v in doc.metadata.items():
                doc.metadata[k] = str(v)
            # # 确保每个Milvus字段都存在于文档的元数据中，如果不存在则设置为空字符串
            # for field in self.milvus.fields:
            #     doc.metadata.setdefault(field, "")
            # doc.metadata.pop(self.milvus._text_field, None)
            # doc.metadata.pop(self.milvus._vector_field, None)
        print("-----------------------------")
        print(docs)
        # 这里是 milvus 实例继承自 LangChain的 VectorStore 基类 中的 add_documents 方法
        # https://api.python.langchain.com/en/v0.1/_modules/langchain_core/vectorstores.html#VectorStore
        ids = self.milvus.add_documents(docs)
        doc_infos = [{"id": id, "metadata": doc.metadata} for id, doc in zip(ids, docs)]
        return doc_infos

    def do_add_file(self, docs: List[Document], **kwargs) -> List[Dict]:
        self.milvus.add_texts(docs)
        return ""

    async def do_delete_doc(self, kb_file: KnowledgeFile, **kwargs):
        id_list = await list_file_num_docs_id_by_kb_name_and_file_name(kb_file.kb_name, kb_file.filename)
        if self.milvus.col:
            self.milvus.col.delete(expr=f'pk in {id_list}')

        # if self.milvus.col:
        #     file_path = kb_file.filepath.replace("\\", "\\\\")
        #     file_name = os.path.basename(file_path)
        #     id_list = [item.get("pk") for item in
        #                self.milvus.col.query(expr=f'source == "{file_name}"', output_fields=["pk"])]
        #     self.milvus.col.delete(expr=f'pk in {id_list}')

    def do_clear_vs(self):
        if self.milvus.col:
            self.do_drop_kb()
            self.do_init()


# 主调用代码需要被包装在一个异步函数中
async def main():
    # 创建一个 milvus 对象, 用于测试milvus的连通性
    milvusService = MilvusKBService("milvus_test")
    print(f"milvus_kb_service: {milvusService}")

    from server.knowledge_base.kb_service.base import KBServiceFactory
    kb = await KBServiceFactory.get_service_by_name("milvus_test")

    # 如果想要使用的向量数据库的collecting name 不存在，则进行创建
    if kb is None:
        from server.db.repository.knowledge_base_repository import add_kb_to_db

        # 先在Mysql中创建向量数据库的基本信息
        await add_kb_to_db(kb_name="milvus_test",
                           kb_info="milvus",
                           vs_type="milvus",
                           embed_model="bge-large-zh-v1.5",
                           user_id="ff4f9954-da8c-492f-965d-dc18224b1176")

    # 调用 add_doc 方法添加一个名为 "README.md" 的文桗，确保使用 await
    await milvusService.add_doc(KnowledgeFile("README.md", "milvus_test"))
    # 根据输入进行检索
    search_ans = await milvusService.search_docs(query="RAG增强可以使用的框架？")
    print(search_ans)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
