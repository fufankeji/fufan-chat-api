import pandas as pd


def load_data_and_generate_info(csv_file_path):
    # 读取 CSV 文件到 DataFrame
    df_course = pd.read_csv(
        csv_file_path,
        sep=',',
        header=0
    )

    # 定义需要合并的列的顺序
    concatenation_order = ["Course", "ModuleName", "Title", "Abstract"]

    # 生成 embedding_info 列，这个列包含 JSON-like 的信息
    embedding_info_list = df_course['embedding_info'] = df_course.apply(
        lambda row: '{ ' + ', '.join(
            f'"{col}": "{row[col]}"' for col in concatenation_order) + ' }',
        axis=1
    )

    # 返回 DataFrame 和 embedding_info 列的内容
    return df_course, embedding_info_list


if __name__ == '__main__':
    # 使用此函数
    csv_file_path = '/home/00_rag/fufan-chat-api/scripts/course_data/final_course.csv'  # 指定 CSV 文件路径
    df_course, embedding_info_list = load_data_and_generate_info(csv_file_path)

    # # 如果需要，这里可以打印或处理 DataFrame 和 embedding_info_list
    # print(df_course.head())  # 显示前几行 DataFrame 以检查输出
    # print(embedding_info_list.head())  # 显示前几行 embedding_info 列

    from server.knowledge_base.kb_service.base import KBServiceFactory
    from server.knowledge_base.kb_service.milvus_kb_service import MilvusKBService

    # 获取向量数据库的实例，用于接下来的数据存储。
    milvusService = MilvusKBService("recommend_system")

    # 添加文档到 milvus 服务
    milvusService.do_add_file(docs=embedding_info_list)
