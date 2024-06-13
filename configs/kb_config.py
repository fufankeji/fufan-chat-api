# 更新以下字段为你本地数据库的实际用户名、密码和数据库名
username = 'root'
hostname = '192.168.110.131'
database_name = 'fufanapi'
password = "snowball950123"

SQLALCHEMY_DATABASE_URI = f"mysql+asyncmy://{username}:{password}@{hostname}/{database_name}?charset=utf8mb4"


# 默认使用的知识库
DEFAULT_KNOWLEDGE_BASE = ""

DEFAULT_VS_TYPE = "faiss"

# 知识库匹配向量数量
VECTOR_SEARCH_TOP_K = 3

SCORE_THRESHOLD = 1.0