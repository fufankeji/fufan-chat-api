# 使用基础镜像
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 设定维护者标签
LABEL maintainer=fufan_chat

# 设置工作目录和环境变量，其中/opt/conda/bin 是 Miniconda安装后，其可执行文件所在的目录。
ENV HOME=/fufan-chat-api \
    PATH=/opt/conda/bin:$PATH

# 设置工作目录
WORKDIR /fufan-chat-api

# 安装Miniconda
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends curl libgl1 libglib2.0-0 jq && \
    curl -sLo /miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 配置时区
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone

# 环境变量，确保使用的是conda环境中的Python
ENV PATH=/opt/conda/bin:$PATH

# 复制requirements.txt并安装依赖
COPY requirements.txt /fufan-chat-api/
WORKDIR /fufan-chat-api

# 使用conda创建虚拟环境并安装依赖
# 预先复制 requirements.txt 并安装 Python 依赖
COPY requirements.txt .
RUN conda create -n fufan_chat_api python=3.10 && \
    echo "source activate fufan_chat_api" > ~/.bashrc && \
    /opt/conda/envs/fufan_chat_api/bin/pip install -r requirements.txt

# 复制其他所有文件
COPY . .

# 暴露端口，记录容器侦听的端口
EXPOSE 8000

# 容器启动时执行的命令
ENTRYPOINT ["python", "startup.py"]
