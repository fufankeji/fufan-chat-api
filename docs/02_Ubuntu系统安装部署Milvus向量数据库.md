# <center>Ubuntu 系统安装部署Milvus向量数据库

&emsp;&emsp;Milvus 是一个开源的向量数据库项目，由 ZILLIZ 创建和开源。它旨在为大规模向量数据的存储和检索提供高效的解决方案，特别是在 AI 和大数据应用中。官方github：https://github.com/milvus-io/milvus

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240703135903649.png" width=70%></div>

&emsp;&emsp;Milvus作为一个开源的向量相似度搜索引擎，专注于大规模向量数据的快速相似度搜索。它提供了高效的向量索引和搜索功能，支持多种向量数据类型和查询方式。作为一个专门设计用于处理输入向量查询的数据库，Milvus能够在万亿规模上对向量进行索引。与现有的关系数据库主要按照预定义的模式处理结构化数据不同，Milvus是从自底向上设计的，以处理从非结构化数据转换而来的嵌入向量。
        
&emsp;&emsp;Faiss的索引和量化算法我们在课程前期已经给大家详细的介绍过在问答系统的接入方法，Milvus和Faiss的实现非常相似，只是可维护更强、可以做到分布式部署。越来越多的Faiss项目正在往Milvus迁移。

&emsp;&emsp;Milvus官方网站：https://milvus.io/docs

- **Step 1. 安装Docker**

&emsp;&emsp;Milvus 运行在 Docker 容器中，所以需要先安装 Docker 和 Docker Compose，依次执行下述命令即可：

```bash

   # 更新Ubuntu软件包列表和已安装软件的版本:
   sudo apt update

   # 安装Ubuntu系统的依赖包
   sudo apt-get install ca-certificates curl gnupg lsb-release

   # 添加Docker官方的GPG密钥
   curl -fsSL http://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo apt-key add -

   # 安装Docker
   sudo apt-get install docker-ce docker-ce-cli containerd.io

   # 启动Docker服务
   sudo systemctl start docker
   sudo systemctl status docker
```

&emsp;&emsp;安装成功后，会如下所示：

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240704130557014.png" width=70%></div>

&emsp;&emsp;安装Docker Compose

&emsp;&emsp;Docker Compose是一个强大的工具，可以帮助开发人员更高效地管理和部署复杂的多容器Docker应用程序。我们可以通过编写一个YAML文件来定义应用程序的服务、网络和卷，然后使用一条命令启动整个应用程序。这样可以避免手动管理每个容器的启动和连接，简化了开发和部署流程。


```bash
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose
```

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240704130711924.png" width=100%></div>

&emsp;&emsp;进入Milvus官方文档：https://milvus.io/docs

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240704130822643.png" width=70%></div>

&emsp;&emsp;下载YAML文件，并启动Milvus服务，如下所示：

```bash
    wget https://github.com/milvus-io/milvus/releases/download/v2.2.16/milvus-standalone-docker-compose.yml -O docker-compose.yml
    sudo docker-compose up -d


```

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240703140558978.png" width=100%></div>

&emsp;&emsp;在 Docker 和 Docker Compose 环境中，可以使用如下命令查看当前正在运行的服务或容器。

```bash
sudo docker ps
```

&emsp;&emsp;这个命令将列出所有正在运行的容器，包括容器 ID、创建时间、使用的镜像、命令、状态、端口等信息。

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240703140931729.png" width=100%></div>

&emsp;&emsp;启动milvus服务后，该开源项目还提供了一个可视化的管理工具：Attu。地址为：https://milvus.io/docs/v2.0.x/attu.md 。 兼容性好，就是要注意版本对应的问题。我们可以直接在当前的环境下执行如下命令：

```bash
    docker run -p 9002:3000 -e MILVUS_URL=[你的服务ip]:19530 zilliz/attu:v2.3.7
```

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240703141129011.png" width=100%></div>

&emsp;&emsp;访问地址：http://192.168.110.131:8000/?#/connect (注意：这个地址需要替换为自己的实际服务启动IP)

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240703141346314.png" width=70%></div>

&emsp;&emsp;进入后管理工具页面如下图所示：

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240704133634233.png" width=70%></div>
