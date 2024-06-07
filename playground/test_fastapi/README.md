# 注意：FastAPI 测试相关的内容：https://fastapi.tiangolo.com/tutorial/testing/
       FastAPI Http的内容：https://fastapi.tiangolo.com/deployment/https/
       HttpS的基础知识：https://howhttps.works/

# FastAPI 官网：https://github.com/tiangolo/fastapi

FastAPI 是一个现代、快速（高性能）的 Web 框架，用于基于标准 Python 类型提示使用 Python 构建 API。
FastAPI 使用称为 ASGI 的标准来构建 Python Web 框架和服务器。 FastAPI 是一个 ASGI Web 框架。
在远程服务器计算机上运行 FastAPI 应用程序（或任何其他 ASGI 应用程序）所需的主要内容是像 Uvicorn 这样的 ASGI 服务器程序，这是 fastapi 命令中默认提供的程序。

# 一、使用

- pip install fastapi


# 二、关键的启动信息：
1. 当安装 FastAPI（例如使用 pip install fastapi ）时，它包含一个名为 fastapi-cli 的包，该包在终端中提供 fastapi 命令。
2. 开发模式：使用 fastapi dev 会读取 指定的启动文件，检测其中的 FastAPI 应用程序，并使用 Uvicorn 启动服务器。
   - 默认情况下，它将启用自动重新加载，因此当更改代码时，它将自动重新加载服务器。这是资源密集型的。 
3. 生产模式：使用 fastapi run 代替。
   - 默认情况下，它将禁用自动重新加载。

4. Starlette 是一个轻量级的 ASGI 框架，专注于构建异步 Web 应用程序和服务。ASGI（Asynchronous Server Gateway Interface）是 WSGI（Web Server Gateway Interface）的异步版本，旨在支持异步编程。
5. Pydantic 是一个用于数据验证和数据模型管理的 Python 库。它提供了强大的数据解析和验证功能，可以帮助开发者轻松处理和验证复杂的数据结构。Pydantic 主要用于确保数据符合预期的格式和类型，并提供自动的类型转换和错误处理机制


# 三、如何部署FastAPI

对于 Web API，通常涉及将其放置在远程计算机中，并使用提供良好性能、稳定性等的服务器程序， 以便用户可以有效地访问应用程序，而不会出现中断或问题。
这与开发阶段形成鲜明对比，在开发阶段，我们是不断更改代码、破坏代码并修复代码、停止并重新启动开发服务器等。

具体步骤：
- 第一件事是将正在使用的 FastAPI 版本“固定”，使其适用于开发的应用程序的特定最新版本。
- 当安装 FastAPI 时，它会附带一个生产服务器 Uvicorn，可以使用 fastapi run 命令启动它。
  - 如果是手动安装 ASGI 服务器，通常需要传递特殊格式的导入字符串才能导入 FastAPI 应用程序：uvicorn main:app --host 0.0.0.0 --port 80


# 四、多进程

概念地址：https://fastapi.tiangolo.com/deployment/server-workers/
Gunicorn： https://www.uvicorn.org/