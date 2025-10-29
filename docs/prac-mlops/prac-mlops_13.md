# 附录 A. 关键术语

作者：Noah Gift

本节包含经常出现在教授云计算、MLOps 和机器学习工程中的选择性关键术语：

告警

告警是与之相关联有操作的健康度量。例如，当网络服务返回多个错误状态码时，可能会触发发送文本消息到软件工程师的告警。

Amazon ECR

Amazon ECR 是存储 Docker 格式容器的容器注册表。

Amazon EKS

Amazon EKS 是由亚马逊创建的托管 Kubernetes 服务。

自动缩放

自动缩放是根据节点使用的资源量自动缩放负载的过程。

AWS Cloud9

AWS Cloud9 是在 AWS 运行的基于云的开发环境。它具有特殊的钩子，用于开发无服务器应用程序。

AWS Lambda

由 AWS 提供的无服务器计算平台，具有 FaaS 能力。

Azure 容器实例（ACI）

Azure 容器实例是 Microsoft 提供的托管服务，允许您运行容器镜像而无需管理托管它们的服务器。

Azure Kubernetes Service（AKS）

Azure Kubernetes Service 是由 Microsoft 创建的托管 Kubernetes 服务。

`black`

`black` 工具能够自动格式化 Python 源代码的文本。

构建服务器

构建服务器是在软件测试和部署中都起作用的应用程序。流行的构建服务器可以是 SaaS 或开源的。以下是一些流行的选择：

+   [Jenkins](https://jenkins.io) 是一个开源构建服务器，可以在 AWS、GCP、Azure 或 Docker 容器上运行，也可以在您的笔记本电脑上运行。

+   [CircleCI](https://circleci.com) 是一个 SaaS 构建服务，可以与像 GitHub 这样的流行 Git 托管提供商集成。

CircleCI

一个在 DevOps 工作流中使用的流行的 SaaS（软件即服务）构建系统。

云原生应用

云原生应用是利用云的独特能力（如无服务器）的服务。

容器

一个容器是与操作系统的其余部分隔离的一组进程。它们通常是几兆字节大小。

持续交付

持续交付是将经过测试的软件自动交付到任何环境的过程。

持续集成

持续集成是在提交到源代码控制系统后自动测试软件的过程。

数据工程

数据工程是自动化数据流的过程。

灾难恢复

灾难恢复是设计软件系统以在灾难中恢复的过程。此过程可能包括将数据归档到另一个位置。

Docker 格式容器

容器有多种格式。一种新兴的形式是 Docker，它涉及定义*Dockerfile*。

Docker

Docker 是一家创建容器技术的公司，包括执行引擎、通过 DockerHub 的协作平台以及名为*Dockerfile*的容器格式。

FaaS（函数即服务）

一种云计算类型，促进响应事件的函数。

Google GKE

Google GKE 是由 Google 创建的托管 Kubernetes 服务。

IPython

`ipython` 解释器是 Python 的交互式终端。它是 Jupyter 笔记本的核心。

JSON

JSON 代表 JavaScript 对象表示法，它是一种轻量级、人类可读的数据格式，在 Web 服务中被广泛使用。

Kubernetes 集群

Kubernetes 集群是部署 Kubernetes 的一个实例，包括节点、Pod、API 和容器的整个生态系统。

Kubernetes 容器

Kubernetes 容器是部署到 Kubernetes 集群中的 Docker 镜像。

Kubernetes pod

Kubernetes Pod 是一个包含一个或多个容器的组。

Kubernetes

Kubernetes 是一个用于自动化容器化应用操作的开源系统。谷歌在 2014 年创建并开源了它。

负载测试

负载测试是验证软件系统规模特性的过程。

Locust

Locust 是一个接受 Python 格式的负载测试场景的负载测试框架。

记录

记录是创建关于软件应用运行状态的消息的过程。

Makefile

`Makefile` 是包含用于构建软件的一组指令的文件。大多数 Unix 和 Linux 操作系统都内建支持这种文件格式。

指标

指标是为软件应用创建关键绩效指标（KPI）的过程。一个参数的示例是服务器使用的 CPU 百分比。

微服务

微服务是一个轻量级、松耦合的服务。它可以小到一个函数的大小。

迁移

迁移是将应用程序从一个环境迁移到另一个环境的能力。

Moore 定律

对于一段时间来看，微芯片上的晶体管数量每两年翻倍一次。

运维化

使应用程序准备好进行生产部署的过程。这些操作可能包括监控、负载测试和设置警报。

pip

`pip` 工具用于安装 Python 包。

端口

端口是网络通信端点。一个端口的示例是通过 HTTP 协议在端口 80 上运行的 Web 服务。

Prometheus

Prometheus 是一个带有高效时间序列数据库的开源监控系统。

pylint

`pylint` 工具检查 Python 源代码的语法错误。

PyPI

Python 包索引，发布的包可供工具如 `pip` 安装。

pytest

`pytest` 工具是用于在 Python 源代码上运行测试的框架。

Python 虚拟环境

Python 虚拟环境通过将 Python 解释器隔离到一个目录并在该目录中安装包来创建。Python 解释器可以通过 `python -m venv yournewenv` 执行此操作。

无服务器

无服务器是基于函数和事件构建应用程序的技术。

SQS 队列

由亚马逊构建的具有近无限读写能力的分布式消息队列。

Swagger

Swagger 工具是一个简化 API 文档创建的开源框架。

虚拟机

虚拟机是物理操作系统的仿真。它可能有几 GB 大小。

**YAML**

YAML 是一种人类可读的序列化格式，通常用于配置系统。它很容易转换成 JSON 格式。
