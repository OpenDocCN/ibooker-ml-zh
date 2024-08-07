# 序言

欢迎阅读《*使用 Spark 扩展机器学习：MLlib、TensorFlow 和 PyTorch 的分布式机器学习*》。本书旨在指导您在学习更多关于机器学习系统的过程中。Apache Spark 目前是大规模数据处理的最流行框架。它有许多 API 在 Python、Java 和 Scala 中实现，并被 Netflix、Microsoft 和 Apple 等许多大公司使用。PyTorch 和 TensorFlow 是最流行的机器学习框架之一。结合这些工具，这些工具已经在许多组织中得到使用，让您可以充分利用它们的优势。

不过，在我们开始之前，也许您想知道为什么我决定写这本书。好问题。有两个原因。第一个是通过分享我在过去十年中作为机器学习算法研究员积累的知识、经验和专业知识，来支持机器学习生态系统和社区。我大部分职业生涯都在作为数据基础设施工程师工作，为大规模数据分析构建基础设施，包括各种格式、类型和模式等，整合从客户、社区成员和同事那里收集到的知识，他们在头脑风暴和开发解决方案时分享了他们的经验。我们的行业可以利用这样的知识以更快的速度推动自己前进，通过利用他人的专业知识。虽然这本书的内容并不是所有人都适用，但大部分将为各种从业者提供新的方法。

这使我想到我写这本书的第二个原因：我想提供一个全面的方法来构建端到端可扩展的机器学习解决方案，超越传统方法。今天，许多解决方案都是根据组织特定需求和具体业务目标定制的。这很可能会继续成为未来多年的行业标准。在这本书中，我旨在挑战现状，激发更多创意解决方案，并解释多种方法和工具的利弊，使您能够利用组织中使用的任何工具，并获得最佳效果。我的总体目标是让数据和机器学习实践者更简单地合作，并更好地理解彼此。

# 谁应该阅读这本书？

本书适用于具有先前行业经验的机器学习实践者，他们希望了解 Apache Spark 的 MLlib 并增加对整个系统和流程的理解。数据科学家和机器学习工程师会特别感兴趣，但 MLOps 工程师、软件工程师以及任何对学习或构建分布式机器学习模型和使用 MLlib、分布式 PyTorch 和 TensorFlow 构建流水线感兴趣的人也会发现价值。理解机器学习工作的高级概念，并希望深入技术方面的技术人员也应该会对本书感兴趣且易于理解。

# 你是否需要分布式机器学习？

和所有好东西一样，这取决于情况。如果你有适合机器内存的小数据集，答案是否定的。如果你将来需要扩展你的代码并确保可以在不适合单台机器内存的更大数据集上训练模型，那么答案就是肯定的。

通常最好在整个软件开发生命周期中使用相同的工具，从本地开发环境到暂存和生产环境。但请注意，这也引入了管理分布式系统的其他复杂性，这通常将由组织中的不同团队处理。与您的同事合作时，共享一个通用的语言是一个好主意。

此外，今天创建机器学习模型的人们面临的最大挑战之一是将其从本地开发移至生产环境。我们中的许多人会犯“意大利面代码”的错误，这些代码应该是可重现的，但通常并非如此，并且很难进行维护和协作。在讨论实验生命周期管理的一部分中，我将涉及该主题。

# 本书导航

本书旨在从前几章的基础信息开始构建，涵盖使用 Apache Spark 和 PySpark 进行机器学习工作流程以及使用 MLflow 管理机器学习实验生命周期，最后进入到第 7、8 和 9 章，介绍专门的机器学习平台。本书以部署模式、推断和生产环境中的模型监控结束。以下是每章内容的详细介绍：

第一章，“分布式机器学习术语和概念”

本章介绍了机器学习的高级概述，并涵盖了与分布式计算和网络拓扑相关的术语和概念。我将带你深入各种概念和术语，为后续章节打下坚实的基础。

第二章，“Spark 和 PySpark 简介”

本章的目标是让您快速掌握 Spark 及其 Python 库 PySpark。我们将讨论术语、软件抽象及更多内容。

第三章，“使用 MLflow 管理机器学习实验生命周期”

本章介绍了 MLflow，这是一个管理机器学习生命周期的平台。我们将讨论什么是机器学习实验，以及为什么管理其生命周期如此重要，还将审视 MLflow 的各种组件，使这一切成为可能。

第四章，“数据摄取、预处理和描述性统计”

接下来，我们将深入研究数据处理。在这一章中，我将讨论如何使用 Spark 摄取您的数据，执行基本预处理（以图像文件为例），并对数据有所了解。我还将介绍如何通过利用 PySpark API 来避免所谓的小文件问题。

第五章，“特征工程”

在完成前一章的步骤后，您将准备好为训练机器学习模型使用的特征进行工程化。本章详细解释了特征工程是什么，涵盖了各种类型，并展示了如何利用 Spark 的功能提取特征。我们还将探讨何时以及如何使用`applyInPandas`和`pandas_udf`来优化性能。

第六章，“使用 Spark MLlib 训练模型”

本章将带您了解如何使用 MLlib 训练模型，评估和构建管道以复现模型，并最终将其持久化到磁盘。

第七章，“连接 Spark 与深度学习框架”

本章详细讲解如何构建一个数据系统，将 Spark 的强大能力与深度学习框架结合起来。讨论了连接 Spark 和深度学习集群，并介绍了 Petastorm、Horovod 以及 Spark 计划中的 Project Hydrogen。

第八章，“TensorFlow 分布式机器学习方法”

在这里，我将带您逐步示例使用分布式 TensorFlow——特别是`tf.keras`——同时利用您在 Spark 中完成的预处理工作。您还将了解有关扩展机器学习的各种 TensorFlow 模式和支持其的组件架构。

第九章，“PyTorch 分布式机器学习方法”

本章涵盖了 PyTorch 的扩展机器学习方法，包括其内部架构。我们将逐步演示如何使用分布式 PyTorch，同时利用您在前几章中与 Spark 完成的预处理工作。

第十章，“机器学习模型部署模式”

在本章中，我介绍了我们可以使用的各种部署模式，包括使用 Spark 和 MLflow 进行批处理和流式推理，并提供了在 MLflow 中使用`pyfunc`功能的示例，该功能允许我们部署几乎任何机器学习模型。本章还涵盖了监控和分阶段实施生产机器学习系统。

# 未涵盖的内容

有许多方法可以进行分布式机器学习。一些方法涉及并行运行多个实验，使用多个超参数，在已加载到内存中的数据上。您可能能够将数据集加载到单台机器的内存中，或者数据集可能太大，必须分区到多台机器上。我们将简要讨论网格搜索，一种用于查找一组超参数最优值的技术，但本书仅限于此。

本书不涵盖以下话题：

机器学习算法简介

有许多精彩的书籍深入探讨了各种机器学习算法及其用途，本书不会重复它们。

将模型部署到移动设备或嵌入式设备

这通常需要使用 TinyML 和专用算法来缩小最终模型的大小（最初可能是从大型数据集创建的）。

TinyML

TinyML 专注于构建相对较小的机器学习模型，这些模型可以在资源受限的设备上运行。要了解更多，请查看[*TinyML*](https://oreil.ly/tinyML)，作者是彼得·沃登和丹尼尔·西图纳亚克（O'Reilly）。

在线学习

当数据随时间变化或机器学习算法需要动态适应数据中的新模式时，使用在线学习。在整个数据集上进行训练是计算上不可行的时候，需要使用外存算法。这是一种用于专业应用的机器学习基本不同的方法，本书未涵盖此内容。

并行实验

尽管本书讨论的工具，如 PyTorch 和 TensorFlow，使我们能够进行并行实验，本书将专注于并行数据训练，其中逻辑保持不变，每台机器处理不同的数据块。

这不是一个详尽的列表——因为所有的途径都以某种方式导向分布式，我可能忘记在这里提及一些话题，或者自写作以来行业中新的话题可能已经开始受到关注。如前所述，我的目标是分享我的观点，基于我在机器学习领域积累的经验和知识，为其他人提供一种全面的方法来应用于他们自己的努力中；我的意图是尽可能涵盖尽可能多的关键点，为提供一个基础，并鼓励您进一步探索，以加深对这些讨论话题的理解。

# 环境和工具

现在你已经了解了将要（和不会）涵盖的主题，接下来是设置你的教程环境的时间了。你将使用各种平台和库来开发一个机器学习管道，同时完成本书中的练习。

## **工具**

本节简要介绍了我们将用来构建本书中讨论的解决方案的工具。如果你对这些工具不熟悉，可能需要在开始之前查看它们的文档。为了在你自己的机器上实现书中的代码示例，你需要本地安装以下工具：

**Apache Spark**

一个通用的大规模数据处理分析引擎。

**PySpark**

Apache Spark 的 Python 接口。

**PyTorch**

一个由 Facebook 开发的机器学习框架，基于 Torch 库，用于计算机视觉和自然语言处理应用。我们将利用它的分布式训练能力。

**TensorFlow**

由 Google 开发的机器学习管道平台。我们将利用它的分布式训练能力。

**MLflow**

一个开源平台，用于管理机器学习生命周期。我们将用它来管理本书中的实验。

**Petastorm**

一个支持使用 Apache Parquet 格式数据集进行深度学习模型分布式训练和评估的库。Petastorm 支持 TensorFlow 和 PyTorch 等机器学习框架。我们将用它来在 Spark 和深度学习集群之间架起桥梁。

**Horovod**

一个用于 TensorFlow、Keras、PyTorch 和 Apache MXNet 的分布式训练框架。该项目旨在支持开发者将单 GPU 训练脚本扩展到多个 GPU 并行训练。我们将用它来优化多个 GPU 上的工作负载，并协调 Spark 集群与深度学习集群的分布式系统，这需要一个专用的分布式系统调度器来管理集群资源，并使它们通过相同的硬件协同工作。

**NumPy**

一个用于科学计算的 Python 库，可以高效地执行各种数组操作（数学、逻辑、形状操作、排序、选择、I/O 等）。我们将用它进行各种可以在单台机器上完成的统计和数学运算。

**PIL**

**Python Imaging Library**，也称为 [Pillow](https://oreil.ly/V2V2j)。我们将使用它来处理图像。

在当今的生态系统中，机器学习和分布式数据领域的新工具每天都在涌现。历史告诉我们，其中一些工具会持续存在，而另一些则不会。关注一下你工作场所中已经使用的工具，并尽可能挖掘它们的能力，然后再考虑引入新的工具。

## **数据集**

在本书的示例中，我们将在实际中利用现有的数据集，并在必要时生成专用的数据集以更好地传达信息。这里列出的数据集，全部可在 [Kaggle](https://www.kaggle.com) 上获取，并在附带的 [GitHub 存储库](https://oreil.ly/smls-git) 中包含：

Caltech 256 数据集

[Caltech 256](https://oreil.ly/Ns9uy) 是 [Caltech 101 数据集](https://oreil.ly/1jgcC) 的扩展，包含了 30,607 张属于 257 个类别的对象图片。这些类别极为多样，从网球鞋到斑马，有背景和无背景的图像，水平和垂直方向的图像。大多数类别约有 100 张图片，但有些类别多达 800 张。

CO[2] Emission by Vehicles 数据集

[CO[2] Emission by Vehicles 数据集](https://oreil.ly/akVrk) 基于加拿大政府开放数据网站七年的车辆 CO[2] 排放数据。数据集包含 7,385 行和 12 列（制造商、型号、变速器等，以及 CO[2] 排放和各种燃油消耗措施）。

Zoo Animal Classification 数据集

为了学习 MLlib 库中可用的统计函数，我们将使用 [Zoo Animal Classification 数据集](https://oreil.ly/lPqbv)。它包含 101 种动物，有 16 个布尔值属性用于描述它们。这些动物可以分为七类：哺乳动物、鸟类、爬行动物、鱼类、两栖动物、昆虫和无脊椎动物。我选择它是因为它有趣且相对简单易懂。

如果你正在本地计算机上完成教程，请使用书中 GitHub 存储库中提供的示例数据集。

# 本书使用的约定

以下是本书使用的排版约定：

*Italic*

表示新术语、URL、文件和目录名称以及文件扩展名。

`Constant width`

用于命令行输入/输出和代码示例，以及出现在文本中的代码元素，包括变量和函数名称、类和模块。

`*Constant width italic*`

显示要在代码示例和命令中用用户提供的值替换的文本。

`**Constant width bold**`

显示用户应按原样键入的命令或其他文本。

###### 提示

此元素表示提示或建议。

###### 注意

此元素表示一般说明。

###### 警告

此元素表示警告或注意事项。

# 使用代码示例

附加资料（代码示例、练习等）可在 [*https://oreil.ly/smls-git*](https://oreil.ly/smls-git) 下载。

这本书旨在帮助您完成工作。通常情况下，如果本书提供示例代码，您可以在您的程序和文档中使用它。除非您复制了代码的大部分内容，否则不需要联系我们请求许可。例如，编写使用本书中多个代码块的程序不需要许可。销售或分发包含 O’Reilly 书籍示例的 CD-ROM 需要许可。引用本书并引用示例代码来回答问题不需要许可。将本书中大量示例代码整合到您产品的文档中需要许可。

我们感谢但不要求署名。通常的署名包括书名、作者、出版商和 ISBN。例如：“*使用 Spark 扩展机器学习*，作者 Adi Polak。2023 年版权归 Adi Polak 所有，ISBN 978-1-098-10682-9。”

如果您认为您对代码示例的使用超出了合理使用范围或上述许可，请随时联系我们：permissions@oreilly.com。
# 致谢

这本书要感谢 Spark、数据工程和机器学习社区的支持，没有你们的帮助，这本技术书籍是无法问世的。真的，要让一本技术书籍成功出版，确实需要一个村庄的力量，因此非常感谢你们的帮助！

感谢所有早期读者和审阅者的帮助和建议：Holden Karau，Amitai Stern，Andy Petrella，Joe Reis，Laura Uzcátegui，Noah Gift，Kyle Gallatin，Parviz Deyhim，Sean Owen，Chitra Agastya，Kyle Hamilton，Terry McCann，Joseph Kambourakis，Marc Ramirez Invernon，Bartosz Konieczny，Beegee Alop 等许多其他人。

任何剩下的错误都是作者的责任，有时违背审阅者的建议。

最后，我要感谢我的生活伴侣，包容了我长时间的夜晚写作，早起，假期和周末。
