# 第八章：使用 TensorFlow Serving 进行模型部署

在其他人可以使用您的模型进行预测之前，部署您的机器学习模型是最后一步。不幸的是，机器学习模型的部署在当今的数字世界分工思维中存在灰色地带。它不仅仅是 DevOps 的任务，因为它需要一些对模型架构及其硬件需求的了解。同时，部署机器学习模型有点超出了机器学习工程师和数据科学家的舒适区。他们对自己的模型了如指掌，但在部署机器学习模型方面往往遇到困难。在本章和接下来的章节中，我们希望弥合这一差距，指导数据科学家和 DevOps 工程师完成部署机器学习模型的步骤。图 8-1 显示了机器学习流水线中部署步骤的位置。

![](img/00026.jpg)

图 8-1. 作为 ML 流水线一部分的模型部署

机器学习模型可以通过三种主要方式部署：使用模型服务器、在用户的浏览器中或者在边缘设备上。今天最常见的部署机器学习模型的方式是使用模型服务器，本章我们将重点讨论这一点。请求预测的客户端将输入数据提交给模型服务器，然后接收预测结果。这要求客户端能够连接到模型服务器。

有些情况下，你不希望将输入数据提交到模型服务器（例如当输入数据很敏感或存在隐私问题时）。在这些情况下，你可以将机器学习模型部署到用户的浏览器上。例如，如果你想要确定一张图片是否包含敏感信息，你可以在上传到云服务器之前对图片的敏感级别进行分类。

然而，还有第三种模型部署方式：部署到边缘设备。有些情况下，你无法连接到模型服务器进行预测（例如远程传感器或物联网设备）。部署到边缘设备的应用数量正在增加，使其成为模型部署的一个有效选项。在第十章中，我们讨论了如何将 TensorFlow 模型转换为 TFLite 模型，以便在边缘设备上执行。

在本章中，我们突出了 TensorFlow 的 Serving 模块，这是通过模型服务器部署 TensorFlow 模型的一种简单且一致的方式。我们将介绍其设置并讨论高效的部署选项。这并不是部署深度学习模型的唯一方式；在本章末尾，我们还将讨论一些替代选项。

在我们深入探讨 TensorFlow Serving 之前，让我们从如何不应该设置模型服务器开始这一章节。

一个简单的模型服务器

大多数关于部署机器学习模型的介绍大致遵循相同的工作流程：

+   > > > > 使用 Python 创建 Web 应用程序（例如使用 Flask 或 Django）。
+   > > > > 
+   > > > > 在 Web 应用程序中创建一个 API 端点，正如我们在 示例 8-1 中展示的。
+   > > > > 
+   > > > > 加载模型结构及其权重。
+   > > > > 
+   > > > > 在加载的模型上调用 predict 方法。
+   > > > > 
+   > > > > 返回预测结果作为 HTTP 请求。

示例 8-1\. 使用 Flask 端点设置来推断模型预测的示例

`import``json``from``flask``import``Flask``,``request``from``tensorflow.keras.models``import``load_model``from``utils``import``preprocess`![](img/00002.jpg)`model``=``load_model``(``'model.h5'``)`![](img/00075.jpg)`app``=``Flask``(``__name__``)``@app.route``(``'/classify'``,``methods``=``[``'POST'``])``def``classify``():``complaint_data``=``request``.``form``[``"complaint_data"``]``preprocessed_complaint_data``=``preprocess``(``complaint_data``)``prediction``=``model``.``predict``([``preprocessed_complaint_data``])`![](img/00064.jpg)`return``json``.``dumps``({``"score"``:``prediction``})`![](img/00055.jpg)

![](img/00002.jpg)

> > 预处理以转换数据结构。

![](img/00075.jpg)

> > 加载您训练过的模型。

![](img/00064.jpg)

> > 执行预测。

![](img/00055.jpg)

> > 返回预测结果的 HTTP 响应。

这种设置适用于演示项目的快速简易实现。然而，我们不建议将 示例 8-1 用于将机器学习模型部署到生产端点。

接下来，让我们讨论为何不建议使用此类设置部署机器学习模型的原因。原因在于我们对提议的部署解决方案的基准。

使用 Python-Based API 部署模型的缺点

虽然 示例 8-1 的实现对演示目的可能足够，但这类部署通常面临挑战。挑战从 API 和数据科学代码之间的适当分离开始，一个一致的 API 结构和由此产生的不一致的模型版本管理，以及低效的模型推断。我们将在接下来的部分更详细地探讨这些挑战。

缺乏代码分离

在 示例 8-1 中，我们假设训练过的模型正在与同一 API 代码库一起部署。这意味着 API 代码和机器学习模型之间没有分离，当数据科学家想要更新模型时，这种更新就需要与 API 团队协调。这种协调还要求 API 和数据科学团队同步工作，以避免在模型部署上造成不必要的延迟。

纠缠的 API 和数据科学代码库还会在 API 拥有权方面造成歧义。

代码分离的缺乏还要求模型必须在与 API 代码相同的编程语言中加载。这种后端与数据科学代码的混合最终可能阻止 API 团队升级 API 后端。然而，它也提供了良好的责任分离：数据科学家可以专注于模型训练，DevOps 同事可以专注于训练模型的部署。

我们强调了如何有效地将您的模型与 API 代码分离，并简化您的部署工作流程，详见 “TensorFlow Serving”。

模型版本控制的缺乏

示例 8-1 没有为不同的模型版本做任何准备。如果您想要添加一个新版本，您将不得不创建一个新的端点（或在现有端点中添加一些分支逻辑）。这需要额外注意以保持所有端点在结构上相同，并且需要大量样板代码。

模型版本控制的缺乏还要求 API 和数据科学团队协调确定默认版本及如何逐步引入新模型。

低效的模型推理

对于写入 Flask 设置中预测端点的任何请求，如 示例 8-1 所示，都会执行一个完整的往返。这意味着每个请求都会单独进行预处理和推断。我们认为这种设置仅用于演示的关键原因是其效率极低。在训练模型时，您可能会使用批处理技术，允许您同时计算多个样本，然后将批处理的梯度变化应用于网络的权重。当您希望模型进行预测时，可以应用相同的技术。模型服务器可以在可接受的时间范围内或直到批处理满时收集所有请求，并询问模型其预测结果。当推理运行在 GPU 上时，这是一种特别有效的方法。

在 “批处理推断请求” 中，我们介绍了如何为您的模型服务器轻松设置这样的批处理行为。

TensorFlow Serving

正如你在本书的前几章中所见，TensorFlow 提供了一个出色的扩展和工具生态系统。早期的开源扩展之一是 TensorFlow Serving。它允许你部署任何 TensorFlow 图，并通过其标准化的端点从图中进行预测。正如我们接下来会讨论的，TensorFlow Serving 为你处理模型和版本管理，让你根据策略提供模型，允许你从各种来源加载你的模型。同时，它专注于高性能吞吐量以进行低延迟的预测。TensorFlow Serving 在 Google 内部使用，并已被许多公司和初创企业采用。1

TensorFlow 架构概述

TensorFlow Serving 为您提供了从给定来源（例如 AWS S3 存储桶）加载模型并在源更改时通知加载器的功能。如 Figure 8-2 所示，TensorFlow Serving 的背后所有操作都由模型管理器控制，该管理器负责何时更新模型以及哪个模型用于预测。推断决策规则由策略管理，该策略也由模型管理器管理。根据您的配置，例如，您可以一次加载一个模型，并且一旦源模块检测到更新版本，模型将自动更新。

![](img/00038.jpg)

图 8-2\. TensorFlow Serving 架构概述

导出 TensorFlow Serving 的模型

在我们深入讨论 TensorFlow Serving 配置之前，让我们讨论一下如何导出您的机器学习模型，以便它们可以被 TensorFlow Serving 使用。

根据您的 TensorFlow 模型类型，导出步骤略有不同。导出的模型与我们在 Example 8-2 中看到的文件结构相同。对于 Keras 模型，您可以使用：

> `saved_model_path``=``model``.``save``(``file``path``=``"./saved_models"``,``save_format``=``"tf"``)`
> 
> 为您的导出路径添加时间戳
> 
> 当您手动保存模型时，建议为 Keras 模型的导出路径添加导出时间戳。与`tf.Estimator`的`save`方法不同，`model.save()`不会自动创建带时间戳的路径。您可以使用以下 Python 代码轻松创建文件路径：
> 
> > `import``time``ts``=``int``(``time``.``time``())``file``path``=``"./saved_models/{}"``.``format``(``ts``)``saved_model_path``=``model``.``save``(``file``path``=``file``path``,``save_format``=``"tf"``)`

对于 TensorFlow Estimator 模型，您需要首先声明一个`接收函数`：

> `import``tensorflow``as``tf``def``serving_input_receiver_fn``():``# an example input feature``input_feature``=``tf``.``compat``.``v1``.``placeholder``(``dtype``=``tf``.``string``,``shape``=``[``None``,``1``],``name``=``"input"``)``fn``=``tf``.``estimator``.``export``.``build_raw_serving_input_receiver_fn``(``features``=``{``"input_feature"``:``input_feature``})``return``fn`

使用`Estimator`的`export_saved_model`方法导出 Estimator 模型：

> `estimator``=``tf``.``estimator``.``Estimator``(``model_fn``,``"model"``,``params``=``{})``estimator``.``export_saved_model``(``export_dir_base``=``"saved_models/"``,``serving_input_receiver_fn``=``serving_input_receiver_fn``)`

两种导出方法产生的输出与以下示例类似：

> `... INFO:tensorflow:在导出中包含的签名: None INFO:tensorflow:在导出中包含的签名: None INFO:tensorflow:在导出中包含的签名: ` `[``'serving_default'``]` `INFO:tensorflow:在导出中包含的签名: None INFO:tensorflow:在导出中包含的签名: None INFO:tensorflow:没有要保存的资产。 INFO:tensorflow:没有要写入的资产。 INFO:tensorflow:SavedModel 已写入: saved_models/1555875926/saved_model.pb 模型已导出至:  b``'saved_models/1555875926'`

在我们的模型导出示例中，我们指定了文件夹 saved_models/作为模型的目标位置。对于每个导出的模型，TensorFlow 会创建一个以导出时间戳命名的目录：

示例 8-2\. 导出模型的文件夹和文件结构

`$` `tree saved_models/ saved_models/ └── 1555875926     ├── assets     │   └── saved_model.json     ├── saved_model.pb     └── variables         ├── checkpoint         ├── variables.data-00000-of-00001         └── variables.index` `3` `directories,` `5` `files`

文件夹包含以下文件和子目录：

saved_model.pb

> > 二进制协议缓冲文件以`MetaGraphDef`对象形式包含导出模型图结构。

变量

> > 文件夹包含具有导出变量值的二进制文件以及对应于导出模型图的检查点。

assets

> > 当需要加载导出模型的其他文件时，会创建此文件夹。附加文件可以包括词汇表，这些词汇表在第五章中有介绍。

模型签名

模型签名标识模型图的输入和输出，以及图签名的方法。定义输入和输出签名允许我们将服务输入映射到给定图节点以进行推理。如果我们想要更新模型而不改变模型服务器的请求，这些映射就非常有用。

此外，模型的方法定义了输入和输出的预期模式。目前支持三种签名类型：预测、分类或回归。我们将在下一节详细讨论这些细节。

签名方法

最灵活的签名方法是预测。如果我们没有指定不同的签名方法，TensorFlow 将使用预测作为默认方法。示例 8-3 展示了方法预测的签名示例。在该示例中，我们将键`inputs`映射到名为 sentence 的图节点。模型的预测输出是图节点 y 的输出，我们将其映射到输出键`scores`。

预测方法允许您定义额外的输出节点。当您希望捕获注意力层输出以进行可视化或调试网络节点时，添加更多推理输出非常有用。

示例 8-3\. 模型预测签名示例

`signature_def``:``{``key``:``"prediction_signature"``value``:``{``inputs``:``{``key``:``"inputs"``value``:``{``name``:``"sentence:0"``dtype``:``DT_STRING``tensor_shape``:``...``},``...``}``outputs``:``{``key``:``"scores"``value``:``{``name``:``"y:0"``dtype``:``...``tensor_shape``:``...``}``}``method_name``:``"tensorflow/serving/predict"``}``}`

另一种签名方法是 `classify`。该方法期望一个名为 `inputs` 的输入，并提供两个输出张量，`classes` 和 `scores`。至少需要定义一个输出张量。在我们示例中显示的示例 8-4 中，分类模型接受输入 `sentence` 并输出预测的 `classes` 以及相应的 `scores`。

示例 8-4\. 示例模型分类签名

`signature_def``:``{``key``:``"classification_signature"``value``:``{``inputs``:``{``key``:``"inputs"``value``:``{``name``:``"sentence:0"``dtype``:``DT_STRING``tensor_shape``:``...``}``}``outputs``:``{``key``:``"classes"``value``:``{``name``:``"y_classes:0"``dtype``:``DT_UINT16``tensor_shape``:``...``}``}``outputs``:``{``key``:``"scores"``value``:``{``name``:``"y:0"``dtype``:``DT_FLOAT``tensor_shape``:``...``}``}``method_name``:``"tensorflow/serving/classify"``}``}`

第三种可用的签名方法是 `regress`。此方法仅接受名为 `inputs` 的输入，并仅提供名为 `outputs` 的输出。这种签名方法设计用于回归模型。示例 8-5 展示了回归签名的示例。

示例 8-5\. 示例模型回归签名

`signature_def``:``{``key``:``"regression_signature"``value``:``{``inputs``:``{``key``:``"inputs"``value``:``{``name``:``"input_tensor_0"``dtype``:``...``tensor_shape``:``...``}``}``outputs``:``{``key``:``"outputs"``value``:``{``name``:``"y_outputs_0"``dtype``:``DT_FLOAT``tensor_shape``:``...``}``}``method_name``:``"tensorflow/serving/regress"``}``}`

在“URL 结构”中，当我们为模型端点定义 URL 结构时，我们将再次看到签名方法。

检查导出的模型

在讨论了如何导出模型及其对应模型签名之后，让我们讨论在部署 TensorFlow Serving 之前如何检查导出的模型。

您可以通过以下 `pip` 命令安装 TensorFlow Serving Python API：

> `$` `pip install tensorflow-serving-api`

安装后，您可以使用一个称为 SavedModel 命令行接口（CLI）的实用命令行工具。此工具可以让您：

检查导出模型的签名

> > 主要在您没有自行导出模型且希望了解模型图的输入和输出时非常有用。

测试导出的模型

> > 使用 TensorFlow Serving 的 CLI 工具，您可以推断模型而无需部署它。这在您希望测试模型输入数据时非常有用。

我们将在以下两个部分中涵盖这两种用例。

检查模型

`saved_model_cli` 可帮助您了解模型的依赖关系，而无需检查原始图形代码。

如果您不知道可用的标签集合，2 您可以使用以下命令检查模型：

> `$` `saved_model_cli show --dir saved_models/ The given SavedModel contains the following tag-sets: serve`

如果您的模型包含不同环境的不同图形（例如，用于 CPU 或 GPU 推断的图形），则会看到多个标签。如果模型包含多个标签，则需要指定一个标签以检查模型的详细信息。

一旦您知道要检查的 `tag_set`，请将其添加为参数，`saved_model_cli` 将提供可用的模型签名。我们的演示模型只有一个名为 `serving_default` 的签名：

> `$` `saved_model_cli show --dir saved_models/ --tag_set serve The given SavedModel` `'MetaGraphDef'` `contains` `'SignatureDefs'` `with the following keys: SignatureDef key:` `"serving_default"`

借助 `tag_set` 和 `signature_def` 信息，您现在可以检查模型的输入和输出。要获取详细信息，请将 `signature_def` 添加到 CLI 参数中。

以下示例签名取自我们演示流水线生成的模型。在 示例 6-4 中，我们定义了签名函数，该函数将序列化的 `tf.Example` 记录作为输入，并通过输出张量 `outputs` 提供预测结果，如以下模型签名所示：

> `$` `saved_model_cli show --dir saved_models/` `\` `--tag_set serve --signature_def serving_default The given SavedModel SignatureDef contains the following input``(``s``)``:   inputs``[``'examples'``]` `tensor_info:       dtype: DT_STRING       shape:` `(``-1``)` `name: serving_default_examples:0 The given SavedModel SignatureDef contains the following output``(``s``)``:   outputs``[``'outputs'``]` `tensor_info:       dtype: DT_FLOAT       shape:` `(``-1, 1``)` `name: StatefulPartitionedCall_1:0 Method name is: tensorflow/serving/predict`

如果要查看所有签名而不考虑 `tag_set` 和 `signature_def`，可以使用 `--all` 参数：

> `$` `saved_model_cli show --dir saved_models/ --all ...`

在我们调查模型签名后，我们现在可以在部署机器学习模型之前测试模型推理。

测试模型

`saved_model_cli` 还允许您使用示例输入数据测试导出模型。

您有三种不同的方式提交模型测试推理的示例输入数据：

`--inputs`

> > 该参数指向一个包含以 NumPy `ndarray` 格式编码的输入数据的 NumPy 文件。

`--input_exprs`

> > 该参数允许您定义 Python 表达式以指定输入数据。您可以在表达式中使用 NumPy 功能。

`--input_examples`

> > 该参数期望以 `tf.Example` 数据结构格式化的输入数据（请参见 第四章）。

要测试模型，您可以准确指定一个输入参数。此外，`saved_model_cli` 提供了三个可选参数：

`--outdir`

> > `saved_model_cli` 将任何图形输出写入 `stdout`。如果您更喜欢将输出写入文件，则可以使用 `--outdir` 指定目标目录。

`--overwrite`

> > 如果选择将输出写入文件，可以使用 `--overwrite` 指定文件可以被覆盖。

`--tf_debug`

> > 如果您想进一步检查模型，可以使用 TensorFlow Debugger (TFDBG) 逐步查看模型图。
> > 
> `$` `saved_model_cli run --dir saved_models/` `\` `--tag_set serve` `\` `--signature_def x1_x2_to_y` `\` `--input_examples` `'examples=[{"company": "汇丰银行", ...}]'`

在介绍如何导出和检查模型后，让我们深入了解 TensorFlow Serving 的安装、设置和操作。

设置 TensorFlow Serving

有两种简单的方法可以在您的服务实例上安装 TensorFlow Serving。您可以在 Docker 上运行 TensorFlow Serving，或者如果您在服务实例上运行 Ubuntu 操作系统，则可以安装 Ubuntu 包。

Docker 安装

安装 TensorFlow Serving 的最简单方法是下载预构建的 Docker 镜像。正如您在第二章中看到的那样，您可以通过运行以下命令获取该镜像：

> `$` `docker pull tensorflow/serving`

如果您在具有 GPU 的实例上运行 Docker 容器，则需要下载具有 GPU 支持的最新版本构建。

> `$` `docker pull tensorflow/serving:latest-gpu`

具有 GPU 支持的 Docker 镜像需要 Nvidia 的 Docker 支持。安装步骤可以在[公司的网站](https://oreil.ly/7N5uv)找到。

本地 Ubuntu 安装

如果您想要在没有运行 Docker 的开销的情况下运行 TensorFlow Serving，则可以安装适用于 Ubuntu 发行版的 Linux 二进制包。

安装步骤与其他非标准 Ubuntu 软件包类似。首先，您需要在 Linux 终端中执行以下操作，将一个新的软件包源添加到发行版的源列表中或者将一个新的列表文件添加到 `sources.list.d` 目录中：

> `$` `echo``"deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt \``  stable tensorflow-model-server tensorflow-model-server-universal"``\``|` `sudo tee /etc/apt/sources.list.d/tensorflow-serving.list`

在更新您的软件包注册表之前，您应将软件包的公钥添加到您发行版的密钥链中：

> `$` `curl https://storage.googleapis.com/tensorflow-serving-apt/``\` `tensorflow-serving.release.pub.gpg` `|` `sudo apt-key add -`

在更新您的软件包注册表后，您可以在您的 Ubuntu 操作系统上安装 TensorFlow Serving：

> `$` `apt-get update` `$` `apt-get install tensorflow-model-server`
> 
> 用于 TensorFlow Serving 的两个 Ubuntu 软件包
> 
> 谷歌为 TensorFlow Serving 提供了两个 Ubuntu 软件包！前文提到的 `tensorflow-model-server` 软件包是首选软件包，并带有特定的 CPU 优化预编译（例如，AVX 指令）。
> 
> 在撰写本章时，还提供了名为 `tensorflow-model-server-universal` 的第二个软件包。它不包含预编译的优化，因此可以在旧硬件上运行（例如，没有 AVX 指令集的 CPU）。

从源代码构建 TensorFlow Serving。

建议使用预构建的 Docker 镜像或 Ubuntu 软件包运行 TensorFlow Serving。在某些情况下，您需要编译 TensorFlow Serving，例如当您希望为底层硬件优化模型服务时。目前，您只能为 Linux 操作系统构建 TensorFlow Serving，需要使用构建工具 `bazel`。您可以在 [TensorFlow Serving 文档](https://oreil.ly/tUJTw) 中找到详细的说明。

> 优化您的 TensorFlow Serving 实例。
> 
> 如果您从头开始构建 TensorFlow Serving，我们强烈建议为您的模型的特定 TensorFlow 版本和您服务实例的可用硬件编译 Serving 版本。

配置 TensorFlow 服务器。

TensorFlow Serving 出厂设置可以在两种不同模式下运行。首先，您可以指定一个模型，并让 TensorFlow Serving 始终提供最新的模型。或者，您可以指定一个包含所有您想要加载的模型和版本的配置文件，并让 TensorFlow Serving 加载所有命名模型。

单一模型配置。

如果您想通过加载单个模型并在可用时切换到更新的模型版本来运行 TensorFlow Serving，则首选单一模型配置。

如果您在 Docker 环境中运行 TensorFlow Serving，则可以使用以下命令运行 `tensorflow\serving` 镜像：

> `$` `docker run -p 8500:8500` `\` ![](img/00002.jpg) `-p 8501:8501` `\` `--mount` `type``=``bind``,source``=``/tmp/models,target``=``/models/my_model` `\` ![](img/00075.jpg) `-e` `MODEL_NAME``=``my_model` `\` ![](img/00064.jpg) `-e` `MODEL_BASE_PATH``=``/models/my_model` `\` `-t tensorflow/serving` ![](img/00055.jpg)

![](img/00002.jpg)

> > 指定默认端口。

![](img/00075.jpg)

> > 挂载模型目录。

![](img/00064.jpg)

> > 指定您的模型。

![](img/00055.jpg)

> > 指定 Docker 镜像。

默认情况下，TensorFlow Serving 配置为创建一个表述状态传输（REST）和 Google 远程过程调用（gRPC）端点。通过指定端口 8500 和 8501，我们暴露了 REST 和 gRPC 的能力。4 Docker `run` 命令在主机（源）和容器（目标）文件系统之间创建了一个挂载点。在第二章中，我们讨论了如何向 Docker 容器传递环境变量。要在单个模型配置中运行服务器，您需要指定模型名称 `MODEL_NAME`。

如果您想要运行预先构建的用于 GPU 的 Docker 镜像，您需要将 docker 镜像的名称更换为最新的 GPU 构建版本：

> `$` `docker run ...              -t tensorflow/serving:latest-gpu`

如果您决定在没有 Docker 容器的情况下运行 TensorFlow Serving，您可以使用以下命令运行它：

> `$` `tensorflow_model_server --port``=``8500``\` `--rest_api_port``=``8501``\` `--model_name``=``my_model` `\` `--model_base_path``=``/models/my_model`

在这两种情况下，您应该在终端上看到类似以下的输出：

> `2019-04-26 03:51:20.304826: I tensorflow_serving/model_servers/ server.cc:82``]` `构建单个 TensorFlow 模型文件配置：   model_name: my_model model_base_path: /models/my_model 2019-04-26 03:51:20: I tensorflow_serving/model_servers/server_core.cc:461``]` `添加/更新模型。 2019-04-26 03:51:20: I tensorflow_serving/model_servers/ server_core.cc:558``]``(``重新``)``添加模型：my_model ... 2019-04-26 03:51:34.507436: I tensorflow_serving/core/loader_harness.cc:86``]` `成功加载可服务版本` `{``name: my_model version: 1556250435``}` `2019-04-26 03:51:34.516601: I tensorflow_serving/model_servers/server.cc:313``]` `在 0.0.0.0:8500 运行 gRPC ModelServer ...` `[``warn``]` `getaddrinfo：不支持节点名称的地址族` `[``evhttp_server.cc:237``]` `RAW：进入事件循环 ... 2019-04-26 03:51:34.520287: I tensorflow_serving/model_servers/server.cc:333``]` `在 localhost:8501 导出 HTTP/REST API ...`

从服务器输出中，您可以看到服务器成功加载了我们的模型 `my_model`，并创建了两个端点：一个 REST 端点和一个 gRPC 端点。

TensorFlow Serving 使得部署机器学习模型变得极其简单。使用 TensorFlow Serving 提供模型的一个巨大优势是其热替换能力。如果上传了新模型，服务器的模型管理器将检测到新版本，卸载现有模型，并加载更新的模型以进行推理。

假设您更新了模型并将新模型版本导出到主机机器上的挂载文件夹（如果您正在使用 Docker 设置），不需要进行配置更改。模型管理器将检测到更新的模型并重新加载端点。它将通知您有关旧模型卸载和新模型加载的消息。在您的终端中，您应该会找到类似以下的消息：

> `2019-04-30 00:21:56.486988: I tensorflow_serving/core/basic_manager.cc:739` `]` `成功保留资源以加载可服务的` `{` `name: my_model version: 1556583584` `}` `2019-04-30 00:21:56.487043: I tensorflow_serving/core/loader_harness.cc:66` `]` `批准加载` `服务版本` `{` `name: my_model version: 1556583584` `}` `2019-04-30 00:21:56.487071: I tensorflow_serving/core/loader_harness.cc:74` `]` `加载服务版本` `{` `name: my_model version: 1556583584` `}` `... 2019-04-30 00:22:08.839375: I tensorflow_serving/core/loader_harness.cc:119` `]` `卸载服务版本` `{` `name: my_model version: 1556583236` `}` `2019-04-30 00:22:10.292695: I ./tensorflow_serving/core/simple_loader.h:294` `]` `在释放带有 1262338988 的服务卸载后调用 MallocExtension_ReleaseToSystem` `()` `2019-04-30 00:22:10.292771: I tensorflow_serving/core/loader_harness.cc:127` `]` `完成卸载服务版本` `{` `name: my_model version: 1556583236` `}`

默认情况下，TensorFlow Serving 将加载具有最高版本号的模型。如果您使用本章前面展示的导出方法，所有模型将以时代时间戳作为文件夹名称导出。因此，新模型的版本号将高于旧模型。

TensorFlow Serving 的相同默认模型加载策略还允许模型回滚。如果您想回滚模型版本，可以从基本路径删除模型版本。模型服务器将在下一次轮询文件系统时检测到版本的删除，卸载已删除的模型，并加载最近的现有模型版本。

多模型配置

您还可以配置 TensorFlow Serving 同时加载多个模型。为此，您需要创建一个配置文件来指定模型：

> `model_config_list` `{` `config` `{` `name:` `'my_model'` `base_path:` `'/models/my_model/'` `model_platform:` `'tensorflow'` `}` `config` `{` `name:` `'another_model'` `base_path:` `'/models/another_model/'` `model_platform:` `'tensorflow'` `}` `}`

配置文件包含一个或多个 `config` 字典，所有这些都在 `model_config_list` 键下列出。

在您的 Docker 配置中，您可以挂载配置文件，并使用配置文件而不是单个模型加载模型服务器：

> `$` `docker run -p 8500:8500` `\` `-p 8501:8501` `\` `--mount` `type``=``bind``,source``=``/tmp/models,target``=``/models/my_model` `\` `--mount` `type``=``bind``,source``=``/tmp/model_config,``\` `target``=``/models/model_config` `\` ![](img/00002.jpg) `-e` `MODEL_NAME``=``my_model` `\` `-t tensorflow/serving` `\` `             --model_config_file``=``/models/model_config` ![](img/00075.jpg)

![](img/00002.jpg)

> > 挂载配置文件。

![](img/00075.jpg)

> > 指定模型配置文件。

如果您在 Docker 容器之外使用 TensorFlow Serving，可以使用参数 `model_config_file` 将模型服务器指向配置文件，该文件加载并从文件中进行配置：

> `$` `tensorflow_model_server --port``=``8500``\` `--rest_api_port``=``8501``\` `--model_config_file``=``/models/model_config`
> 
> 配置特定模型版本
> 
> 有些情况下，您不仅想加载最新的模型版本，而是所有或特定的模型版本。例如，您可能想进行模型 A/B 测试，如我们将在“使用 TensorFlow Serving 进行模型 A/B 测试”中讨论的那样，或者提供一个稳定版本和一个开发版本。TensorFlow Serving 默认始终加载最新的模型版本。如果您想加载一组可用的模型版本，您可以通过以下方式扩展模型配置文件：
> 
> `...``config``{``name:``'another_model'``base_path:``'/models/another_model/'``model_version_policy``{``specific``{``versions:``1556250435``versions:``1556251435``}``}``}``...
> 
> 如果您想指定特定的模型版本，您也可以定义它们：
> 
> `...``config``{``name:``'another_model'``base_path:``'/models/another_model/'``model_version_policy``{``specific``{``versions:``1556250435``versions:``1556251435``}``}``}``...
> 
> 您甚至可以提供模型版本的标签。这些标签在以后从模型进行预测时非常方便。在撰写本文时，版本标签仅通过 TensorFlow Serving 的 gRPC 端点提供：
> 
> `...``model_version_policy``{``specific``{``versions:``1556250435``versions:``1556251435``}``}``version_labels``{``key:``'stable'``value:``1556250435``}``version_labels``{``key:``'testing'``value:``1556251435``}``...
> 
> 现在配置了模型版本，我们可以使用这些端点来运行我们的模型 A/B 测试版本。如果您想了解如何推断这些模型版本，我们建议参考“使用 TensorFlow Serving 进行模型 A/B 测试”，这里有一个简单实现的例子。
> 
> 自 TensorFlow Serving 2.3 开始，版本标签功能将适用于 REST 端点，除了现有的 TensorFlow Serving 的 gRPC 功能之外。

REST 与 gRPC

在“单一模型配置”中，我们讨论了 TensorFlow Serving 如何允许两种不同的 API 类型：REST 和 gRPC。这两种协议都有各自的优缺点，在我们深入介绍如何与这些端点通信之前，我们想先介绍一下它们。

REST

REST 是当今 Web 服务使用的通信“协议”。它不是正式协议，而是定义客户端如何与 Web 服务通信的通信样式。REST 客户端使用标准的 HTTP 方法（如`GET`，`POST`，`DELETE`等）与服务器通信。请求的有效负载通常编码为 XML 或 JSON 数据格式。

gRPC

gRPC 是由 Google 开发的远程过程调用协议。虽然 gRPC 支持不同的数据格式，但与 gRPC 一起使用的标准数据格式是协议缓冲区，这在本书中被广泛使用。gRPC 提供低延迟通信和较小的有效负载（如果使用协议缓冲区）。gRPC 设计时考虑了 API。缺点是有效负载以二进制格式存在，这可能使快速检查变得困难。

> 使用哪种协议？
> 
> 一方面，通过 REST 与模型服务器进行通信看起来非常方便。端点易于推断，有效负载可以轻松检查，并且可以使用`curl`请求或浏览器工具测试端点。REST 库广泛适用于各种客户端，并且通常已在客户端系统上可用（例如移动应用程序）。
> 
> 另一方面，gRPC API 在初始阶段具有更高的入门门槛。 gRPC 库通常需要在客户端上安装。但是，根据模型推断所需的数据结构，它们可以带来显著的性能改进。如果您的模型接收到许多请求，则由于协议缓冲区序列化而减少的有效负载大小可能会有所帮助。
> 
> 在内部，TensorFlow Serving 将通过 REST 提交的 JSON 数据结构转换为`tf.Example`数据结构，这可能导致性能较慢。因此，如果转换需要许多类型转换（即，如果您提交包含浮点值的大数组），则通过 gRPC 请求可能会获得更好的性能。

从模型服务器获取预测

到目前为止，我们完全专注于模型服务器的设置。在本节中，我们想展示客户端（例如 Web 应用程序）如何与模型服务器交互。所有关于 REST 或 gRPC 请求的代码示例都在客户端上执行。

通过 REST 获取模型预测

要通过 REST 调用模型服务器，您需要一个 Python 库来为您简化通信。目前的标准库是`requests`。安装该库：

> `$` `pip install requests`

下面的示例展示了一个`POST`请求的示例。

> `import``requests``url``=``"http://some-domain.abc"``payload``=``{``"key_1"``:``"value_1"``}``r``=``requests``.``post``(``url``,``json``=``payload``)`![](img/00002.jpg)`print``(``r``.``json``())`![](img/00075.jpg)`# {'data': ...}`

![](img/00002.jpg)

> > 提交请求。

![](img/00075.jpg)

> > 查看 HTTP 响应。

URL 结构

您的 HTTP 请求 URL 包含有关您想要推断的模型和版本的信息：

> `http://``{``HOST``}``:``{``PORT``}``/v1/models/``{``MODEL_NAME``}``:``{``VERB``}`

主机

> > 主机是您模型服务器的 IP 地址或域名。如果您在同一台机器上运行模型服务器和客户端代码，您可以将主机设置为 `localhost`。

端口

> > 在您的请求 URL 中需要指定端口。REST API 的标准端口为 8501\. 如果这与您服务生态系统中的其他服务冲突，您可以在服务器启动期间通过服务器参数更改端口。

模型名称

> > 当您设置模型配置或启动模型服务器时，模型名称需要与您模型的名称匹配。

动词

> > URL 中的动词指定了模型的类型。您有三个选项：`predict`、`classify` 或 `regress`。动词对应于终点的签名方法。

模型版本

> > 如果您想要从特定模型版本进行预测，您需要使用模型版本标识符扩展 URL：
> > 
> `http://``{``HOST``}``:``{``PORT``}``/v1/models/``{``MODEL_NAME``}[``/versions/``${``MODEL_VERSION``}``]``:``{``VERB``}`

负载

在 URL 就绪后，让我们讨论请求负载。TensorFlow Serving 期望将输入数据作为 JSON 数据结构提交，如以下示例所示：

> `{``"signature_name"``:``<string>``,``"instances"``:``<value>``}`

不需要 `signature_name`。如果未指定，模型服务器将推断使用默认的 `serving` 标签签署的模型图。

输入数据预期为对象列表或输入值列表。要提交多个数据样本，您可以将它们作为列表提交在 `instances` 键下。

如果您想要为推理提交一个数据示例，您可以使用 `inputs` 并将所有输入值列为列表。其中的一个键，`instances` 和 `inputs`，必须存在，但不能同时出现：

> `{``"signature_name"``:``<string>``,``"inputs"``:``<value>``}`

示例 8-6 展示了如何从我们的 TensorFlow 服务端点请求模型预测的示例。在我们的示例中，我们只提交了一个数据示例用于推理，但我们可以轻松地提交代表多个请求的数据输入列表。

示例 8-6\. 使用 Python 客户端进行模型预测请求的示例

`import``requests``def``get_rest_request``(``text``,``model_name``=``"my_model"``):``url``=``"http://localhost:8501/v1/models/{}:predict"``.``format``(``model_name``)`![](img/00002.jpg)`payload``=``{``"instances"``:``[``text``]}`![](img/00075.jpg)`response``=``requests``.``post``(``url``=``url``,``json``=``payload``)``return``response``rs_rest``=``get_rest_request``(``text``=``"classify my text"``)``rs_rest``.``json``()`

![](img/00002.jpg)

> > 将服务器未在同一台机器上运行时，将`localhost`替换为 IP 地址。

![](img/00075.jpg)

> > 如果要推断更多示例，请在`instance`列表中添加更多示例。

通过 gRPC 使用 TensorFlow Serving

如果要使用 gRPC 模型，步骤与 REST API 请求略有不同。

首先，建立一个 gRPC `channel`。该通道提供与给定主机地址上的 gRPC 服务器的连接，并通过给定端口。如果需要安全连接，则需要在此时建立安全通道。通道建立后，将创建一个`stub`。`stub`是一个本地对象，复制自服务器中可用的方法：

> `import``grpc``from``tensorflow_serving.apis``import``predict_pb2``from``tensorflow_serving.apis``import``prediction_service_pb2_grpc``import``tensorflow``as``tf``def``create_grpc_stub``(``host``,``port``=``8500``):``hostport``=``"{}:{}"``.``format``(``host``,``port``)``channel``=``grpc``.``insecure_channel``(``hostport``)``stub``=``prediction_service_pb2_grpc``.``PredictionServiceStub``(``channel``)``return``stub`

创建 gRPC stub 后，我们可以设置模型和签名，以访问正确模型的预测并提交我们的数据进行推理：

> `def``grpc_request``(``stub``,``data_sample``,``model_name``=``'my_model'``,` `\` `signature_name``=``'classification'``):``request``=``predict_pb2``.``PredictRequest``()``request``.``model_spec``.``name``=``model_name``request``.``model_spec``.``signature_name``=``signature_name``request``.``inputs``[``'inputs'``]``.``CopyFrom``(``tf``.``make_tensor_proto``(``data_sample``,``shape``=``[``1``,``1``]))`![](img/00002.jpg)`result_future``=``stub``.``Predict``.``future``(``request``,``10``)`![](img/00075.jpg)`return``result_future`

![](img/00002.jpg)

> > `inputs`是我们神经网络输入的名称。

![](img/00075.jpg)

> > 10 是函数超时前的最大时间（秒数）。

现在，有了这两个可用的函数，我们可以使用这两个函数调用推断我们的示例数据集：

> `stub``=``create_grpc_stub``(``host``,``port``=``8500``)``rs_grpc``=``grpc_request``(``stub``,``data``)`
> 
> 安全连接
> 
> `grpc`库还提供了与 gRPC 端点安全连接的功能。以下示例显示如何从客户端端创建安全通道与 gRPC：
> 
> `import grpc` `cert``=` `open``(``client_cert_file,` `'rb'``)``.read``()``key``=` `open``(``client_key_file,` `'rb'``)``.read``()``ca_cert``=` `open``(``ca_cert_file,` `'rb'``)``.read``()``if` `ca_cert_file` `else``''``credentials``=` `grpc.ssl_channel_credentials``(` `ca_cert, key, cert` `)``channel``=` `implementations.secure_channel``(``hostport, credentials``)`
> 
> 在服务器端，如果配置了 SSL，TensorFlow Serving 可以终止安全连接。要终止安全连接，请按以下示例创建 SSL 配置文件：6
> 
> `server_key:` `"-----BEGIN PRIVATE KEY-----\n``              <your_ssl_key>\n``              -----END PRIVATE KEY-----"` `server_cert:` `"-----BEGIN CERTIFICATE-----\n``              <your_ssl_cert>\n``              -----END CERTIFICATE-----"` `custom_ca:` `""` `client_verify:` `false`
> 
> 创建了配置文件后，您可以在启动 TensorFlow Serving 时将文件路径传递给 `--ssl_config_file` 参数：
> 
> `$` `tensorflow_model_server --port``=``8500``\` `--rest_api_port``=``8501``\` `--model_name``=``my_model` `\` `--model_base_path``=``/models/my_model` `\` `--ssl_config_file``=``"<path_to_config_file>"`

从分类和回归模型获取预测结果

如果您对从分类和回归模型进行预测感兴趣，可以使用 gRPC API。

如果您想从分类模型获取预测结果，则需要替换以下行：

> `from``tensorflow_serving.apis``import``predict_pb2``...``request``=``predict_pb2``.``PredictRequest``()`

with:

> `from``tensorflow_serving.apis``import``classification_pb2``...``request``=``classification_pb2``.``ClassificationRequest``()`

如果您想从回归模型获取预测结果，可以使用以下导入语句：

> `from``tensorflow_serving.apis``import``regression_pb2``...``regression_pb2``.``RegressionRequest``()`

Payloads

gRPC API 使用协议缓冲作为 API 请求的数据结构。通过使用二进制协议缓冲有效减少了 API 请求所需的带宽，与 JSON 负载相比。此外，根据模型输入数据结构的不同，您可能会体验到更快的预测速度，如同 REST 终端节点一样。性能差异在于，提交的 JSON 数据将被转换为 `tf.Example` 数据结构。这种转换可能会减慢模型服务器的推断速度，您可能会遇到比 gRPC API 情况下更慢的推断性能。

您提交给 gRPC 终端节点的数据需要转换为协议缓冲数据结构。TensorFlow 为您提供了一个便捷的实用函数来执行转换，称为 `tf.make_tensor_proto`。它允许各种数据格式，包括标量、列表、NumPy 标量和 NumPy 数组。该函数将会把给定的 Python 或 NumPy 数据结构转换为推断所需的协议缓冲格式。

使用 TensorFlow Serving 进行模型 A/B 测试

A/B 测试是在实际情况下测试不同模型的优秀方法论。在这种情况下，一定比例的客户将接收模型版本 A 的预测，所有其他请求将由模型版本 B 提供服务。

我们之前讨论过，您可以配置 TensorFlow Serving 加载多个模型版本，然后在您的 REST 请求 URL 或 gRPC 规范中指定模型版本。

TensorFlow Serving 不支持服务器端的 A/B 测试，这意味着模型服务器将所有客户端请求重定向到两个模型版本中的单一端点。但是通过稍微调整我们的请求 URL，我们可以为客户端提供对随机 A/B 测试的适当支持：7

> `from``random``import``random`![](img/00002.jpg)`def``get_rest_url``(``model_name``,``host``=``'localhost'``,``port``=``8501``,``verb``=``'predict'``,``version``=``None``):``url``=``"http://{}:{}/v1/models/{}/"``.``format``(``host``,``port``,``model_name``)``if``version``:``url``+=``"versions/{}"``.``format``(``version``)``url``+=``":{}"``.``format``(``verb``)``return``url``...``# 从此客户端的所有请求中提交 10% 给版本 1。``# 90% 的请求应该转到默认模型。``threshold``=``0.1``version``=``1``if``random``()``<``threshold``else``None`![](img/00075.jpg)`url``=``get_rest_url``(``model_name``=``'complaints_classification'``,``version``=``version``)`

![](img/00002.jpg)

> > `random` 库将帮助我们选择一个模型。

![](img/00075.jpg)

> > 如果 `version = None`，TensorFlow Serving 将使用默认版本进行推断。

正如您所看到的，随机更改我们模型推理的请求 URL（在我们的 REST API 示例中），可以为您提供一些基本的 A/B 测试功能。如果您希望通过在服务器端执行模型推理的随机路由来扩展这些能力，我们强烈建议使用像 [Istio](https://istio.io) 这样的路由工具。Istio 最初设计用于网络流量，可用于将流量路由到特定模型，逐步引入模型，执行 A/B 测试或创建数据路由到特定模型的策略。

当您使用模型进行 A/B 测试时，通常有必要从模型服务器请求模型的信息。在接下来的部分中，我们将解释如何从 TensorFlow Serving 请求元数据信息。

从模型服务器请求模型元数据

在本书的开头，我们阐述了模型生命周期，并解释了我们希望自动化机器学习生命周期的方式。连续生命周期的关键组成部分是生成关于模型版本的准确性或一般性能反馈。我们将深入研究如何生成这些反馈循环，详见第十三章，但现在，想象一下，您的模型对某些数据进行分类（例如文本的情感），然后请用户对预测结果进行评分。模型是否正确预测了某些事物的信息对于改进未来的模型版本至关重要，但仅当我们知道哪个模型版本执行了预测时才有用。

模型服务器提供的元数据将包含用于注释反馈循环的信息。

模型元数据的 REST 请求

使用 TensorFlow Serving 请求模型元数据非常直接。TensorFlow Serving 为模型元数据提供了一个端点：

> `http://``{``HOST``}``:``{``PORT``}``/v1/models/``{``MODEL_NAME``}[``/versions/``{``MODEL_VERSION``}]``/metadata`

类似我们之前讨论的 REST API 推断请求，您可以选择在请求 URL 中指定模型版本，或者如果不指定，模型服务器将提供关于默认模型的信息。

如示例 8-7 所示，我们可以通过单个 `GET` 请求请求模型元数据。

示例 8-7\. 使用 Python 客户端请求模型元数据的示例

`import``requests``def``metadata_rest_request``(``model_name``,``host``=``"localhost"``,``port``=``8501``,``version``=``None``):``url``=``"http://{}:{}/v1/models/{}/"``.``format``(``host``,``port``,``model_name``)``if``version``:``url``+=``"versions/{}"``.``format``(``version``)``url``+=``"/metadata"`![](img/00002.jpg)`response``=``requests``.``get``(``url``=``url``)`![](img/00075.jpg)`return``response`

![](img/00002.jpg)

> > 在请求 URL 中追加 `/metadata` 获取模型信息。

![](img/00075.jpg)

> > 执行 `GET` 请求。

模型服务器将以 `model_spec` 字典和 `metadata` 字典的形式返回模型规格和模型定义：

> `{``"model_spec"``:``{``"name"``:``"complaints_classification"``,``"signature_name"``:``""``,``"version"``:``"1556583584"``},``"metadata"``:``{``"signature_def"``:``{``"signature_def"``:``{``"classification"``:``{``"inputs"``:``{``"inputs"``:``{``"dtype"``:``"DT_STRING"``,``"tensor_shape"``:``{``...`

gRPC 请求模型元数据

使用 gRPC 请求模型元数据与 REST API 情况几乎一样简单。在 gRPC 情况下，您需要提交一个 `GetModelMetadataRequest`，将模型名称添加到规范中，并通过 `stub` 的 `GetModelMetadata` 方法提交请求。

> `from``tensorflow_serving.apis``import``get_model_metadata_pb2``def``get_model_version``(``model_name``,``stub``):``request``=``get_model_metadata_pb2``.``GetModelMetadataRequest``()``request``.``model_spec``.``name``=``model_name``request``.``metadata_field``.``append``(``"signature_def"``)``response``=``stub``.``GetModelMetadata``(``request``,``5``)``return``response``.``model_spec``model_name``=``'complaints_classification'``stub``=``create_grpc_stub``(``'localhost'``)``get_model_version``(``model_name``,``stub``)``name``:``"complaints_classification"``version``{``value``:``1556583584``}`

gRPC 响应包含一个`ModelSpec`对象，其中包含加载模型的版本号。

-   更有趣的是获取已加载模型的模型签名信息的用例。几乎使用相同的请求函数，我们可以确定模型的元数据。唯一的区别在于，我们不访问响应对象的`model_spec`属性，而是访问`metadata`。为了便于阅读，需要对信息进行序列化，因此我们将使用`SerializeToString`方法将协议缓冲区信息转换为人类可读的格式。

> `from``tensorflow_serving.apis``import``get_model_metadata_pb2``def``get_model_meta``(``model_name``,``stub``):``request``=``get_model_metadata_pb2``.``GetModelMetadataRequest``()``request``.``model_spec``.``name``=``model_name``request``.``metadata_field``.``append``(``"signature_def"``)``response``=``stub``.``GetModelMetadata``(``request``,``5``)``return``response``.``metadata``[``'signature_def'``]``model_name``=``'complaints_classification'``stub``=``create_grpc_stub``(``'localhost'``)``meta``=``get_model_meta``(``model_name``,``stub``)``print``(``meta``.``SerializeToString``()``.``decode``(``"utf-8"``,``'ignore'``))``# type.googleapis.com/tensorflow.serving.SignatureDefMap``# serving_default``# complaints_classification_input``#         input_1:0``#                2@``# complaints_classification_output(``# dense_1/Sigmoid:0``#                tensorflow/serving/predict`

gRPC 请求比 REST 请求更复杂；然而，在对性能要求较高的应用中，它们可以提供更快的预测性能。通过批量处理预测请求，我们还可以增加模型的预测性能。

批量推断请求

-   TensorFlow Serving 中批量推断请求的使用案例是最强大的功能之一。在模型训练过程中，批处理加速了我们的训练，因为我们可以并行计算我们的训练样本。同时，如果我们将批次的内存需求与 GPU 的可用内存匹配，还可以高效地使用计算硬件。

如果您在未启用批处理的情况下运行 TensorFlow Serving，如图 8-3 所示，则每个客户端请求都会单独并按顺序处理。例如，如果您对图像进行分类，您的第一个请求将在您的 CPU 或 GPU 上推断模型，然后才是第二个请求、第三个请求等。在这种情况下，我们未充分利用 CPU 或 GPU 的可用内存。

如图 8-4 所示，多个客户端可以请求模型预测，模型服务器将不同的客户端请求批处理为一个“批次”进行计算。通过这个批处理步骤推断每个请求可能需要比单个请求更长的时间，这是由于超时或批次限制。然而，与我们的训练阶段类似，我们可以并行计算批处理并在批处理计算完成后将结果返回给所有客户端。这比单个样本请求更有效地利用硬件。

![](img/00047.jpg)

图 8-3\. TensorFlow Serving 概述（未进行批处理）

![](img/00059.jpg)

图 8-4\. TensorFlow Serving 概述（带有批处理）

配置批处理预测

需要启用 TensorFlow Serving 的批处理预测，并为您的用例进行配置。您有五个配置选项：

`max_batch_size`

> > 此参数控制批次大小。大批量大小将增加请求的延迟，并可能导致耗尽 GPU 内存。小批量大小则会失去使用最佳计算资源的好处。

`batch_timeout_micros`

> > 此参数设置填充批次的最大等待时间。此参数对于限制推理请求的延迟非常有用。

`num_batch_threads`

> > 线程数量配置了可以并行使用多少个 CPU 或 GPU 核心。

`max_enqueued_batches`

> > 此参数设置了用于预测排队的最大批次数。这种配置有助于避免请求的不合理积压。如果达到最大数量，请求将返回错误而不是排队。

`pad_variable_length_inputs`

> > 此布尔参数确定是否对具有可变长度的输入张量进行填充，使所有输入张量的长度相同。

可想而知，为了实现最佳批处理设置参数需要进行一些调整，并且这取决于应用。如果您进行在线推理，应该旨在限制延迟。通常建议首先将`batch_timeout_micros`设置为 0，然后将超时调整为 10,000 微秒。相反，批处理请求将受益于较长的超时时间（毫秒到秒），以始终使用最佳性能的批处理大小。当达到`max_batch_size`或超时时，TensorFlow Serving 将对批次进行预测。

如果您为 CPU 预测配置 TensorFlow Serving，请将 `num_batch_threads` 设置为 CPU 核心数。如果您配置了 GPU 设置，请调整 `max_batch_size` 以实现 GPU 内存的最佳利用率。在调整配置时，请确保将 `max_enqueued_batches` 设置为一个大数，以避免某些请求提前返回而没有进行适当推理。

您可以将参数设置在文本文件中，如下例所示。在我们的示例中，我们创建了一个名为 `batching_parameters.txt` 的配置文件，并添加了以下内容：

> `max_batch_size` `{` `value:` `32``}` `batch_timeout_micros` `{` `value:` `5000``}` `pad_variable_length_inputs:` `true`

如果您想启用批处理，需要将两个额外的参数传递给运行 TensorFlow Serving 的 Docker 容器。要启用批处理，请将 `enable_batching` 设置为 true，并将 `batching_parameters_file` 设置为容器内批处理配置文件的绝对路径。请记住，如果批处理配置文件不位于与模型版本相同的文件夹中，则必须挂载额外的文件夹。

下面是使用启用批处理的 `docker run` 命令的完整示例，它启动了带有批处理功能的 TensorFlow Serving Docker 容器。然后，参数将传递给 TensorFlow Serving 实例：

> `docker run -p 8500:8500` `\` `-p 8501:8501` `\` `--mount` `type``=``bind``,source``=``/path/to/models,target``=``/models/my_model` `\` `--mount` `type``=``bind``,source``=``/path/to/batch_config,target``=``/server_config` `\` `-e` `MODEL_NAME``=``my_model -t tensorflow/serving` `\` `--enable_batching``=``true` `--batching_parameters_file``=``/server_config/batching_parameters.txt`

正如前文所述，批处理的配置需要额外的调整，但性能增益应该弥补初始设置的成本。我们强烈建议启用此 TensorFlow Serving 功能。这对于使用离线批处理过程推断大量数据样本尤为有用。

其他 TensorFlow Serving 优化

TensorFlow Serving 提供了多种额外的优化功能。其他特性标志包括：

`--file_system_poll_wait_seconds=1`

> > TensorFlow Serving 会定期检查是否有新的模型版本可用。您可以通过将其设置为`1`来禁用此功能。如果您只想加载模型一次而不更新它，可以将其设置为`0`。该参数期望一个整数值。如果您从云存储桶加载模型，我们强烈建议您增加轮询时间，以避免因频繁列出云存储桶上的操作而产生不必要的云服务提供商费用。

`--tensorflow_session_parallelism=0`

> > TensorFlow Serving 将自动确定在 TensorFlow 会话中使用多少线程。如果您希望手动设置线程数，可以通过将此参数设置为任意正整数值来覆盖它。

`--tensorflow_intra_op_parallelism=0`

> > 此参数设置了用于运行 TensorFlow Serving 的核心数。可用线程数确定将并行化多少操作。如果值为`0`，将使用所有可用核心。

`--tensorflow_inter_op_parallelism=0`

> > 此参数设置了在池中执行 TensorFlow 操作的可用线程数。这对于最大化 TensorFlow 图中独立操作的执行非常有用。如果值设置为`0`，将使用所有可用核心，并为每个核心分配一个线程。

与之前的示例类似，您可以将配置参数传递给`docker run`命令，如下例所示：

> `docker run -p 8500:8500` `\` `-p 8501:8501` `\` `--mount` `type``=``bind``,source``=``/path/to/models,target``=``/models/my_model` `\` `-e` `MODEL_NAME``=``my_model -t tensorflow/serving` `\` `--tensorflow_intra_op_parallelism``=``4``\` `--tensorflow_inter_op_parallelism``=``4``\` `--file_system_poll_wait_seconds``=``10``\` `--tensorflow_session_parallelism``=``2`

讨论的配置选项可以提高性能并避免不必要的云提供商费用。

TensorFlow Serving 替代方案

TensorFlow Serving 是部署机器学习模型的一个很好的方式。使用 TensorFlow 的`Estimator`和 Keras 模型，您应该可以涵盖大量的机器学习概念。但是，如果您想要部署传统模型或者您选择的机器学习框架不是 TensorFlow 或 Keras，这里有一些选择。

BentoML

[BentoML](https://bentoml.org) 是一个与框架无关的库，用于部署机器学习模型。它支持通过 PyTorch、scikit-learn、TensorFlow、Keras 和 XGBoost 训练的模型。对于 TensorFlow 模型，BentoML 支持`SavedModel`格式。BentoML 支持批处理请求。

Seldon

英国初创公司 Seldon 提供了多种开源工具来管理模型生命周期，其中核心产品之一是[Seldon Core](https://oreil.ly/Yx_U7)。Seldon Core 为您提供一个工具箱，用于将您的模型包装为 Docker 镜像，然后通过 Seldon 在 Kubernetes 集群中部署。

在撰写本章时，Seldon 支持用 TensorFlow、scikit-learn、XGBoost 甚至 R 训练的机器学习模型。

Seldon 自带其生态系统，允许将预处理构建到其自己的 Docker 镜像中，这些镜像与部署镜像一起部署。它还提供了一个路由服务，允许您执行 A/B 测试或多臂老虎机实验。

Seldon 与 Kubeflow 环境高度集成，并且与 TensorFlow Serving 类似，是在 Kubernetes 上部署模型的一种方式。

GraphPipe

[GraphPipe](https://oreil.ly/w_U7U) 是另一种部署 TensorFlow 和非 TensorFlow 模型的方法。Oracle 推动这个开源项目。它允许您部署不仅仅是 TensorFlow（包括 Keras）模型，还可以是 Caffe2 模型和所有可以转换为开放神经网络交换格式（ONNX）的机器学习模型。8 通过 ONNX 格式，您可以使用 GraphPipe 部署 PyTorch 模型。

除了为 TensorFlow、PyTorch 等提供模型服务器外，GraphPipe 还提供了 Python、Java 和 Go 等编程语言的客户端实现。

简单的 TensorFlow 服务

[简单 TensorFlow 服务](https://stfs.readthedocs.io) 是 4Paradigm 的陈迪豪开发的。这个库不仅仅支持 TensorFlow 模型。当前支持的模型框架列表包括 ONNX、scikit-learn、XGBoost、PMML 和 H2O。它支持多个模型，在 GPU 上进行预测，并提供各种语言的客户端代码。

简单 TensorFlow 服务的一个重要方面是它支持对模型服务器的身份验证和加密连接。身份验证目前不是 TensorFlow 服务的功能，支持 SSL 或传输层安全性（TLS）需要自定义构建 TensorFlow 服务。

MLflow

[MLflow](https://mlflow.org) 支持部署机器学习模型，但这只是由 DataBricks 创建的工具的一个方面。MLflow 旨在通过 MLflow Tracking 管理模型实验。该工具具有内置的模型服务器，为通过 MLflow 管理的模型提供 REST API 端点。

MLflow 还提供界面，可以直接将模型从 MLflow 部署到 Microsoft 的 Azure ML 平台和 AWS SageMaker。

Ray Serve

[Ray 项目](https://ray.io) 提供了部署机器学习模型的功能。Ray Serve 是框架无关的，支持 PyTorch、TensorFlow（包括 Keras）、Scikit-Learn 模型或自定义模型预测。该库具备批处理请求的能力，并允许在模型及其版本之间进行流量路由。

Ray Serve 已集成到 Ray 项目生态系统中，并支持分布式计算设置。

使用云服务提供商部署

到目前为止，我们讨论的所有模型服务器解决方案都必须由您安装和管理。但是，所有主要的云提供商——Google Cloud、AWS 和 Microsoft Azure——都提供包括托管机器学习模型在内的机器学习产品。

在本节中，我们将指导您通过 Google Cloud 的 AI 平台进行示例部署。让我们从模型部署开始，稍后我们将解释如何从您的应用程序客户端获取部署模型的预测。

使用案例

如果您希望无缝部署模型且不必担心扩展模型部署，则托管云部署是运行模型服务器实例的良好选择。所有云服务提供商都提供了通过推理请求数量扩展的部署选项。

然而，模型部署的灵活性也伴随着成本。托管服务提供了无忧的部署方式，但其费用较高。例如，两个全天候运行的模型版本（需要两个计算节点）的费用比运行 TensorFlow Serving 实例的可比计算实例更高。托管部署的另一个缺点是产品的限制。一些云服务提供商要求您通过它们自己的软件开发工具包进行部署，其他提供商则对节点大小和模型占用内存的限制。对于庞大的机器学习模型来说，特别是包含很多层的模型（即语言模型），这些限制可能是严重的限制。

使用 GCP 的部署示例

在本节中，我们将为您介绍如何在 Google Cloud 的 AI 平台上进行部署。与编写配置文件和执行终端命令不同，我们可以通过 Web UI 设置模型端点。

> GCP AI 平台的模型大小限制
> 
> GCP 的端点限制模型大小最高为 500 MB。但是，如果您通过 N1 类型的计算引擎部署端点，则最大模型限制增加到 2 GB。在撰写本文时，此选项作为测试版功能可用。

模型部署

部署包括三个步骤：

+   > > > > 使模型在 Google Cloud 上可访问。
+   > > > > 
+   > > > > 创建一个新的 Google Cloud AI 平台模型实例。
+   > > > > 
+   > > > > 创建一个包含模型实例的新版本。

部署始于将您的导出 TensorFlow 或 Keras 模型上传到存储桶。如图 8-5 所示，您需要上传整个导出模型。一旦模型上传完成，请复制存储位置的完整路径。

![](img/00069.jpg)

图 8-5\. 将训练好的模型上传到云存储

一旦您上传了您的机器学习模型，请转至 GCP 的 AI 平台，设置您的机器学习模型以进行部署。如果这是您在 GCP 项目中首次使用 AI 平台，则需要启用 API。Google Cloud 的自动启动过程可能需要几分钟。

如图 8-6 所示，您需要为模型提供一个唯一标识符。一旦创建了标识符，选择您首选的部署区域，9 并创建一个可选的项目描述，继续通过点击“创建”来完成设置。

![](img/00080.jpg)

图 8-6\. 创建一个新的模型实例

一旦注册新模型，该模型将显示在仪表板中，如 图 8-7 所示。您可以通过在溢出菜单中点击 “创建版本” 来为仪表板创建新模型版本。

![](img/00090.jpg)

图 8-7\. 创建新模型版本

创建新模型版本时，您需配置一个运行模型的计算实例。Google Cloud 提供多种配置选项，如 图 8-8 所示。`版本名称` 非常重要，因为稍后您将在客户端设置中引用 `版本名称`。请将 `模型 URI` 设置为您在先前步骤中保存的存储路径。

Google Cloud AI 平台支持多种机器学习框架，包括 XGBoost、scikit-learn 和自定义预测例程。

![](img/00100.jpg)

图 8-8\. 设置实例详细信息

GCP 还允许您配置模型实例在模型经历大量推理请求时的扩展方式。您可以选择手动扩展或自动扩展之间的两种扩展行为。

手动扩展为您提供了设置用于模型版本预测的节点数量的选项。相比之下，自动扩展功能使您能够根据端点需求调整实例数量。如果您的节点没有任何请求，节点数量甚至可能降至零。请注意，如果自动扩展将节点数量降至零，则需要一些时间来重新启动您的模型版本，以处理下一个请求命中模型版本端点。此外，如果在自动扩展模式下运行推理节点，则会按照 10 分钟的间隔计费。

一旦整个模型版本配置完成，Google Cloud 将为您启动实例。如果一切就绪以进行模型预测，则会在版本名称旁边看到绿色的检查图标，如 图 8-9 所示。

![](img/00112.jpg)

图 8-9\. 完成部署，新版本可用

您可以同时运行多个模型版本。在模型版本的控制面板中，您可以将一个版本设置为默认版本，任何未指定版本的推理请求将路由到指定的 “默认版本”。请注意，每个模型版本将托管在单独的节点上，并会积累 GCP 成本。

模型推理

由于 TensorFlow Serving 在 Google 内部经过了大量测试，并且在 GCP 幕后也得到了广泛使用，因此它也在 AI 平台中使用。您会注意到，AI 平台不仅仅使用了我们在 TensorFlow Serving 实例中看到的相同模型导出格式，而且负载的数据结构与之前看到的相同。

唯一的显著区别在于 API 连接。正如本节所述，您将通过处理请求认证的 GCP API 连接到模型版本。

要连接到 Google Cloud API，您需要使用以下命令安装`google-api-python-client`库：

> `$` `pip install google-api-python-client`

所有 Google 服务都可以通过服务对象连接。下面代码片段中的辅助函数突出了如何创建服务对象。Google API 客户端采用`服务名称`和`服务版本`，返回一个对象，通过返回的对象的方法提供所有 API 功能：

> `import``googleapiclient.discovery``def``_connect_service``():``return``googleapiclient``.``discovery``.``build``(``serviceName``=``"ml"``,``version``=``"v1"``)`

与我们之前的 REST 和 gRPC 示例类似，我们将推断数据嵌套在一个固定的`instances`键下，其中包含一个输入字典列表。我们创建了一个小助手函数来生成有效负载。此函数包含任何预处理，如果需要在推断之前修改输入数据：

> `def``_generate_payload``(``sentence``):``return``{``"instances"``:``[{``"sentence"``:``sentence``}]}`

在客户端创建的服务对象和生成的有效负载后，现在是时候从托管在 Google Cloud 上的机器学习模型请求预测了。

AI 平台服务的服务对象包含一个预测方法，接受`name`和`body`。`name`是一个路径字符串，包含您的 GCP 项目名称、模型名称，以及如果要使用特定模型版本进行预测，则版本名称。如果您不指定版本号，则将使用默认模型版本进行模型推断。`body`包含我们之前生成的推断数据结构：

> `project``=``"``yourGCPProjectName``"``model_name``=``"demo_model"``version_name``=``"v1"``request``=``service``.``projects``()``.``predict``(``name``=``"projects/{}/models/{}/versions/{}"``.``format``(``project``,``model_name``,``version_name``),``body``=``_generate_payload``(``sentence``)``)``response``=``request``.``execute``()`

Google Cloud AI 平台响应包含对不同类别的预测分数，类似于从 TensorFlow Serving 实例的 REST 响应：

> `{``'predictions'``:``[``{``'label'``:``[``0.9000182151794434``,``0.02840868942439556``,``0.009750653058290482``,``0.06182243302464485``]}``]}`

示范的部署选项是一种快速部署机器学习模型的方式，而无需设置整个部署基础设施。其他云提供商如 AWS 或 Microsoft Azure 提供类似的模型部署服务。根据您的部署要求，云提供商可能是自托管部署选项的良好替代方案。缺点可能包括更高的成本以及不完全优化端点（例如通过提供 gRPC 端点或批处理功能，正如我们在"批处理推断请求"中讨论的）。

使用 TFX Pipelines 进行模型部署

在本章的引言中，在 Figure 8-1 中展示了部署步骤作为机器学习流水线的一个组成部分。在讨论模型部署的内部工作原理，特别是 TensorFlow Serving 后，我们希望在本节中将机器学习流水线与此连接起来。

在 Figure 8-10 中，您可以看到持续模型部署的步骤。我们假设您已经运行并配置了 TensorFlow Serving，以从给定的文件位置加载模型。此外，我们假设 TensorFlow Serving 将从外部文件位置（例如云存储桶或挂载的持久卷）加载模型。TFX 管道和 TensorFlow Serving 实例两个系统需要访问相同的文件系统。

![](img/00007.jpg)

图 8-10\. TFX 管道生成的模型部署

在 “TFX Pusher Component” 中，我们讨论了 `Pusher` 组件。这个 TFX 组件允许我们将经过验证的模型推送到指定位置（例如，云存储桶）。TensorFlow Serving 可以从云存储位置获取新的模型版本，卸载之前的模型版本，并加载给定模型端点的最新版本。这是 TensorFlow Serving 的默认模型策略。

基于默认的模型策略，我们可以相对容易地使用 TFX 和 TensorFlow Serving 构建一个简单的持续部署设置。

摘要

在本章中，我们讨论了如何设置 TensorFlow Serving 来部署机器学习模型，以及为什么模型服务器比通过 Flask web 应用部署机器学习模型更具扩展性。我们详细介绍了安装和配置步骤，介绍了两种主要的通信选项，REST 和 gRPC，并简要讨论了这两种通信协议的优缺点。

此外，我们解释了 TensorFlow Serving 的一些重要优势，包括模型请求的批处理和获取不同模型版本元数据的能力。我们还讨论了如何通过 TensorFlow Serving 快速设置 A/B 测试。

我们在本章结尾简要介绍了托管云服务的概念，以 Google Cloud AI Platform 为例。托管云服务允许您部署机器学习模型，而无需管理自己的服务器实例。

在下一章中，我们将讨论如何增强我们的模型部署，例如通过从云提供商加载模型，或者通过 Kubernetes 部署 TensorFlow Serving。

> 1   有关应用实例，请访问[TensorFlow](https://oreil.ly/qCY6J)。
> 
> 2   模型标签集用于识别用于加载的 MetaGraph。一个模型可以导出为指定用于训练和服务的图形。这两个 MetaGraph 可以通过不同的模型标签提供。
> 
> 3   如果您之前没有安装或使用过 `Docker`，请查看我们在 附录 A 中的简要介绍。
> 
> 4   想要更详细地了解 REST 和 gRPC，请查看 “REST Versus gRPC”。
> 
> 5   模型的加载和卸载仅在 `file_system_poll_wait_seconds` 配置为大于 0 时才有效。默认配置为 2 秒。
> 
> 6   SSL 配置文件基于 SSL 配置协议缓冲区，可以在 [TensorFlow Serving API](https://oreil.ly/ZAEte) 中找到。
> 
> 7   A/B 测试如果没有对与两个模型交互产生的结果进行统计检验，是不完整的。所展示的实现仅提供了 A/B 测试的后端。
> 
> 8    [ONNX](https://onnx.ai) 是描述机器学习模型的一种方式。
> 
> 9   为了达到最低的预测延迟，请选择距离模型请求地理区域最近的区域。
