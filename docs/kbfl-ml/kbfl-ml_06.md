# 第五章：数据和特征准备

机器学习算法的好坏取决于它们的训练数据。获取用于训练的良好数据涉及数据和特征准备。

*数据准备* 是获取数据并确保其有效性的过程。这是一个多步骤的过程¹，可以包括数据收集、增强、统计计算、模式验证、异常值修剪以及各种验证技术。数据量不足可能导致过拟合，错过重要的相关性等问题。在数据准备阶段投入更多精力收集更多的记录和每个样本的信息可以显著提高模型的效果。²

*特征准备*（有时称为*特征工程*）是将原始输入数据转换为机器学习模型可以使用的特征的过程。³ 糟糕的特征准备可能会导致丢失重要的关系，例如线性模型未展开非线性项，或者深度学习模型中图像方向不一致。

数据和特征准备的微小变化可能导致显著不同的模型输出。迭代方法对于特征和数据准备都是最佳选择，随着对问题和模型理解的深入，需要重新访问它们。Kubeflow Pipelines 使我们能够更容易地迭代我们的数据和特征准备。我们将探讨如何使用超参数调整在第十章中迭代。

在本章中，我们将涵盖数据和特征准备的不同方法，并演示如何通过使用流水线使它们可重复。我们假设您已经熟悉本地工具。因此，我们将从如何为流水线构建本地代码开始，并转向更可扩展的分布式工具。一旦我们探索了这些工具，我们将根据“介绍我们的案例研究”中的示例将它们组合成一个流水线。

# 决定正确的工具

有各种各样的数据和特征准备工具。⁴ 我们可以将它们分类为分布式和本地工具。本地工具在单台机器上运行，并提供很大的灵活性。分布式工具在多台机器上运行，因此可以处理更大更复杂的任务。在工具选择上做出错误决策可能需要后续大幅修改代码。

如果输入数据规模相对较小，单台机器可以提供你所需要的所有工具。更大的数据规模往往需要整个流水线或仅作为抽样阶段的分布式工具。即使是对于较小的数据集，像 Apache Spark、Dask 或 TFX with Beam 这样的分布式系统也可能更快，但可能需要学习新的工具。⁵

不必为所有数据和特征准备活动使用相同的工具。在处理使用相同工具会很不方便的不同数据集时，使用多个工具尤为常见。Kubeflow Pipelines 允许您将实现拆分为多个步骤并连接它们（即使它们使用不同的语言），形成一个连贯的系统。

# 本地数据和特征准备

在本地工作限制了数据的规模，但提供了最全面的工具范围。实施数据和特征准备的常见方法是使用 Jupyter 笔记本。在第四章中，我们介绍了如何将笔记本的部分转换为管道，这里我们将看看如何结构化我们的数据和特征准备代码，使其变得简单易行。

使用笔记本进行数据准备可以是开始探索数据的好方法。笔记本在这个阶段特别有用，因为我们通常对数据了解最少，而使用可视化工具来理解我们的数据可能非常有益。

## 获取数据

对于我们的邮件列表示例，我们使用来自互联网公共档案的数据。理想情况下，您希望连接到数据库、流或其他数据存储库。然而，即使在生产中，获取网络数据也可能是必要的。首先，我们将实现我们的数据获取算法，该算法获取 Apache 软件基金会（ASF）项目的电子邮件列表位置以及要获取消息的年份。示例 5-1 返回所获取记录的路径，因此我们可以将其用作下一个管道阶段的输入。

###### 注意

函数下载至多一年的数据，并在调用之间休眠。这是为了防止过度使用 ASF 邮件存档服务器。ASF 是一家慈善组织，请在下载数据时注意这一点，不要滥用此服务。

##### 示例 5-1\. 下载邮件列表数据

```
def download_data(year: int) -> str:

  # The imports are inline here so Kubeflow can serialize the function correctly.
  from datetime import datetime
  from lxml import etree
  from requests import get
  from time import sleep

 import json

  def scrapeMailArchives(mailingList: str, year: int, month: int):
      #Ugly xpath code goes here. See the example repo if you're curious.

   datesToScrape =  [(year, i) for i in range(1,2)]

   records = []
   for y,m in datesToScrape:
     print(m,"-",y)
     records += scrapeMailArchives("spark-dev", y, m)
   output_path = '/data_processing/data.json'
   with open(output_path, 'w') as f:
     json.dump(records, f)

   return output_path
```

此代码下载给定年份的所有邮件列表数据，并将其保存到已知路径。在本例中，需要挂载持久卷，以便在制作管道时使数据在各个阶段之间流动。

作为机器学习管道的一部分，您可能会有数据转储，或者可能会由不同的系统或团队提供。对于 GCS 或 PV 上的数据，您可以使用内置组件 `google-cloud/storage/download` 或 `filesystem/get_subdirectory` 来加载数据，而不是编写自定义函数。

## 数据清理：过滤掉垃圾

现在我们加载了数据，是时候进行一些简单的数据清理了。本地工具更为常见，因此我们将首先专注于它们。尽管数据清理通常取决于领域专业知识，但也有标准工具可用于协助常见任务。首先的步骤可以是通过检查模式验证输入记录。也就是说，我们检查字段是否存在并且类型正确。

要检查邮件列表示例中的模式，我们确保发送者、主题和正文都存在。为了将其转换为独立组件，我们将使我们的函数接受输入路径的参数，并返回已清理记录的文件路径。实现这个功能所需的代码量相对较小，如示例 5-2 所示。

##### 示例 5-2\. 数据清理

```
def clean_data(input_path: str) -> str:
    import json
    import pandas as pd

    print("loading records...")
    with open(input_path, 'r') as f:
        records = json.load(f)
    print("records loaded")

    df = pd.DataFrame(records)
    # Drop records without a subject, body, or sender
    cleaned = df.dropna(subset=["subject", "body", "from"])

    output_path_hdf = '/data_processing/clean_data.hdf'
    cleaned.to_hdf(output_path_hdf, key="clean")

    return output_path_hdf
```

除了丢弃缺失字段之外，还有许多其他标准数据质量技术。其中两种较受欢迎的是补全缺失数据⁶和分析并移除可能是不正确测量结果的离群值。无论您决定执行哪些附加的通用技术，您都可以简单地将它们添加到您的数据清理函数中。

领域特定的数据清理工具也可能是有益的。在邮件列表示例中，我们数据中的一个潜在噪声来源可能是垃圾邮件。解决此问题的一种方法是使用 SpamAssassin。我们可以像示例 5-3 中所示那样将此包添加到我们的容器中。在笔记本镜像之上添加不由 pip 管理的系统软件有点更加复杂，因为权限问题。大多数容器以 root 用户身份运行，可以简单地安装新的系统软件包。然而，由于 Jupyter，笔记本容器以较低权限用户身份运行。像这样安装新包需要切换到 root 用户然后再切换回去，这在其他 Dockerfile 中并不常见。

##### 示例 5-3\. 安装 SpamAssassin

```
ARG base
FROM $base
# Run as root for updates
USER root
# Install SpamAssassin
RUN apt-get update && \
    apt-get install -yq spamassassin spamc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/apt
# Switch back to the expected user
USER jovyan
```

创建完这个 Dockerfile 后，你会希望构建并将生成的镜像推送到 Kubeflow 集群可以访问的某个地方，例如示例 2-8。

推送一个新的容器还不足以让 Kubeflow 知道我们想要使用它。当使用 `func_to_container_op` 构建管道阶段时，您需要在 `func_to_container_op` 函数调用中指定 `base_image` 参数。我们将在示例 5-35 中将此示例作为管道展示。

这里我们再次看到容器的强大。您可以在 Kubeflow 提供的基础上添加我们需要的工具，而不是从头开始制作一切。

数据清理完成后，就该确保数据充足，或者如果不充足，探索增加数据。

## 格式化数据

正确的格式取决于您用于特征准备的工具。如果您继续使用用于数据准备的同一工具，则输出可以与输入相同。否则，您可能会发现这是更改格式的好地方。例如，当使用 Spark 进行数据准备并使用 TensorFlow 进行训练时，我们经常在这里实现转换为 TFRecords。

## 特征准备

如何进行特征准备取决于问题的性质。在邮件列表示例中，我们可以编写各种文本处理函数，并将它们组合成特征，如示例 5-4 所示。

##### 示例 5-4\. 将文本处理函数编写并组合为特征

```
    df['domains'] = df['links'].apply(extract_domains)
    df['isThreadStart'] = df['depth'] == '0'

    # Arguably, you could split building the dataset away from the actual witchcraft.
    from sklearn.feature_extraction.text import TfidfVectorizer

    bodyV = TfidfVectorizer()
    bodyFeatures = bodyV.fit_transform(df['body'])

    domainV = TfidfVectorizer()

    def makeDomainsAList(d):
        return ' '.join([a for a in d if not a is None])

    domainFeatures = domainV.fit_transform(
        df['domains'].apply(makeDomainsAList))

    from scipy.sparse import csr_matrix, hstack

    data = hstack([
        csr_matrix(df[[
            'containsPythonStackTrace', 'containsJavaStackTrace',
            'containsExceptionInTaskBody', 'isThreadStart'
        ]].to_numpy()), bodyFeatures, domainFeatures
    ])
```

到目前为止，示例代码结构使您可以将每个函数转换为单独的管道阶段；然而，还有其他选项。我们将查看如何将整个笔记本作为管道阶段使用，在“将其放入管道中”中。

当然，除了笔记本和 Python 之外还有其他数据准备工具。笔记本并不总是最好的工具，因为它们在版本控制方面可能存在困难。Python 并不总是具有您所需的库（或性能）。因此，我们现在将看看如何使用其他可用工具。

## 定制容器

管道不仅仅限于笔记本，甚至不限于特定语言。⁷ 根据项目的不同，您可能有一个常规的 Python 项目、定制工具、Python 2，甚至是 FORTRAN 代码作为一个重要组成部分。

例如，在第九章中，我们将使用 Scala 来执行管道中的一个步骤。此外，在“使用 RStats”中，我们讨论如何开始使用一个 RStats 容器。

有时候您可能无法找到一个与我们这里所需如此相符的容器。在这些情况下，您可以采用一个通用的基础镜像并在其上构建，我们将在第九章中更详细地讨论这一点。

除了需要定制容器之外，您可能选择放弃笔记本的另一个原因是探索分布式工具。

# 分布式工具

使用分布式平台可以处理大型数据集（超出单个机器内存）并可以提供显著更好的性能。通常，当我们的问题已经超出初始笔记本解决方案时，我们需要开始使用分布式工具。

Kubeflow 中的两个主要数据并行分布式系统是 Apache Spark 和 Google 的 Dataflow（通过 Apache Beam）。Apache Spark 拥有更大的安装基础和支持的格式和库的种类。Apache Beam 支持 TensorFlow Extended（TFX），这是一个端到端的 ML 工具，可以与 TFServing 集成以进行模型推断。由于其集成性最强，我们将首先探索在 Apache Beam 上使用 TFX，然后继续使用更标准的 Apache Spark。

## TensorFlow Extended

TensorFlow 社区为从数据验证到模型服务的一切创建了一套优秀的集成工具。目前，TFX 的数据工具都是基于 Apache Beam 构建的，这是 Google Cloud 上分布式处理支持最多的工具。如果您想使用 Kubeflow 的 TFX 组件，目前仅限于单节点；这在未来的版本中可能会改变。

###### 注意

Apache Beam 在 Google Cloud Dataflow 之外的 Python 支持尚不成熟。TFX 是一个 Python 工具，因此其扩展取决于 Apache Beam 的 Python 支持。您可以通过使用仅 GCP 的 Dataflow 组件来扩展作业。随着 Apache Beam 对 Apache Flink 和 Spark 的支持改进，可能会添加对可移植方式扩展 TFX 组件的支持。⁸

Kubeflow 在其管道系统中包含许多 TFX 组件。TFX 还有其自己的管道概念。这些与 Kubeflow 管道是分开的，在某些情况下，TFX 可以作为 Kubeflow 的替代方案。在这里，我们将重点放在数据和特征准备组件上，因为这些是与 Kubeflow 生态系统的其他部分最简单配合使用的组件。

### 保持数据质量：TensorFlow 数据验证

确保数据质量不会随时间而下降至关重要。数据验证允许我们确保数据的架构和分布仅以预期的方式发展，并在它们变成生产问题之前捕捉数据质量问题。TensorFlow 数据验证（TFDV）使我们能够验证我们的数据。

为了使开发过程更加简单，您应该在本地安装 TFX 和 TFDV。虽然代码可以在 Kubeflow 内部评估，但在本地拥有库会加快开发工作的速度。安装 TFX 和 TFDV 只需使用 pip install 命令，如 Example 5-5 所示。

##### Example 5-5\. 安装 TFX 和 TFDV

```
pip3 install tfx tensorflow-data-validation
```

现在让我们看看如何在 Kubeflow 的管道中使用 TFX 和 TFDV。第一步是加载我们想要使用的相关组件。正如我们在前一章中讨论的，虽然 Kubeflow 确实有一个`load_component`函数，但它自动解析在主分支上，因此不适合生产用例。因此，我们将使用`load_component_from_file`以及从 Example 4-15 下载的 Kubeflow 组件的本地副本来加载我们的 TFDV 组件。我们需要加载的基本组件包括：示例生成器（即数据加载器）、模式、统计生成器和验证器本身。加载组件的示例如 Example 5-6 所示。

##### Example 5-6\. 加载组件

```
tfx_csv_gen = kfp.components.load_component_from_file(
    "pipelines-0.2.5/components/tfx/ExampleGen/CsvExampleGen/component.yaml")
tfx_statistic_gen = kfp.components.load_component_from_file(
    "pipelines-0.2.5/components/tfx/StatisticsGen/component.yaml")
tfx_schema_gen = kfp.components.load_component_from_file(
    "pipelines-0.2.5/components/tfx/SchemaGen/component.yaml")
tfx_example_validator = kfp.components.load_component_from_file(
    "pipelines-0.2.5/components/tfx/ExampleValidator/component.yaml")
```

除了组件之外，我们还需要我们的数据。当前的 TFX 组件通过 Kubeflow 的文件输出机制在管道阶段之间传递数据。这将输出放入 MinIO，自动跟踪与管道相关的工件。为了在推荐示例的输入上使用 TFDV，我们首先使用标准容器操作下载它，就像在 Example 5-7 中所示。

##### Example 5-7\. 下载推荐数据

```
    fetch = kfp.dsl.ContainerOp(name='download',
                                image='busybox',
                                command=['sh', '-c'],
                                arguments=[
                                    'sleep 1;'
                                    'mkdir -p /tmp/data;'
                                    'wget ' + data_url +
                                    ' -O /tmp/data/results.csv'
                                ],
                                file_outputs={'downloaded': '/tmp/data'})
    # This expects a directory of inputs not just a single file
```

###### Tip

如果我们的数据在持久卷上（比如说，在之前的阶段获取的数据），我们可以使用`filesystem/get_file`组件。

一旦数据加载完成，TFX 有一组称为示例生成器的工具来摄取数据。这些支持几种不同的格式，包括 CSV 和 TFRecord。还有其他系统的示例生成器，包括 Google 的 BigQuery。与 Spark 或 Pandas 支持的各种格式不同，可能需要使用另一工具预处理记录。¹⁰ 在我们的推荐示例中，我们使用了 CSV 组件，如示例 5-8 所示。

##### 示例 5-8\. 使用 CSV 组件

```
    records_example = tfx_csv_gen(input_base=fetch.output)
```

现在我们有了示例的通道，可以将其作为 TFDV 的输入之一。创建模式的推荐方法是使用 TFDV 推断模式。为了能够推断模式，TFDV 首先需要计算数据的一些摘要统计信息。示例 5-9 展示了这两个步骤的管道阶段。

##### 示例 5-9\. 创建模式

```
    stats = tfx_statistic_gen(input_data=records_example.output)
    schema_op = tfx_schema_gen(stats.output)
```

如果每次都推断模式，我们可能无法捕捉模式变化。相反，你应该保存模式，并在将来的运行中重复使用它进行验证。流水线的运行网页有指向 MinIO 中模式的链接，你可以使用另一个组件或容器操作，获取或复制它到其他地方。

不管你将模式持久化到何处，你都应该检查它。要检查模式，你需要导入 TFDV 库，如示例 5-10 所示。在开始使用模式验证数据之前，你应该先检查模式。要检查模式，请在本地下载模式（或者笔记本上）并使用 TFDV 的`display_schema`函数，如示例 5-11 所示。

##### 示例 5-10\. 在本地下载模式

```
import tensorflow_data_validation as tfdv
```

##### 示例 5-11\. 显示模式

```
schema = tfdv.load_schema_text("schema_info_2")
tfdv.display_schema(schema)
```

如果需要，可以从[TensorFlow GitHub repo](https://oreil.ly/qjHeI)下载`schema_util.py`脚本，提供修改模式的工具（不论是为了演化或者纠正推断错误）。

现在我们知道我们正在使用正确的模式，让我们验证我们的数据。验证组件接受我们生成的模式和统计数据，如示例 5-12 所示。在生产时，你应该用它们的输出替换模式和统计生成组件。

##### 示例 5-12\. 验证数据

```
    tfx_example_validator(stats=stats.outputs['output'],
                          schema=schema_op.outputs['output'])
```

###### 提示

在推送到生产之前，检查被拒绝记录的大小。你可能会发现数据格式已经改变，需要使用模式演化指南，并可能更新其余的流水线。

### TensorFlow Transform，与 TensorFlow Extended 一起使用 Beam

用于进行特征准备的 TFX 程序称为 TensorFlow Transform（TFT），并集成到 TensorFlow 和 Kubeflow 生态系统中。与 TFDV 一样，Kubeflow 的 TensorFlow Transform 组件目前无法扩展到单节点处理之外。TFT 的最大好处是其集成到 TensorFlow 模型分析工具中，简化了推断过程中的特征准备。

我们需要指定 TFT 要应用的转换。我们的 TFT 程序应该在一个与管道定义分离的文件中，虽然也可以作为字符串内联。首先，我们需要一些标准的 TFT 导入，如示例 5-13 所示。

##### 示例 5-13\. TFT 导入

```
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils
```

现在我们已经导入了所需的模块，是时候为组件创建入口点了，如示例 5-14 所示。

##### 示例 5-14\. 创建入口点

```
def preprocessing_fn(inputs):
```

在这个函数内部，我们进行数据转换以生成我们的特征。并非所有特征都需要转换，这就是为什么还有一个复制方法来将输入镜像到输出，如果你只是添加特征的话。在我们的邮件列表示例中，我们可以计算词汇表，如示例 5-15 所示。

##### 示例 5-15\. 计算词汇表

```
    outputs = {}
    # TFT business logic goes here
    outputs["body_stuff"] = tft.compute_and_apply_vocabulary(inputs["body"],
                                                             top_k=1000)
    return outputs
```

此函数不支持任意的 Python 代码。所有的转换必须表达为 TensorFlow 或 TensorFlow Transform 操作。TensorFlow 操作一次操作一个张量，但在数据准备中，我们通常希望对所有输入数据进行某些计算，而 TensorFlow Transform 的操作使我们能够实现这一点。请参阅[TFT Python 文档](https://oreil.ly/4j0mv)或调用`help(tft)`以查看一些起始操作。

一旦您编写了所需的转换，就可以将它们添加到管道中了。这样做的最简单方法是使用 Kubeflow 的`tfx/Transform`组件。加载该组件与其他 TFX 组件相同，如示例 5-6 所示。使用此组件的独特之处在于需要将转换代码作为上传到 S3 或 GCS 的文件传递给它。它还需要数据，您可以使用 TFDV 的输出（如果使用了 TFDV），或者像我们为 TFDV 所做的那样加载示例。使用 TFT 组件的示例如示例 5-16 所示。

##### 示例 5-16\. 使用 TFT 组件

```
    transformed_output = tfx_transform(
        input_data=records_example.output,
        schema=schema_op.outputs['output'],
        module_file=module_file)  # Path to your TFT code on GCS/S3
```

现在，您拥有一个包含特征准备及在服务时间转换请求的机器学习管道的关键工件。TensorFlow Transform 的紧密集成可以使模型服务变得不那么复杂。TensorFlow Transform 与 Kubeflow 组件结合使用并不能为所有项目提供足够的能力，因此我们将在接下来看看分布式特征准备。

## 使用 Apache Spark 进行分布式数据处理

Apache Spark 是一个开源的分布式数据处理工具，可以在各种集群上运行。Kubeflow 通过几个不同的组件支持 Apache Spark，以便您可以访问特定于云的功能。由于您可能对 Spark 不太熟悉，我们将在数据和特征准备的背景下简要介绍 Spark 的 Dataset/Dataframe API。如果想要超越基础知识，我们推荐[*Learning Spark*](https://learning.oreilly.com/library/view/learning-spark/9781449359034)、[*Spark: The Definitive Guide*](https://learning.oreilly.com/library/view/spark-the-definitive/9781491912201)或[*High Performance Spark*](https://learning.oreilly.com/library/view/high-performance-spark/9781491943199)作为提升 Spark 技能的资源。

###### 注意

在我们的代码中，为了进行所有特征和数据准备，我们将其结构化为单一阶段，因为一旦达到规模，写入和加载数据之间的步骤是昂贵的。

### Kubeflow 中的 Spark 操作者

一旦您超越了实验阶段，使用 Kubeflow 的本地 Spark operator EMR 或 Dataproc 是最佳选择。最具可移植性的操作者是本地 Spark operator，它不依赖于任何特定的云。要使用任何操作者，您需要打包 Spark 程序并将其存储在分布式文件系统（如 GCS、S3 等）中或将其放入容器中。

如果您使用 Python 或 R，我们建议构建一个 Spark 容器以便安装依赖项。对于 Scala 或 Java 代码来说，这不那么关键。如果将应用程序放入容器中，可以使用`local:///`来引用它。您可以使用 gcr.io/spark-operator/spark-py:v2.4.5 容器作为基础，或者构建您自己的容器——遵循 Spark 在 Kubernetes 上的说明，或查看第九章。示例 5-19 展示了如何安装任何依赖项并复制应用程序。如果决定更新应用程序，仍可以使用容器，只需配置主资源为分布式文件系统。

我们在第九章中还涵盖了如何构建自定义容器。

##### 示例 5-19\. 安装要求并复制应用程序

```
# Use the Spark operator image as base
FROM gcr.io/spark-operator/spark-py:v2.4.5
# Install Python requirements
COPY requirements.txt /
RUN pip3 install -r /requirements.txt
# Now you can reference local:///job/my_file.py
RUN mkdir -p /job
COPY *.py /job

ENTRYPOINT ["/opt/entrypoint.sh"]
```

在 Kubeflow 中运行 Spark 的两个特定于云的选项是 Amazon EMR 和 Google Dataproc 组件。然而，它们各自接受不同的参数，这意味着您需要调整您的流水线。

EMR 组件允许您设置集群、提交作业以及清理集群。两个集群任务组件分别是`aws/emr/create_cluster`用于启动和`aws/emr/delete_cluster`用于删除。用于运行 PySpark 作业的组件是`aws/emr/submit_pyspark_job`。如果您不是在重用外部集群，则无论 submit_pyspark_job 组件是否成功，触发删除组件都非常重要。

尽管它们具有不同的参数，但 Dataproc 集群的工作流程与 EMR 类似。组件的命名类似，使用 `gcp/dataproc/create_cluster/` 和 `gcp/dataproc/delete_cluster/` 来管理生命周期，并使用 `gcp/dataproc/submit_pyspark_job/` 运行我们的作业。

与 EMR 和 Dataproc 组件不同，Spark 运算符没有组件。对于没有组件的 Kubernetes 运算符，您可以使用 `dsl.ResourceOp` 调用它们。示例 5-20 展示了使用 ResourceOp 启动 Spark 作业。

##### 示例 5-20\. 使用 ResourceOp 启动 Spark 作业

```
resource = {
    "apiVersion": "sparkoperator.k8s.io/v1beta2",
    "kind": "SparkApplication",
    "metadata": {
        "name": "boop",
        "namespace": "kubeflow"
    },
    "spec": {
        "type": "Python",
        "mode": "cluster",
        "image": "gcr.io/boos-demo-projects-are-rad/kf-steps/kubeflow/myspark",
        "imagePullPolicy": "Always",
        # See the Dockerfile OR use GCS/S3/...
        "mainApplicationFile": "local:///job/job.py",
        "sparkVersion": "2.4.5",
        "restartPolicy": {
            "type": "Never"
        },
        "driver": {
            "cores": 1,
            "coreLimit": "1200m",
            "memory": "512m",
            "labels": {
                "version": "2.4.5",
            },
            # also try spark-operatoroperator-sa
            "serviceAccount": "spark-operatoroperator-sa",
        },
        "executor": {
            "cores": 1,
            "instances": 2,
            "memory": "512m"
        },
        "labels": {
            "version": "2.4.5"
        },
    }
}

@dsl.pipeline(name="local Pipeline", description="No need to ask why.")
def local_pipeline():

    rop = dsl.ResourceOp(
        name="boop",
        k8s_resource=resource,
        action="create",
        success_condition="status.applicationState.state == COMPLETED")
```

###### 警告

Kubeflow 对 ResourceOp 请求不执行任何验证。例如，在 Spark 中，作业名称必须能够用作有效 DNS 名称的开头，而在容器操作中，容器名称会被重写，但 ResourceOps 只是直接通过请求。

### 读取输入数据

Spark 支持多种数据源，包括（但不限于）：Parquet、JSON、JDBC、ORC、JSON、Hive、CSV、ElasticSearch、MongoDB、Neo4j、Cassandra、Snowflake、Redis、Riak Time Series 等¹¹。加载数据非常简单，通常只需要指定格式。例如，在我们的邮件列表示例中，读取我们数据准备阶段的 Parquet 格式输出就像 示例 5-25 中所示。

##### 示例 5-25\. 读取我们数据的 Parquet 格式输出

```
initial_posts = session.read.format("parquet").load(fs_prefix +
                                                    "/initial_posts")
ids_in_reply = session.read.format("parquet").load(fs_prefix + "/ids_in_reply")
```

如果它被格式化为 JSON，我们只需要将 “parquet” 更改为 “JSON”¹²。

### 验证架构的有效性

我们通常认为我们了解数据的字段和类型。Spark 可以快速发现数据的架构，当我们的数据是自描述格式（如 Parquet）时。在其他格式（如 JSON）中，直到 Spark 读取记录时，架构才会知道。无论数据格式如何，指定架构并确保数据匹配是良好的做法，如 示例 5-26 中所示。与在模型部署期间出现的错误相比，数据加载期间的错误更容易调试。

##### 示例 5-26\. 指定架构

```
ids_schema = StructType([
    StructField("In-Reply-To", StringType(), nullable=True),
    StructField("message-id", StringType(), nullable=True)
])
ids_in_reply = session.read.format("parquet").schema(ids_schema).load(
    fs_prefix + "/ids_in_reply")
```

您可以配置 Spark 处理损坏和不符合规范的记录，通过删除它们、保留它们或停止过程（即失败作业）。默认为宽容模式，保留无效记录同时设置字段为空，允许我们使用相同的技术处理缺失字段来处理架构不匹配。

### 处理缺失字段

在许多情况下，我们的数据中会有一些缺失。您可以选择删除缺少字段的记录，回退到次要字段，填充平均值，或者保持原样。Spark 的内置工具用于这些任务在 `DataFrameNaFunctions` 中。正确的解决方案取决于您的数据和最终使用的算法。最常见的是删除记录并确保我们没有过多地筛选记录，这在邮件列表数据中使用 示例 5-27 进行了说明。

##### 示例 5-27\. 删除记录

```
initial_posts_count = initial_posts.count()
initial_posts_cleaned = initial_posts.na.drop(how='any',
                                              subset=['body', 'from'])
initial_posts_cleaned_count = initial_posts_cleaned.count()
```

### 过滤掉坏数据

检测不正确的数据可能是具有挑战性的。然而，如果不进行至少一些数据清洗，模型可能会在噪声中训练。通常，确定坏数据取决于从业者对问题的领域知识。

在 Spark 中支持的常见技术是异常值移除。然而，简单应用这一技术可能会移除有效记录。利用你的领域经验，你可以编写自定义验证函数，并使用 Spark 的`filter`函数删除任何不符合条件的记录，就像我们在示例 5-28 中的邮件列表示例中所示。

##### 示例 5-28\. 过滤掉坏数据

```
def is_ok(post):
    # Your special business logic goes here
    return True

spark_mailing_list_data_cleaned = spark_mailing_list_data_with_date.filter(
    is_ok)
```

### 保存输出

当数据准备好后，是时候保存输出了。如果你将使用 Apache Spark 进行特征准备，现在可以跳过此步骤。

如果你想返回到单机工具，将数据保存到持久存储通常是最简单的。为此，通过调用`toPandas()`将数据带回主程序，就像在示例 5-30 中展示的那样。现在你可以按照下一个工具期望的格式保存数据了。

##### 示例 5-30\. 保存到持久存储

```
initial_posts.toPandas()
```

如果数据量大，或者你想使用对象存储，Spark 可以写入多种不同的格式（就像它可以从多种不同的格式加载一样）。正确的格式取决于你打算用于特征准备的工具。在示例 5-31 中展示了写入 Parquet 格式的方法。

##### 示例 5-31\. 写入 Parquet 格式

```
initial_posts.write.format("parquet").mode('overwrite').save(fs_prefix +
                                                             "/initial_posts")
ids_in_reply.write.format("parquet").mode('overwrite').save(fs_prefix +
                                                            "/ids_in_reply")
```

现在你已经看到了各种可以用来获取和清理数据的工具。我们已经看到了本地工具的灵活性，分布式工具的可扩展性以及来自 TensorFlow Extended 的集成。数据形状已经到位，现在让我们确保正确的特征可用，并以可用于机器学习模型的格式获取它们。

## 使用 Apache Spark 进行分布式特征准备

Apache Spark 拥有大量内置的特征准备工具，在`pyspark.ml.feature`中，你可以使用这些工具生成特征。你可以像在数据准备阶段一样使用 Spark。你可能会发现使用 Spark 自带的 ML 管道是将多个特征准备阶段组合在一起的一种简便方法。

对于 Spark 邮件列表示例，我们有文本输入数据。为了能够训练多种模型，将其转换为词向量是我们首选的特征准备形式。这涉及首先使用 Spark 的分词器对数据进行分词。一旦有了这些标记，我们可以训练一个 Word2Vec 模型并生成我们的词向量。示例 5-32 展示了如何使用 Spark 为邮件列表示例准备特征。

##### 示例 5-32\. 准备邮件列表的特征

```
tokenizer = Tokenizer(inputCol="body", outputCol="body_tokens")
body_hashing = HashingTF(inputCol="body_tokens",
                         outputCol="raw_body_features",
                         numFeatures=10000)
body_idf = IDF(inputCol="raw_body_features", outputCol="body_features")
body_word2Vec = Word2Vec(vectorSize=5,
                         minCount=0,
                         numPartitions=10,
                         inputCol="body_tokens",
                         outputCol="body_vecs")
assembler = VectorAssembler(inputCols=[
    "body_features", "body_vecs", "contains_python_stack_trace",
    "contains_java_stack_trace", "contains_exception_in_task"
],
                            outputCol="features")
```

通过这个最终的分布式特征准备示例，你可以准备好扩展以处理更大的数据量（如果它们碰巧出现）。如果你处理的是更小的数据，你已经看到了如何使用容器化的简单技术继续使用你喜欢的工具。无论哪种方式，你几乎准备好进入机器学习管道的下一阶段。

# 在管道中将它放在一起

我们已经展示了如何解决数据和特征准备中的个别问题，但现在我们需要把它们整合起来。在我们的本地示例中，我们编写了带有类型和返回参数的函数，以便轻松地放入管道中。由于我们在每个阶段返回输出路径，我们可以使用函数输出来为我们创建依赖关系图。将这些函数放入管道中的示例在 示例 5-33 中说明。

##### 示例 5-33\. 将函数放在一起

```
@kfp.dsl.pipeline(name='Simple1', description='Simple1')
def my_pipeline_mini(year: int):
    dvop = dsl.VolumeOp(name="create_pvc",
                        resource_name="my-pvc-2",
                        size="5Gi",
                        modes=dsl.VOLUME_MODE_RWO)
    tldvop = dsl.VolumeOp(name="create_pvc",
                          resource_name="tld-volume-2",
                          size="100Mi",
                          modes=dsl.VOLUME_MODE_RWO)
    download_data_op = kfp.components.func_to_container_op(
        download_data, packages_to_install=['lxml', 'requests'])
    download_tld_info_op = kfp.components.func_to_container_op(
        download_tld_data,
        packages_to_install=['requests', 'pandas>=0.24', 'tables'])
    clean_data_op = kfp.components.func_to_container_op(
        clean_data, packages_to_install=['pandas>=0.24', 'tables'])

    step1 = download_data_op(year).add_pvolumes(
        {"/data_processing": dvop.volume})
    step2 = clean_data_op(input_path=step1.output).add_pvolumes(
        {"/data_processing": dvop.volume})
    step3 = download_tld_info_op().add_pvolumes({"/tld_info": tldvop.volume})

kfp.compiler.Compiler().compile(my_pipeline_mini, 'local-data-prep-2.zip')
```

您可以看到这里的特征准备步骤遵循了所有本地组件的相同一般模式。然而，我们用于特征准备的库有些不同，所以我们已经将 `packages_to_install` 的值更改为安装 Scikit-learn，如 示例 5-34 所示。

##### 示例 5-34\. 安装 Scikit-learn

```
    prepare_features_op = kfp.components.func_to_container_op(
        prepare_features,
        packages_to_install=['pandas>=0.24', 'tables', 'scikit-learn'])
```

###### 提示

当你开始探索一个新的数据集时，你可能会发现，像往常一样使用笔记本会更容易，而不使用管道组件。在可能的情况下，遵循与管道相同的一般结构将使得将你的探索工作投入到生产中更加容易。

这些步骤没有指定要使用的容器。对于你刚刚构建的 SpamAssassin 容器，你可以按照 示例 5-35 的方式编写。

##### 示例 5-35\. 指定一个容器

```
clean_data_op = kfp.components.func_to_container_op(
    clean_data,
    base_image="{0}/kubeflow/spammassisan".format(container_registry),
    packages_to_install=['pandas>=0.24', 'tables'])
```

有时候，在各个阶段之间写入数据的成本太高。在我们的推荐系统示例中，与邮件列表示例不同，我们选择将数据和特征准备放在一个单一的管道阶段中。在我们的分布式邮件列表示例中，我们也构建了一个单一的 Spark 作业。在这些情况下，我们迄今为止的整个工作只是一个阶段。使用单一阶段可以避免在中间写文件，但可能会增加调试的复杂性。

# 将整个笔记本作为数据准备管道阶段

如果你不想将数据准备笔记本的各个部分转换为管道，你可以将整个笔记本作为一个阶段。你可以使用 JupyterHub 使用的相同容器来以编程方式运行笔记本。

要做到这一点，您需要制作一个新的 `Dockerfile`，指定它基于另一个容器使用 `FROM`，然后添加一个 `COPY` 指令将笔记本打包到新容器中。由于人口普查数据示例中有一个现成的笔记本，这就是我们在 示例 5-36 中采取的方法。

##### 示例 5-36\. 将整个笔记本作为数据准备

```
FROM gcr.io/kubeflow-images-public/tensorflow-1.6.0-notebook-cpu

COPY ./ /workdir /
```

如果您需要额外的 Python 依赖项，可以使用`RUN`指令来安装它们。将依赖项放入容器可以加快流水线速度，特别是对于复杂的包。对于我们的邮件列表示例，Dockerfile 将如示例 5-37 所示。

##### 示例 5-37\. 使用 RUN 命令将 Python 依赖项添加到容器中

```
RUN pip3 install --upgrade lxml pandas
```

我们可以像在第四章中的推荐器示例中一样，在流水线中使用`dsl.ContainerOp`处理此容器。现在你有两种方法可以在 Kubeflow 中使用笔记本，接下来我们将介绍笔记本以外的选项。

###### 提示

笔记本是否需要 GPU 资源？在指定`dsl.ContainerOp`时，调用`set_gpu_limit`并指定所需的 GPU 类型，可以满足您的需求。

# 结论

现在您已经准备好数据来训练模型。我们已经看到，在特征和数据准备方面，没有一种适合所有情况的方法；我们的不同示例需要不同的工具支持。我们还看到了如何在同一个问题中可能需要修改方法，例如我们在扩展邮件列表示例的范围以包含更多数据时。特征的数量和质量，以及产生它们的数据，对机器学习项目的成功至关重要。您可以通过使用较小的数据集运行示例并比较模型来测试这一点。

还要记住，数据和特征准备不是一劳永逸的活动，您可能希望在开发模型时重新审视此步骤。您可能会发现有些功能是您希望拥有的，或者您认为性能良好的功能实际上暗示了数据质量问题。在接下来的章节中，当我们训练和服务模型时，请随时重新审视数据和特征准备的重要性。

¹ 如果您对数据准备还不熟悉，可以参考[TFX 文档](https://www.tensorflow.org/tfx)进行详细了解。

² 使用更多数据进行训练的积极影响在 A. Halevy 等人的文章“数据的非合理有效性”中已经清晰地表现出来，《IEEE 智能系统》24 卷 2 期（2009 年 3-4 月）：8-12 页，[*https://oreil.ly/YI820*](https://oreil.ly/YI820)，以及 T. Schnoebelen 的“更多数据胜过更好的算法”，Data Science Central，2016 年 9 月 23 日，[*https://oreil.ly/oLe1R*](https://oreil.ly/oLe1R)。

³ 更多详细定义，请参见[“使用数据准备掌握机器学习的六个步骤”](https://oreil.ly/qyKTT)。

⁴ 这里涵盖了太多工具，但这篇[博文](https://oreil.ly/Iv9xi)包含了很多信息。

⁵ 数据集往往随着时间增长而不是减少，因此从分布式工具开始可以帮助您扩展工作规模。

⁶ 查看这篇关于缺失数据填充的[博客文章](https://oreil.ly/t5xal)。

⁷ 有一些 VB6 代码需要运行吗？查看第九章，探索超越 TensorFlow 的内容，并做出一点点放弃红酒的牺牲。

⁸ 在[此 Apache 页面](https://oreil.ly/bD1vf)上有一个兼容性矩阵，尽管目前 Beam 的 Python 支持需要启动额外的 Docker 容器，使得在 Kubernetes 上的支持更加复杂。

⁹ 虽然 TFX 自动安装了 TFDV，但如果你使用的是旧版本且没有指定`tensorflow-data-validation`，可能会出现`Could not find a version that satisfies the requirement`的错误，因此我们在这里明确说明需要同时安装两者。

¹⁰ 虽然严格来说不是文件格式，由于 TFX 可以接受 Pandas 数据帧，常见的模式是首先使用 Pandas 加载数据。

¹¹ 虽然没有明确的列表，但许多供应商在[此 Spark 页面](https://spark-packages.org)上列出了它们的格式。

¹² 当然，由于大多数格式存在轻微差异，如果默认设置不起作用，它们具有配置选项。
