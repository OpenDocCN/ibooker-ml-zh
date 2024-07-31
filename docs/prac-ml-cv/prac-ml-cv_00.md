# 前言

图像上的机器学习正在改变医疗保健、制造业、零售业和许多其他行业。通过训练机器学习（ML）模型识别图像中的对象，许多以前难以解决的问题现在可以解决。本书的目标是提供对支撑这一快速发展领域的 ML 架构的直观解释，并提供实用代码来应用这些 ML 模型解决涉及分类、测量、检测、分割、表征、生成、计数等问题。

图像分类是深度学习的“hello world”。因此，本书还提供了深度学习的实用端到端介绍。它可以作为进入其他深度学习领域（如自然语言处理）的基础。

您将学习如何设计用于计算机视觉任务的 ML 架构，并使用 TensorFlow 和 Keras 中流行的、经过充分测试的预建模型进行模型训练。您还将学习提高准确性和可解释性的技术。最后，本书将教您如何设计、实施和调整端到端的 ML 管道来理解图像任务。

# 本书适合谁？

本书的主要读者是希望在图像上进行机器学习的软件开发人员。它适用于将使用 TensorFlow 和 Keras 解决常见计算机视觉用例的开发人员。

本书讨论的方法附带在[*https://github.com/GoogleCloudPlatform/practical-ml-vision-book*](https://github.com/GoogleCloudPlatform/practical-ml-vision-book)的代码示例。本书大部分涉及开源 TensorFlow 和 Keras，并且无论您在本地、Google Cloud 还是其他云上运行代码，它都能正常工作。

希望使用 PyTorch 的开发人员将会发现文本解释有用，但可能需要在其他地方寻找实用代码片段。我们欢迎提供 PyTorch 等效代码示例的贡献，请向我们的 GitHub 存储库提交拉取请求。

# 如何使用本书

我们建议您按顺序阅读本书。确保阅读、理解并运行书中的附带笔记本在[GitHub 存储库](https://github.com/GoogleCloudPlatform/practical-ml-vision-book)中的章节——您可以在 Google Colab 或 Google Cloud 的 Vertex 笔记本中运行它们。我们建议在阅读每个文本部分后尝试运行代码，以确保您充分理解引入的概念和技术。我们强烈建议在转到下一章之前完成每章的笔记本。

Google Colab 是免费的，可以运行本书中大多数的笔记本；Vertex Notebooks 更为强大，因此可以帮助您更快地运行笔记本。在第三章、4 章、11 章和 12 章中更复杂的模型和更大的数据集将受益于使用 Google Cloud TPUs。因为本书中所有代码都使用开源 API 编写，所以代码*应该*也能在任何其他安装了最新版本 TensorFlow 的 Jupyter 环境中运行，无论是您的笔记本电脑，还是 Amazon Web Services（AWS）Sagemaker，或者 Azure ML。然而，我们尚未在这些环境中进行过测试。如果您发现需要进行任何更改才能使代码在其他环境中工作，请提交一个拉取请求，以帮助其他读者。

本书中的代码在 Apache 开源许可下向您提供。它主要作为教学工具，但也可以作为您生产模型的起点。

# 书籍组织

本书的其余部分组织如下：

+   在第二章中，我们介绍了机器学习、如何读取图像以及如何使用 ML 模型进行训练、评估和预测。我们在第二章中介绍的模型是通用的，因此在图像上的表现不是特别好，但本章介绍的概念对本书的其余部分至关重要。

+   在第三章中，我们介绍了一些在图像上表现良好的机器学习模型。我们从迁移学习和微调开始，然后介绍了各种卷积模型，这些模型随着我们深入章节变得越来越复杂。

+   在第四章中，我们探讨了利用计算机视觉解决目标检测和图像分割问题的方法。任何在第三章介绍的主干架构都可以在第四章中使用。

+   在第 5 到第九章，我们深入探讨了创建生产级计算机视觉机器学习模型的细节。我们逐个阶段地进行标准机器学习流程，包括在第五章中的数据集创建，第六章中的预处理，第七章中的训练，第八章中的监控和评估，以及第九章中的部署。这些章节讨论的方法适用于第三章和第四章中讨论的任何模型架构和用例。

+   在第十章中，我们讨论了三大新兴趋势。我们将第 5 至第九章涵盖的所有步骤连接成端到端的、容器化的机器学习管道，然后尝试了一个无代码图像分类系统，可用于快速原型设计，也可作为更定制模型的基准。最后，我们展示了如何在图像模型预测中加入可解释性。

+   在第十一章和第十二章中，我们演示了计算机视觉的基本构建模块如何用于解决各种问题，包括图像生成、计数、姿态检测等。这些高级用例都有相应的实现。

# 本书使用的约定

本书使用以下排版约定：

*Italic*

指示新术语、URL、电子邮件地址、文件名和文件扩展名。

`Constant width`

用于程序清单，以及段落内用来指代程序元素如变量名、函数名、数据类型、环境变量、语句和关键字的内容。

`**Constant width bold**`

用于强调代码片段中的内容，以及显示用户应直接输入的命令或其他文本。

`*Constant width italic*`

显示应由用户提供的值或由上下文确定的值。

###### 提示

这一元素表示提示或建议。

###### 注意

这一元素表示一般注释。

###### 警告

这一元素表示警告。

# 使用代码示例

补充材料（代码示例、练习等）可在[*https://github.com/GoogleCloudPlatform/practical-ml-vision-book*](https://github.com/GoogleCloudPlatform/practical-ml-vision-book)下载。

如果你有技术问题或者在使用代码示例时遇到问题，请发送邮件至 bookquestions@oreilly.com。

本书旨在帮助您完成工作。一般情况下，如果本书提供了示例代码，您可以在您的程序和文档中使用它。除非您重现了大量代码，否则无需联系我们以获得许可。例如，编写一个使用本书多个代码块的程序不需要许可。出售或分发包含 O'Reilly 书籍示例的 CD-ROM 需要许可。引用本书并引用示例代码来回答问题不需要许可。将本书中大量示例代码整合到产品文档中需要许可。

我们感谢您的支持，但不要求您署名。署名通常包括标题、作者、出版商和 ISBN。例如：“*Practical Machine Learning for Computer Vision*，作者 Valliappa Lakshmanan、Martin Görner 和 Ryan Gillard，版权所有 2021 年 Valliappa Lakshmanan、Martin Görner 和 Ryan Gillard，978-1-098-10236-4。”

如果您认为您使用的代码示例超出了合理使用范围或以上述许可授权之外，请随时通过 permissions@oreilly.com 联系我们。

# O’Reilly 在线学习

超过 40 年来，O’Reilly Media 一直为企业提供技术和业务培训、知识和洞察，帮助它们取得成功。

我们独特的专家和创新者网络通过书籍、文章和我们的在线学习平台分享他们的知识和专业知识。O’Reilly 的在线学习平台为您提供按需访问的实时培训课程、深入学习路径、交互式编码环境以及来自 O’Reilly 和 200 多个其他出版商的大量文本和视频。更多信息，请访问[*http://oreilly.com*](http://oreilly.com)。

# 如何联系我们

请将有关本书的评论和问题发送至出版商：

+   O’Reilly Media, Inc.

+   1005 Gravenstein Highway North

+   Sebastopol, CA 95472

+   800-998-9938（美国或加拿大）

+   707-829-0515（国际或本地）

+   707-829-0104（传真）

我们为本书设有网页，列出勘误、示例和任何其他信息。您可以访问此页面[*https://oreil.ly/practical-ml-4-computer-vision*](https://oreil.ly/practical-ml-4-computer-vision)。

通过 bookquestions@oreilly.com 发送电子邮件以评论或提出关于本书的技术问题。

关于我们的书籍和课程的新闻和信息，请访问[*http://www.oreilly.com*](http://www.oreilly.com)。

在 Facebook 上找到我们：[*http://facebook.com/oreilly*](http://facebook.com/oreilly)

在 Twitter 上关注我们：[*http://twitter.com/oreillymedia*](http://twitter.com/oreillymedia)

在 YouTube 上观看我们：[*http://www.youtube.com/oreillymedia*](http://www.youtube.com/oreillymedia)

# 致谢

我们非常感谢我们的超级审阅员 Salem Haykal 和 Filipe Gracio，他们审阅了本书的每一章节——他们对细节的把握无处不在。同时也感谢 O’Reilly 的技术审阅员 Vishwesh Ravi Shrimali 和 Sanyam Singhal 提出的重新排序建议，改进了书籍的组织。此外，我们还要感谢 Rajesh Thallam、Mike Bernico、Elvin Zhu、Yuefeng Zhou、Sara Robinson、Jiri Simsa、Sandeep Gupta 和 Michael Munn，他们审阅了与他们专业领域相关的章节。当然，任何剩余的错误都属于我们自己。

我们要感谢 Google Cloud 用户、我们的团队成员以及 Google Cloud 高级解决方案实验室的许多同事，他们推动我们使解释更加简洁。同时也感谢 TensorFlow、Keras 和 Google Cloud AI 工程团队成为深思熟虑的合作伙伴。

我们的 O'Reilly 团队提供了重要的反馈和建议。Rebecca Novack 建议更新早期的 O'Reilly 关于这一主题的书籍，并且接受了我们关于实际计算机视觉书籍现在涉及机器学习的建议，因此这本书需要完全重写。我们的编辑 Amelia Blevins 在 O'Reilly 保持了我们的进展。我们的副本编辑 Rachel Head 和制作编辑 Katherine Tozer 大大提高了我们写作的清晰度。

最后但也是最重要的是，还要感谢我们各自的家人给予的支持。

Valliappa Lakshmanan，华盛顿州贝尔维尤

Martin Görner，华盛顿州贝尔维尤

Ryan Gillard，加利福尼亚州普莱森顿
