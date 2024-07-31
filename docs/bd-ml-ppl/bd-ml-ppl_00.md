# 序言

每个人都在谈论机器学习。它已经从学术学科转变为最令人兴奋的技术之一。从理解自动驾驶汽车中的视频反馈到个性化药物，它在每个行业中都变得非常重要。虽然模型架构和概念受到了广泛关注，但机器学习还没有像软件行业在过去二十年中经历的过程标准化那样。在本书中，我们想向您展示如何构建一个标准化的、自动化的机器学习系统，产生可重复的模型。

什么是机器学习流水线？

在过去几年中，机器学习领域的发展令人惊讶。随着图形处理单元（GPUs）的广泛普及以及诸如[BERT](https://arxiv.org/abs/1810.04805)这样的新深度学习概念的兴起，或者像深度卷积 GAN 这样的生成对抗网络（GANs），AI 项目的数量激增。AI 初创公司的数量庞大。组织越来越多地将最新的机器学习概念应用于各种业务问题中。在追求最高效的机器学习解决方案的过程中，我们注意到一些未被重视的事情。我们发现数据科学家和机器学习工程师缺乏用于加速、重用、管理和部署其开发的概念和工具的良好信息来源。需要的是机器学习流水线的标准化。

机器学习流水线实现和规范化了加速、重用、管理和部署机器学习模型的过程。大约十年前，软件工程经历了类似的变革，引入了持续集成（CI）和持续部署（CD）。从前，测试和部署 Web 应用是一个漫长的过程。如今，通过一些工具和概念，这些过程已经大大简化。以前，Web 应用的部署需要 DevOps 工程师和软件开发人员之间的协作。今天，应用程序可以在几分钟内可靠地测试和部署。数据科学家和机器学习工程师可以从软件工程中学到很多关于工作流的知识。我们的目的是通过本书帮助读者理解整个机器学习流水线的标准化过程。

根据我们的个人经验，大多数旨在将模型部署到生产环境的数据科学项目并没有一个庞大的团队。这使得在内部从头开始构建整个流水线变得困难。这可能意味着机器学习项目会变成一次性的努力，在时间过去后性能会下降，数据科学家会花费大量时间在基础数据发生变化时修复错误，或者模型未被广泛使用。一个自动化、可重复的流水线可以减少部署模型所需的工作量。该流水线应包括以下步骤：

+   > > > > 有效地对数据进行版本控制，并启动新的模型训练运行
+   > > > > 
+   > > > > 验证接收到的数据并检查数据漂移情况
+   > > > > 
+   > > > > 有效地预处理数据用于模型训练和验证
+   > > > > 
+   > > > > 有效地训练你的机器学习模型
+   > > > > 
+   > > > > 追踪你的模型训练
+   > > > > 
+   > > > > 分析和验证你训练和调优的模型
+   > > > > 
+   > > > > 部署经过验证的模型
+   > > > > 
+   > > > > 扩展部署的模型
+   > > > > 
+   > > > > 使用反馈循环捕获新的训练数据和模型性能指标

这个列表遗漏了一个重要的点：选择模型架构。我们假设你已经对这一步骤有了良好的工作知识。如果你刚开始接触机器学习或深度学习，以下资源是熟悉机器学习的绝佳起点：

+   > > > > 《深度学习基础：设计下一代机器智能算法》第一版，作者 Nikhil Buduma 和 Nicholas Locascio（O'Reilly）
+   > > > > 
+   > > > > 《Scikit-Learn、Keras 和 TensorFlow 实战》第二版，作者 Aurélien Géron（O'Reilly）

这本书适合谁？

本书的主要受众是数据科学家和机器学习工程师，他们希望不仅仅是训练一次性的机器学习模型，而是成功地将其数据科学项目产品化。你应该对基本的机器学习概念感到舒适，并且熟悉至少一种机器学习框架（例如 PyTorch、TensorFlow、Keras）。本书中的机器学习示例基于 TensorFlow 和 Keras，但核心概念可以应用于任何框架。

本书的次要受众是数据科学项目的经理、软件开发人员或 DevOps 工程师，他们希望帮助组织加速其数据科学项目。如果您有兴趣更好地理解自动化机器学习生命周期及其如何使您的组织受益，本书将介绍一个工具链来实现这一目标。

为什么选择 TensorFlow 和 TensorFlow Extended？

在本书中，我们所有的流水线示例将使用 TensorFlow 生态系统中的工具，特别是 TensorFlow Extended（TFX）。我们选择这一框架背后有多个原因：

+   > > > > TensorFlow 生态系统在撰写本文时是最广泛可用的机器学习生态系统。除了其核心焦点外，它还包括多个有用的项目和支持库，例如 TensorFlow Privacy 和 TensorFlow Probability。
+   > > > > 
+   > > > > 它在小型和大型生产设置中都很受欢迎和广泛使用，并且有一个积极的感兴趣用户社区。
+   > > > > 
+   > > > > 支持的用例涵盖从学术研究到生产中的机器学习。TFX 与核心 TensorFlow 平台紧密集成，用于生产用例。
+   > > > > 
+   > > > > TensorFlow 和 TFX 都是开源工具，使用没有限制。

然而，我们在本书中描述的所有原则也适用于其他工具和框架。

章节概览

每章中，我们将介绍构建机器学习流水线的具体步骤，并演示这些步骤如何与示例项目配合使用。

第一章：介绍了机器学习流水线的概述，讨论了何时应该使用它们，并描述了构成流水线的所有步骤。我们还介绍了本书中将用作示例项目的实例项目。

第二章：介绍了 TensorFlow Extended，介绍了 TFX 生态系统，解释了任务之间如何通信，描述了 TFX 组件在内部工作的方式。我们还深入了解了 ML MetadataStore 在 TFX 上下文中的使用方式，以及 Apache Beam 如何在幕后运行 TFX 组件。

第三章：数据摄取讨论了如何以一致的方式将数据引入我们的流水线，并涵盖了数据版本控制的概念。

第四章：数据验证解释了如何使用 TensorFlow 数据验证有效地验证流入流水线的数据。这将在新数据与先前数据在可能影响模型性能的方式上发生显著变化时提醒您。

第五章：数据预处理侧重于使用 TensorFlow Transform 对数据进行预处理（特征工程），将原始数据转换为适合训练机器学习模型的特征。

第六章：模型训练讨论了如何在机器学习流水线中训练模型。我们还解释了模型调优的概念。

第七章：模型分析与验证介绍了在生产中理解模型的有用指标，包括可能帮助您发现模型预测中的偏差的指标，并讨论解释模型预测的方法。“TFX 中的分析与验证” 解释了当新版本改进指标时如何控制模型的版本。流水线中的模型可以自动更新到新版本。

第八章：使用 TensorFlow Serving 进行模型部署专注于如何高效地部署您的机器学习模型。我们从一个简单的 Flask 实现开始，突出了这种自定义模型应用的局限性。我们将介绍 TensorFlow Serving 以及如何配置您的服务实例。我们还讨论了其批处理功能，并指导您设置客户端以请求模型预测。

第九章：使用 TensorFlow Serving 进行高级模型部署讨论了如何优化您的模型部署以及如何监控它们。我们涵盖了优化 TensorFlow 模型以提高性能的策略。我们还指导您通过 Kubernetes 进行基本的部署设置。

第十章：高级 TensorFlow Extended 引入了为您的机器学习管道定制组件的概念，使您不受 TFX 标准组件的限制。无论您是想添加额外的数据摄取步骤，还是将导出的模型转换为 TensorFlow Lite（TFLite），我们都将指导您完成创建这些组件所需的步骤。

第十一章：管道第一部分：Apache Beam 和 Apache Airflow 将前几章的所有内容联系起来。我们讨论如何将您的组件转换为管道，以及如何配置它们以适配您选择的编排平台。我们还将指导您如何在 Apache Beam 和 Apache Airflow 上运行整个端到端管道。

第十二章：管道第二部分：Kubeflow Pipelines 从上一章继续，介绍了如何使用 Kubeflow Pipelines 和 Google 的 AI 平台进行端到端管道。

第十三章：反馈回路讨论了如何将您的模型管道转变为可以通过最终产品用户的反馈来改进的循环。我们将讨论捕获哪些类型的数据以改进未来版本的模型，以及如何将数据反馈到管道中。

第十四章：机器学习的数据隐私介绍了快速增长的隐私保护机器学习领域，并讨论了三种重要方法：差分隐私、联邦学习和加密机器学习。

第十五章：管道的未来和下一步展望了将影响未来机器学习管道的技术，并讨论了我们将如何思考未来几年的机器学习工程问题。

附录 A：机器学习基础设施简介简要介绍了 Docker 和 Kubernetes。

附录 B：在 Google Cloud 上设置 Kubernetes 集群提供了有关在 Google Cloud 上设置 Kubernetes 的补充材料。

附录 C：操作 Kubeflow Pipelines 的技巧提供了一些有关操作 Kubeflow Pipelines 设置的实用提示，包括 TFX 命令行界面概述。

本书使用的惯例

本书使用以下排版惯例：

Italic

> > 指示新术语、URL、电子邮件地址、文件名和文件扩展名。

`Constant width`

> > 用于程序清单，以及在段落中引用程序元素，如变量或函数名称、数据库、数据类型、环境变量、语句和关键字。

`Constant width bold`

> > 显示用户应该直接输入的命令或其他文本。

`Constant width italic`

> > 显示应由用户提供值或由上下文确定值的文本。
> > 
> 提示
> 
> 此元素表示提示或建议。
> 
> 注意
> 
> 此元素表示一般说明。
> 
> 警告
> 
> 此元素表示警告或注意事项。

使用代码示例

可以从[`oreil.ly/bmlp-git`](https://oreil.ly/bmlp-git)下载补充材料（例如代码示例）。

如果您在使用代码示例时有技术问题或问题，请发送电子邮件至 bookquestions@oreilly.com 或 buildingmlpipelines@gmail.com。

本书旨在帮助您完成工作。通常情况下，如果本书提供了示例代码，您可以在您的程序和文档中使用它。除非您复制了代码的大部分，否则无需联系我们请求许可。例如，编写一个使用本书多个代码片段的程序不需要许可。销售或分发 O’Reilly 图书中的示例代码需要许可。通过引用本书回答问题并引用示例代码不需要许可。将本书大量示例代码整合到您产品的文档中需要许可。

我们感谢但不要求署名。署名通常包括标题、作者、出版社和 ISBN。例如：“《Building Machine Learning Pipelines》由 Hannes Hapke 和 Catherine Nelson（O’Reilly）著作。2020 年版权所有 Hannes Hapke 和 Catherine Nelson，978-1-492-05319-4。”

如果您认为使用代码示例超出了合理使用范围或上述许可，请随时通过 permissions@oreilly.com 与我们联系。

[O’Reilly Online Learning](http://oreilly.com)

> 注意
> 
> 超过 40 年来，[O’Reilly Media](http://oreilly.com)提供技术和商业培训、知识和见解，帮助公司取得成功。

我们独特的专家和创新者网络通过书籍、文章和我们的在线学习平台分享他们的知识和专业知识。O’Reilly 的在线学习平台为您提供按需访问的现场培训课程、深入学习路径、交互式编码环境以及来自 O’Reilly 和其他 200 多家出版商的大量文本和视频。更多信息，请访问[`oreilly.com`](http://oreilly.com)。

如何联系我们

两位作者想感谢您选择阅读本书并给予关注。如果您希望与他们联系，可以通过他们的网站 www.buildingmlpipelines.com 或通过电子邮件 buildingmlpipelines@gmail.com 与他们联系。祝您在构建自己的机器学习流水线过程中取得成功！

请将有关本书的评论和问题寄给出版商：

+   O’Reilly Media, Inc.

+   1005 Gravenstein Highway North

+   加利福尼亚州 Sebastopol，95472

+   800-998-9938（美国或加拿大）

+   707-829-0515（国际或本地）

+   707-829-0104（传真）

我们为这本书创建了一个网页，上面列出了勘误、示例和任何额外信息。您可以访问此页面：[`oreil.ly/build-ml-pipelines`](https://oreil.ly/build-ml-pipelines)。

电子邮件 bookquestions@oreilly.com 以对本书提出评论或技术问题。

要获取关于我们的图书和课程的新闻和信息，请访问[`oreilly.com`](http://oreilly.com)。

在 Facebook 上找到我们：[`facebook.com/oreilly`](http://facebook.com/oreilly)

关注我们的 Twitter：[`twitter.com/oreillymedia`](http://twitter.com/oreillymedia)

在 YouTube 上观看我们：[`www.youtube.com/oreillymedia`](http://www.youtube.com/oreillymedia)

致谢

在撰写本书的整个过程中，我们得到了许多人的大力支持。非常感谢所有帮助使其成为现实的人！特别感谢以下人员。

O’Reilly 的每个人在整本书的整个生命周期中都非常出色。感谢我们的编辑 Melissa Potter，Nicole Taché和 Amelia Blevins，感谢他们的出色支持、持续鼓励和深思熟虑的反馈。同时也感谢 Katie Tozer 和 Jonathan Hassell 在路上的支持。

感谢 Aurélien Géron、Robert Crowe、Margaret Maynard-Reid、Sergii Khomenko 和 Vikram Tiwari，他们审阅了整本书并提供了许多有益的建议和深刻的评论。您的审阅使最终稿变成了一本更好的书。感谢您花费的时间对书籍进行如此详细的审阅。

感谢 Yann Dupis、Jason Mancuso 和 Morten Dahl 对机器学习隐私章节的彻底审查和深入分析。

我们在 Google 有很多出色的支持者。感谢你们帮助我们找到和修复 bug，以及使这些工具作为开源包发布！除了提到的 Google 员工外，特别感谢 Amy Unruh、Anusha Ramesh、Christina Greer、Clemens Mewald、David Zats、Edd Wilder-James、Irene Giannoumis、Jarek Wilkiewicz、Jiayi Zhao、Jiri Simsa、Konstantinos Katsiapis、Lak Lakshmanan、Mike Dreves、Paige Bailey、Pedram Pejman、Sara Robinson、Soonson Kwon、Thea Lamkin、Tris Warkentin、Varshaa Naganathan、Zhitao Li 和 Zohar Yahav。

感谢 TensorFlow 和 Google Developer Expert 社区及其出色的成员们。我们对社区深表感激。感谢你们支持这一努力。

感谢其他在不同阶段帮助过我的贡献者们：Barbara Fusinska、Hamel Husain、Michał Jastrzębski 和 Ian Hensel。

感谢 Concur Labs（过去和现在）以及 SAP Concur 其他地方的人们，为书籍提供了有益的讨论和建议。特别感谢 John Dietz 和 Richard Puckett 对这本书的极大支持。

Hannes

> > 我想要感谢我的伟大搭档**惠特尼**，在写作这本书的过程中给予我巨大的支持。感谢你的持续鼓励和反馈，以及忍受我长时间写作的陪伴。感谢我的家人，特别是我的父母，让我能够追随我的梦想走遍世界。
> > 
> > 没有了伟大的朋友，这本书就不可能问世。感谢 Cole Howard 成为我出色的朋友和导师。我们当初的合作开启了这本书的出版，也启发了我对机器学习流水线的思考。对我的朋友 Timo Metzger 和 Amanda Wright，感谢你们教会我语言的力量。同时也感谢 Eva 和 Kilian Rambach，以及 Deb 和 David Hackleman。没有你们的帮助，我不会一路走到俄勒冈。
> > 
> > 我要感谢像 Cambia Health、Caravel 和 Talentpair 这样的前雇主，让我能够将这本书中的概念应用到生产环境中，尽管这些概念是新颖的。
> > 
> > 这本书的问世离不开我的合著者 Catherine。感谢你的友谊、鼓励和无穷的耐心。很高兴我们因生活的偶然性而相遇。我们一起完成这本出版物，我感到非常开心。

Catherine

> > 我在这本书中写了很多字，但没有足够的言语来表达我对丈夫 Mike 的支持之深。感谢你的鼓励、做饭、有益的讨论、讽刺和深刻的反馈。感谢我的父母很久以前就给我种下编程的种子，虽然它花了一段时间才生根发芽，但你们一直都是对的！
> > 
> > 感谢我有幸参与的所有美好社区。通过西雅图 PyLadies、数据科学女性和更广泛的 Python 社区，我结识了很多优秀的人。我非常感谢你们的鼓励。
> > 
> > 特别感谢汉内斯邀请我一同走过这段旅程！没有你，这一切都不可能发生！你的专业知识、注重细节和坚持不懈使整个项目取得了成功。而且，这一切也非常有趣！
