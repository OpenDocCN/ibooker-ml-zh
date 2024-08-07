# 前言

# 使用机器学习驱动应用的目标

在过去的十年中，机器学习（ML）越来越多地被用于驱动各种产品，如自动支持系统、翻译服务、推荐引擎、欺诈检测模型等等。

令人惊讶的是，目前几乎没有资源可以教授工程师和科学家如何构建这样的产品。许多书籍和课程会教如何训练 ML 模型或如何构建软件项目，但很少有结合两者来教如何构建由 ML 驱动的实用应用。

将 ML 部署为应用的一部分需要创造力、强大的工程实践和分析思维的结合。ML 产品建设的挑战性在于，它们需要的远不止简单地在数据集上训练模型。为特定特征选择正确的 ML 方法，分析模型错误和数据质量问题，并验证模型结果以保证产品质量，这些都是 ML 建设过程的核心挑战。

本书将详细介绍此过程的每一个步骤，并旨在通过分享方法、代码示例以及我和其他经验丰富的从业者的建议来帮助您完成每一个步骤。我们将覆盖设计、构建和部署 ML 驱动应用所需的实际技能。本书的目标是帮助您在 ML 过程的每个环节都取得成功。

## 利用 ML 构建实用应用

如果您经常阅读 ML 论文和企业工程博客，可能会被线性代数方程式和工程术语的组合所压倒。该领域的混合性质使许多工程师和科学家感到对 ML 领域感到畏惧，他们本可以贡献他们的多样专业知识。同样，企业家和产品领导者经常很难将他们的业务理念与 ML 今天（以及明天可能）所可能实现的联系起来。

本书涵盖了我在多家公司数据团队工作和帮助数百名数据科学家、软件工程师和产品经理通过我在洞察数据科学领导人工智能项目工作中积累的经验，构建应用 ML 项目的经验。

本书的目标是分享构建 ML 驱动应用的逐步实用指南。它是实用的，并专注于具体的提示和方法，帮助您原型设计、迭代和部署模型。因涵盖多个主题，我们将在每一步中仅提供所需的详细信息。在可能的情况下，我会提供资源，以帮助您深入研究所涵盖的主题。

重要的概念通过实际示例进行了说明，包括一个案例研究，该案例将在本书结束时从构想到部署模型。大多数示例都将附带插图，并且许多示例将包含代码。本书中使用的所有代码均可在[书籍的配套 GitHub 仓库](https://oreil.ly/ml-powered-applications)中找到。

由于本书侧重描述机器学习的过程，每一章都建立在前面定义的概念基础之上。因此，我建议按顺序阅读，以便了解每个后续步骤如何融入整个过程中。如果你希望探索机器学习过程的子集，你可能更适合选择更专业的书籍。如果是这种情况，我会分享几个推荐。

## 附加资源

+   如果你想深入了解机器学习，甚至能够从零开始编写自己的算法，我推荐阅读[*《从零开始的数据科学》*](https://www.oreilly.com/library/view/data-science-from/9781492041122/)，作者是 Joel Grus。如果你对深度学习理论感兴趣，那么《*深度学习*》（MIT Press），作者是 Ian Goodfellow、Yoshua Bengio 和 Aaron Courville，是一本全面的资源。

+   如果你想知道如何在特定数据集上高效而准确地训练模型，[Kaggle](https://www.kaggle.com/) 和 [fast.ai](https://fast.ai) 是很好的资源。

+   如果你想学习如何构建能够处理大量数据的可扩展应用程序，我建议阅读[*《设计数据密集型应用》*](http://shop.oreilly.com/product/0636920032175.do)（O’Reilly），作者是 Martin Kleppmann。

如果你具有编码经验和一些基本的机器学习知识，并且希望构建以机器学习驱动的产品，这本书将引导你完成从产品理念到交付原型的整个过程。如果你已经作为数据科学家或机器学习工程师工作，这本书将为你的机器学习开发工具添加新的技术。如果你不懂如何编码，但与数据科学家合作，这本书可以帮助你理解机器学习的过程，只要你愿意跳过一些深入的代码示例。

让我们从更深入地探讨实用机器学习的含义开始。

# 实用机器学习

对于本次介绍来说，可以将机器学习视为利用数据中的模式自动调整算法的过程。这是一个通用的定义，因此你不会感到意外，许多应用程序、工具和服务开始将机器学习集成到它们的核心功能中。

其中一些任务是面向用户的，例如搜索引擎、社交平台上的推荐、翻译服务，或者自动检测照片中熟悉面孔、遵循语音命令并试图为电子邮件中的句子提供有用建议的系统。

一些以不那么显眼的方式工作，悄悄地过滤垃圾邮件和欺诈账户，提供广告服务，预测未来的使用模式以有效地分配资源，或者尝试个性化网站体验以适应每个用户。

目前许多产品正在利用 ML，而更多产品可能也会这样做。实际的 ML 指的是识别可能受益于 ML 的实际问题，并为这些问题提供成功的解决方案。从高层次的产品目标到 ML 驱动的结果，是一个具有挑战性的任务，本书试图帮助您完成。

一些 ML 课程将通过提供数据集并要求学生在其中训练模型来教授学生 ML 方法，但在数据集上训练算法只是 ML 过程的一小部分。引人注目的 ML 驱动产品依赖于不仅仅是一个累积准确度分数，并且是一个漫长过程的结果。本书将从构思开始，一直到生产，以一个示例应用程序的每一步来说明。我们将分享从与部署这些类型系统的应用团队合作中学到的工具、最佳实践和常见问题。

## 本书涵盖的内容

为了涵盖构建由 ML 驱动的应用程序的主题，本书的重点是具体和实际的。特别是，本书旨在说明构建 ML 驱动应用程序的整个过程。

因此，我将首先描述解决每个步骤的方法。然后，我将使用一个实例项目作为案例研究来说明这些方法。本书还包含许多在工业界应用 ML 的实际例子，并采访了那些构建和维护生产 ML 模型的专业人士。

### 整个 ML 的过程

要成功地为用户提供一个 ML 产品，你需要做的不仅仅是简单地训练一个模型。你需要深思熟虑地*转化*你的产品需求为一个 ML 问题，*收集*充足的数据，在不同模型之间高效*迭代*，*验证*你的结果，并以稳健的方式*部署*它们。

构建模型通常只占据 ML 项目总工作量的十分之一。精通整个 ML 流程对于成功构建项目，成功完成 ML 面试并成为 ML 团队的重要贡献者至关重要。

### 一个技术实际的案例研究

虽然我们不会从头开始用 C 重新实现算法，但我们将通过使用提供更高级抽象的库和工具来保持实际和技术。我们将在本书中一起构建一个示例 ML 应用程序，从最初的想法到部署产品。

当适用时，我将使用代码片段来说明关键概念，并描述我们的应用程序的图像。学习 ML 的最佳方式是通过实践，因此我鼓励您逐步学习本书中的示例，并根据自己的需求调整它们以构建您自己的 ML 驱动应用程序。

### 真实的商业应用

在整本书中，我将包括来自在 StitchFix、Jawbone 和 FigureEight 等科技公司的数据团队工作过的机器学习领袖的对话和建议。这些讨论将涵盖在与数百万用户建立机器学习应用后积累的实用建议，并纠正一些关于如何使数据科学家和数据科学团队成功的流行误解。

## 先决条件

本书假设读者对编程有一定了解。我将主要使用 Python 进行技术示例，并假设读者熟悉其语法。如果你想要恢复你的 Python 知识，我推荐[*Python 之旅*](http://shop.oreilly.com/product/0636920042921.do)（O'Reilly 出版），作者是 Kenneth Reitz 和 Tanya Schlusser。

此外，虽然我会定义书中引用的大多数机器学习概念，但我不会涵盖所有使用的机器学习算法的内部工作原理。大多数这些算法都是标准的机器学习方法，这些方法在入门级机器学习资源中有所涵盖，比如在“额外资源”中提到的资源。

## 我们的案例研究：基于机器学习的写作辅助

为了具体说明这个想法，当我们阅读本书时，我们将一起构建一个机器学习应用程序。

作为一个案例研究，我选择了一个能够准确展示迭代和部署机器学习模型复杂性的应用。我还想要涵盖一个能够产生价值的产品。这就是为什么我们将实施一个*基于机器学习的写作助手*。

我们的目标是构建一个系统，帮助用户写得更好。特别是，我们将旨在帮助人们写出更好的问题。这可能看起来是一个非常模糊的目标，随着我们界定项目范围，我将更清晰地定义它，但这是一个很好的示例，因为它有几个关键原因。

文本数据无处不在

对于大多数你能想到的用例，文本数据都是充足可用的，也是许多实际机器学习应用的核心。无论我们是试图更好地理解产品评论，准确分类传入的支持请求，还是将我们的促销信息定制给潜在受众，我们都将使用和生成文本数据。

写作助手非常有用

从 Gmail 的文本预测功能到 Grammarly 的智能拼写检查器，基于机器学习的编辑器已经证明它们可以以多种方式为用户提供价值。这使得我们特别有兴趣探索如何从头开始构建它们。

基于机器学习的写作辅助是独立的

许多机器学习应用程序只有在与更广泛的生态系统紧密集成时才能正常运行，比如预测骑行应用的 ETA、在线零售商的搜索和推荐系统以及广告竞价模型。然而，一个文本编辑器，即使它可能受益于集成到文档编辑生态系统中，也可以单独提供价值，并通过一个简单的网站进行暴露。

在整本书中，这个项目将允许我们突出显示我们建议用于构建基于机器学习的应用程序的挑战和相关解决方案。

## 机器学习过程

从一个想法到部署的机器学习应用的道路是曲折而漫长的。在看到许多公司和个人构建这样的项目后，我确定了四个关键的连续阶段，每个阶段都将在本书的一个部分中讨论。

1.  *确定正确的机器学习方法：* 机器学习领域广泛，并经常提出多种方法来解决特定的产品目标。对于给定问题的最佳方法将取决于许多因素，例如成功标准、数据可用性和任务复杂性。此阶段的目标是设定正确的成功标准，并确定适当的初始数据集和模型选择。

1.  *构建初始原型：* 在开始模型工作之前，首先建立一个端到端的原型。此原型应旨在解决产品目标，不涉及机器学习，并将允许您确定如何最佳应用机器学习。建立原型后，您应该知道是否需要机器学习，并且应该能够开始收集数据集来训练模型。

1.  *迭代模型：* 现在您已经有了数据集，可以训练模型并评估其缺点。这一阶段的目标是在错误分析和实施之间反复交替。增加此迭代循环发生的速度是提高机器学习开发速度的最佳方法。

1.  *部署和监控：* 一旦模型显示出良好的性能，您应选择合适的部署选项。一旦部署，模型往往会以意想不到的方式失败。本书的最后两章将介绍减少和监控模型错误的方法。

有很多内容需要涵盖，因此让我们直接开始并从第一章开始吧！

# 本书中使用的约定

本书中使用以下排版约定：

*Italic*

指示新术语、网址、电子邮件地址、文件名和文件扩展名。

`Constant width`

用于程序列表，以及在段落中引用程序元素，如变量或函数名称、数据库、数据类型、环境变量、语句和关键字。

**`Constant width bold`**

显示应由用户按字面意义输入的命令或其他文本。

*`Constant width italic`*

显示应由用户提供的值或由上下文确定的值替换的文本。

###### 提示

此元素表示提示或建议。

###### 注

此元素表示一般注释。

###### 警告

此元素表示警告或注意事项。

# 使用代码示例

本书的补充代码示例可在[*https://oreil.ly/ml-powered-applications*](https://oreil.ly/ml-powered-applications)下载。

如果您有技术问题或在使用代码示例时遇到问题，请发送电子邮件至*bookquestions@oreilly.com*。

本书旨在帮助您完成工作任务。一般情况下，如果本书提供了示例代码，您可以在自己的程序和文档中使用它。除非您要复制大部分代码，否则无需征得我们的许可。例如，编写一个使用本书多个代码片段的程序并不需要征得许可。销售或分发 O’Reilly 图书的示例则需要许可。如果您通过引用本书并引用示例代码来回答问题，则无需征得许可。将本书的大量示例代码整合到产品文档中需要征得许可。

我们感谢您的致谢，但通常不需要。致谢通常包括标题、作者、出版商和 ISBN。例如：*Building Machine Learning Powered Applications* by Emmanuel Ameisen (O’Reilly). Copyright 2020 Emmanuel Ameisen, 978-1-492-04511-3.”

如果您觉得使用的代码示例超出了合理使用范围或此处给出的权限，请随时联系我们，邮箱是*permissions@oreilly.com*。

# 致谢

写这本书的项目始于我在 Insight Data Science 指导 Fellows 和监督 ML 项目的工作。感谢 Jake Klamka 和 Jeremy Karnowski 分别给了我领导这个项目的机会，并鼓励我写下所学的教训。我还要感谢我在 Insight 与之合作的数百位 Fellows，让我有机会帮助他们推动 ML 项目的界限。

写一本书是一项艰巨的任务，而 O’Reilly 的工作人员帮助在每一步骤中都更加可控。特别是，我要感谢我的编辑 Melissa Potter，她在写书的旅程中不知疲倦地提供指导、建议和精神支持。感谢 Mike Loukides，他以某种方式说服我写书是一个合理的事业。

感谢技术审阅人员在书的初稿中仔细查找错误并提出改进建议。感谢 Alex Gude，Jon Krohn，Kristen McIntyre 和 Douwe Osinga 抽出宝贵时间帮助这本书变得更好。对于那些我向他们询问实际 ML 挑战的数据实践者们，感谢你们的时间和见解，希望你们会发现这本书充分涵盖了这些挑战。

最后，在这本书的写作过程中，经历了一系列繁忙的周末和深夜，我要感谢我的坚定伴侣 Mari，我那挑剔的搭档 Eliott，我的智慧和耐心的家人，以及那些没有将我报失踪的朋友们。因为有你们，这本书才成为现实。
