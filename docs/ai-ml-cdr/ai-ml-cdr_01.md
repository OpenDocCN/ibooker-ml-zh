# 前言

欢迎来到《编程人员的 AI 与机器学习》，这是一本我多年来一直想写的书，但直到最近机器学习（ML）特别是 TensorFlow 的最新进展才真正变得可能。本书的目标是为你准备好，作为一个程序员，应对许多可以用机器学习解决的场景，目的是让你成为一个 AI 和 ML 开发者，*而无需*博士学位！我希望你会发现它有用，并且它会赋予你开始这段美妙且有回报的旅程的信心。

# 该书适合谁阅读

如果你对人工智能（AI）和机器学习（ML）感兴趣，并且希望快速上手构建能从数据中学习的模型，那么这本书适合你。如果你想要开始学习常见的 AI 和 ML 概念——计算机视觉、自然语言处理、序列建模等等，并且想要了解神经网络如何被训练来解决这些领域的问题，我认为你会喜欢这本书。如果你已经训练好了模型，并且想要将它们交付给移动端用户、浏览器端用户，或者通过云端服务，那么这本书同样适合你！

最重要的是，如果你因为认为人工智能领域过于困难而被拒绝进入这个有价值的计算机科学领域，特别是认为你需要拾起你的旧微积分书籍，那么不用担心：本书采用的是先代码后理论的方法，向你展示了使用 Python 和 TensorFlow 轻松入门机器学习和人工智能世界的方式。

# 我为什么写这本书

我第一次真正接触人工智能是在 1992 年春季。当时我是一名新晋的物理学毕业生，居住在伦敦，正值一场严重的经济衰退中，我已经失业了六个月。英国政府启动了一个培训计划，培训 20 人掌握 AI 技术，并发出了招聘通知。我是第一个被选中的参与者。三个月后，这个计划失败了，因为虽然可以在 AI 领域做很多*理论*工作，但实际上却没有简便的方法来*实际*操作。人们可以用一种叫做 Prolog 的语言编写简单的推理，用一种叫做 Lisp 的语言进行列表处理，但在工业界应用它们的明确路径却并不清晰。著名的“AI 寒冬”随之而来。

后来，2016 年，当我在 Google 工作，参与一个名为 Firebase 的产品时，公司为所有工程师提供了机器学习培训。我和其他几个人坐在一间屋子里，听讲授微积分和梯度下降的课程。我无法将这些直接应用到机器学习的实际实现中，突然间，我回到了 1992 年。我给 TensorFlow 团队提供了关于这一点的反馈，以及我们应该如何教育人们机器学习的建议，他们于 2017 年雇用了我。随着 2018 年 TensorFlow 2.0 的发布，尤其是强调易于开发者入门的高级 API，我意识到需要一本书来利用这一点，并扩大 ML 的普及，使其不再仅限于数学家或博士。

我相信更多的人使用这项技术，并将其部署到最终用户，将会导致 AI 和 ML 的爆发，避免另一次 AI 寒冬，并使世界变得更加美好。我已经看到了这一点的影响，从 Google 在糖尿病视网膜病变上的工作，到宾夕法尼亚州立大学和 PlantVillage 为移动设备建立 ML 模型来帮助农民诊断木薯病，再到无国界医生使用 TensorFlow 模型来帮助诊断抗生素耐药性，等等！

# 浏览本书

本书分为两个主要部分。第一部分（章节 1–11）讨论如何使用 TensorFlow 构建各种场景的机器学习模型。它从基础原理开始——使用仅含一个神经元的神经网络建模——通过计算机视觉、自然语言处理和序列建模。第二部分（章节 12–20）则指导您将模型部署到 Android 和 iOS 设备、浏览器 JavaScript 以及通过云端服务。大多数章节都是独立的，因此您可以随时学习新知识，当然也可以从头到尾阅读整本书。

# 您需要了解的技术

本书上半部分的目标是帮助您学习如何使用 TensorFlow 构建各种架构的模型。这方面唯一的真正先决条件是理解 Python，特别是用于数据和数组处理的 Python 符号。您可能还希望探索 NumPy，这是一个用于数值计算的 Python 库。如果您对这些完全不熟悉，也很容易学会，您可以在学习过程中掌握所需的内容（尽管数组符号可能有点难以理解）。

本书的后半部分，我一般不会教授所展示的语言，而是展示如何在其中使用 TensorFlow 模型。例如，在 Android 章节（第十三章）中，您将探索使用 Kotlin 和 Android Studio 构建应用程序；在 iOS 章节（第十四章）中，您将探索使用 Swift 和 Xcode 构建应用程序。我不会教授这些语言的语法，所以如果您不熟悉它们，可能需要一本入门书籍——[*Learning Swift*](https://oreil.ly/MnEVD)，作者 Jonathan Manning，Paris Buttfield-Addison 和 Tim Nugent（O’Reilly）是一个很好的选择。

# 在线资源

本书使用和支持各种在线资源。至少我建议您关注[TensorFlow](https://www.tensorflow.org)及其相关的[YouTube 频道](https://www.youtube.com/tensorflow)，以获取本书讨论的技术更新和重大变更。

本书的代码可以在[*https://github.com/lmoroney/tfbook*](https://github.com/lmoroney/tfbook)找到，并且我会随着平台的演变而保持更新。

# 本书中使用的约定

本书使用以下印刷约定：

*斜体*

表示新术语、网址、电子邮件地址、文件名和文件扩展名。

`常宽`

用于程序清单，以及段落内指代程序元素，如变量或函数名，数据类型，环境变量，语句和关键字。

`**常宽粗体**`

用于代码片段的强调。

###### 注意

此元素表示一则注释。

# 使用代码示例

本书旨在帮助您完成工作。一般情况下，如果本书提供了示例代码，您可以在您的程序和文档中使用它。除非您复制了大量代码，否则无需联系我们进行授权。例如，编写一个使用本书中多个代码块的程序并不需要授权。销售或分发 O’Reilly 书籍中的示例代码需要授权。引用本书并引用示例代码来回答问题不需要授权。将本书中大量示例代码整合到产品文档中需要授权。

我们赞赏但不要求署名。一般的署名包括标题，作者，出版商和 ISBN。例如：“*AI and Machine Learning for Coders*，作者 Laurence Moroney。版权所有 2021 年 Laurence Moroney，ISBN：978-1-492-07819-7。”

如果您觉得您使用的代码示例超出了合理使用范围或以上授权，请随时联系我们，邮箱地址为 permissions@oreilly.com。

# O’Reilly 在线学习

###### 注意

40 多年来，[*O’Reilly Media*](http://oreilly.com) 提供技术和商业培训，帮助公司获得成功。

我们独特的专家和创新者网络通过书籍、文章和我们的在线学习平台分享他们的知识和专长。O'Reilly 的在线学习平台为您提供按需访问的实时培训课程、深入学习路径、交互式编码环境，以及来自 O'Reilly 和 200 多家其他出版商的大量文本和视频。有关更多信息，请访问[*http://oreilly.com*](http://oreilly.com)。

# 如何联系我们

请将有关本书的评论和问题发送给出版商：

+   O'Reilly Media, Inc.

+   1005 Gravenstein Highway North

+   CA 95472 Sebastopol

+   800-998-9938（美国或加拿大）

+   707-829-0515（国际或本地）

+   707-829-0104（传真）

我们为本书创建了一个网页，其中列出了勘误、示例和任何额外信息。你可以访问这个页面：[*https://oreil.ly/ai-ml*](https://oreil.ly/ai-ml)。

通过邮件联系*bookquestions@oreilly.com*，提出关于本书的评论或技术问题。

有关我们书籍和课程的新闻和信息，请访问[`oreilly.com`](http://oreilly.com)。

在 Facebook 上找到我们：[*http://facebook.com/oreilly*](http://facebook.com/oreilly)

在 Twitter 上关注我们：[*http://twitter.com/oreillymedia*](http://twitter.com/oreillymedia)

在 YouTube 上关注我们：[*http://www.youtube.com/oreillymedia*](http://www.youtube.com/oreillymedia)

# 致谢

我想要感谢许多人在创作本书过程中的帮助。

Jeff Dean 给了我加入 TensorFlow 团队的机会，开启了我 AI 之旅的第二阶段。还有整个团队，虽然无法一一列举，但我要感谢 Sarah Sirajuddin、Megan Kacholia、Martin Wicke 和 Francois Chollet 的出色领导和工程贡献！

TensorFlow 开发者关系团队，由 Kemal El Moujahid、Magnus Hyttsten 和 Wolff Dobson 领导，他们为人们学习 TensorFlow 提供了平台。

Andrew Ng，他不仅为本书撰写了前言，还相信我的 TensorFlow 教学方法，并与我共同在 Coursera 创建了三个专业化课程，教授了数十万人如何在机器学习和人工智能领域取得成功。Andrew 还领导了一个团队，来自[deeplearning.ai](https://www.deeplearning.ai)，包括 Ortal Arel、Eddy Shu 和 Ryan Keenan，他们在帮助我成为更好的机器学习者方面表现出色。

使本书成为可能的 O'Reilly 团队：Rebecca Novack 和 Angela Rufino，没有她们的辛勤工作，我永远无法完成这本书！

了不起的技术审阅团队：Jialin Huang、Laura Uzcátegui、Lucy Wong、Margaret Maynard-Reid、Su Fu、Darren Richardson、Dominic Monn 和 Pin-Yu。

当然，最重要的是（比杰夫和安德鲁更重要 ;) ）我的家人，他们让最重要的事情变得有意义：我的妻子丽贝卡·莫罗尼，我的女儿克劳迪娅·莫罗尼，和我的儿子克里斯托弗·莫罗尼。感谢你们让生活变得比我想象的更加美好。