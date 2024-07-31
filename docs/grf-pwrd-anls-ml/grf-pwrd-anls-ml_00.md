# 前言

# 目标

本书的目标是向您介绍图数据结构、图分析技术和图机器学习工具。当您完成本书后，我们希望您能理解图分析如何应用于解决各种实际问题。我们希望您能够回答以下问题：图在这项任务中是否合适？我应该使用哪些工具和技术？我的数据中有哪些有意义的关系，如何用关系分析的术语来表达任务？

根据我们的经验，我们发现许多人很快掌握了图的一般概念和结构，但要“思考图”，即开发最佳方法将数据建模为图，然后将分析任务表达为图查询，需要更多的努力和经验。每一章都以其目标列表开头。这些目标分为三个一般领域：学习图分析和机器学习的概念；使用图分析解决特定问题；理解如何使用 GSQL 查询语言和 TigerGraph 图平台。

# 受众和先决条件

我们为任何对数据分析感兴趣并希望学习图分析的人设计了这本书。您不必是一位严肃的程序员或数据科学家，但对数据库和编程概念的一些了解肯定会帮助您跟随本书的讲解。当我们深入探讨一些图算法和机器学习技术时，我们会呈现一些涉及集合、求和和极限的数学方程。然而，这些方程式只是对我们用文字和图形解释的补充。

在用例章节中，我们将在 TigerGraph Cloud 平台上运行预写的 GSQL 代码。您只需一台计算机和互联网访问权限。如果您熟悉 SQL 数据库查询语言和任何主流编程语言，则您将能够理解大部分 GSQL 代码。如果不熟悉，您可以简单地按照书中的说明运行预写的用例示例，并跟随书中的评论。

# 方法和路线图

我们的目标是以真实数据分析需求为动机，而不是理论原则。我们总是尝试用尽可能简单的术语解释事物，使用日常概念而不是技术术语。

通过完整的示例引入了 GSQL 语言。在本书的早期部分，我们逐行描述了每行的目的和功能。我们还突出了语言结构、语法和语义，这些内容特别重要。想要全面学习 GSQL 语言的教程，您可以参考本书之外的其他资源。

本书分为三部分：第一部分：连接；第二部分：分析；第三部分：学习。每部分都包含两种类型的章节。第一种是概念章节，后面是关于 TigerGraph Cloud 和 GSQL 的两到三个用例章节。

| 章节 | 格式 | 标题 |
| --- | --- | --- |
| 1 | 介绍 | 一切从连接开始 |
| 第一部分：连接 |
| 2 | 概念 | 连接和探索数据 |
| 3 | 用例，TigerGraph 介绍 | 看到您的客户和业务更清晰：360 图表 |
| 4 | 用例 | 研究初创投资 |
| 5 | 用例 | 检测欺诈和洗钱模式 |
| 第二部分：分析 |
| 6 | 概念 | 分析连接以获得更深入的洞察 |
| 7 | 用例 | 改善推荐和建议 |
| 8 | 用例 | 加强网络安全 |
| 9 | 用例 | 分析航空公司航班路线 |
| 第三部分：学习 |
| 10 | 概念 | 图驱动的机器学习方法 |
| 11 | 用例 | 实体解析再访 |
| 12 | 用例，机器学习工作台介绍 | 提升欺诈检测 |

# 本书中使用的约定

本书使用以下排版约定：

*Italic*

表示新术语、网址、电子邮件地址、文件名和文件扩展名。

`Constant width`

用于程序列表，以及在段落内引用程序元素，例如变量或函数名称、数据库、数据类型、环境变量、语句和关键字。

**`Constant width bold`**

表示顶点或边类型。

###### 提示

此元素表示提示或建议。

###### 注意

此元素表示一般注释。

###### 警告

此元素表示警告或注意事项。

# 使用代码示例

本书在 GitHub 上有自己的存储库，网址为[*https://github.com/TigerGraph-DevLabs/Book-graph-powered-analytics*](https://github.com/TigerGraph-DevLabs/Book-graph-powered-analytics)。

本站点的初始内容将包括所有用例示例的副本。我们还将把该书的 GSQL 技巧汇集成一篇文档，作为入门指南。随着读者的反馈（我们希望听到您的意见！），我们将发布对常见问题的答复。我们还将添加额外或修改后的 GSQL 示例，或指出如何利用 TigerGraph 平台的新功能。

有关 TigerGraph 和 GSQL 语言的更多资源，请访问 TigerGraph 的主要网站（[*https://www.tigergraph.com*](https://www.tigergraph.com)）、其文档网站（[*https://docs.tigergraph.com*](https://docs.tigergraph.com)）或其 YouTube 频道（[*https://www.youtube.com/@TigerGraph*](https://www.youtube.com/@TigerGraph)）。

您可以通过 gpaml.book@gmail.com 联系作者。

# O'Reilly 在线学习

###### 注意

40 多年来，[*O'Reilly Media*](https://oreilly.com)已提供技术和业务培训、知识和见解，帮助公司取得成功。

我们独特的专家和创新者网络通过书籍、文章和我们的在线学习平台分享他们的知识和专长。O’Reilly 的在线学习平台让您随需应变地访问现场培训课程、深度学习路径、交互式编码环境以及来自 O’Reilly 和其他 200 多家出版商的大量文本和视频内容。欲了解更多信息，请访问[*https://oreilly.com*](https://oreilly.com)。

# 如何联系我们

请就此书的评论和问题联系出版商：

+   O’Reilly Media, Inc.

+   1005 Gravenstein Highway North

+   Sebastopol, CA 95472

+   800-889-8969（美国或加拿大）

+   707-829-7019（国际或本地）

+   707-829-0104（传真）

+   *support@oreilly.com*

+   [*https://www.oreilly.com/about/contact.html*](https://www.oreilly.com/about/contact.html)

我们为本书创建了一个网页，列出勘误、示例和任何其他信息。您可以访问此网页：[*https://oreil.ly/gpaml*](https://oreil.ly/gpaml)。

获取有关我们书籍和课程的新闻和信息，请访问[*https://oreilly.com*](https://oreilly.com)。

在 LinkedIn 上找到我们：[*https://linkedin.com/company/oreilly-media*](https://linkedin.com/company/oreilly-media)

关注我们的 Twitter：[*https://twitter.com/oreillymedia*](https://twitter.com/oreillymedia)

在 YouTube 上观看我们：[*https://youtube.com/oreillymedia*](https://youtube.com/oreillymedia)

# 致谢

没有 TigerGraph 的市场副总裁 Gaurav Deshpande 的提议，这本书就不会存在，他建议我们写这本书并且相信我们能够写出来。他撰写了最初的提案和章节大纲；三部分结构也是他的创意。感谢 TigerGraph 的首席执行官兼创始人 Dr. Yu Xu，他支持我们的工作，并赋予我们在这个项目上的灵活性。Dr. Xu 也构想了 GraphStudio 及其 Starter Kits。Mingxi Wu 和 Alin Deutsch 开发了以高效图分析为目标的 GSQL 语言。

除了官方作者外，还有其他几位贡献了本书的内容。Tom Reeve 运用他的专业写作技巧和对图形概念的了解，帮助我们撰写 Chapter 2，当笔者困扰和拖延似乎是我们最大的敌人时。Emily McAuliffe 和 Amanda Morris 设计了本书的早期版本中的几个图表。我们需要一些数据科学家来审查我们关于机器学习的章节。我们求助于 Parker Erickson 和 Bill Shi，他们不仅是图形机器学习方面的专家，还开发了 TigerGraph ML Workbench。

我们要感谢 TigerGraph 的原始 GSQL 查询与解决方案专家 Xinyu Chang，他开发或监督开发了本书中许多使用案例起始工具包和图算法实现。Yiming Pan 也编写或优化了几个图算法和查询。本书的许多示例都基于他们为 TigerGraph 客户开发的设计。这些起始工具包中的模式、查询和输出显示与本书的英文段落一样重要。我们对这些起始工具包进行了几处改进，以适应本书。许多人帮助审查和标准化起始工具包：开发者关系负责人 Jon Herke 以及几位 TigerGraph 实习生：Abudula Aisikaer、Shreya Chaudhary、McKenzie Steenson 和 Kristine Zheng。负责 TigerGraph Cloud 和 GraphStudio 设计与开发的 Renchu Song 和 Duc Le 确保我们修订后的起始工具包已发布至产品中。

非常感谢 O’Reilly 的两位开发编辑。Nicole Taché 指引我们完成了两章的早期发布，并提供了深刻的评论、建议和鼓励。Gary O’Brien 在此基础上带领我们完成了整个项目，经历了风风雨雨。两位都是出色的编辑，与他们合作是一种荣幸。也感谢我们的制作编辑 Jonathon Owen 和副本编辑 Adam Lawrence。

Victor 感谢他的父母 George 和 Sylvia Lee，他们在他学术和非学术追求中的无私支持。他要感谢他的妻子 Susan Haddox，她始终支持他，容忍他深夜写作，陪他看各种《星际迷航》，并成为他如何既聪明又善良和幽默的榜样。

Kien 感谢他的母亲 My Linh Ly，她始终是他职业生涯中的灵感源泉和推动力。他也感谢他的妻子 Sammy Wai-lok Lee，她一直与他同在，为他的生活增添色彩，照顾他们的女儿 Liv Vy Ly Nguyen-Lee，在写作本书期间出生。

Alex 感谢他的父母，Chris 和 Becky Thomas，以及他的姐姐 Ari，在写作过程中作为讨论伙伴给予他们的支持和鼓励。特别感谢他的妻子 Gloria Zhang，她的无限力量、广博智慧和无穷的灵感能力。
