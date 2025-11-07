# 前置材料

## 前言

近年来，机器学习取得了巨大的进步，但大规模机器学习仍然具有挑战性。以模型训练为例。随着 TensorFlow、PyTorch 和 XGBoost 等机器学习框架的多样性，在分布式 Kubernetes 集群上自动训练机器学习模型的过程并不容易。不同的模型需要不同的分布式训练策略，例如利用参数服务器和使用网络结构的集体通信策略。在现实世界的机器学习系统中，许多其他重要组件，如数据摄取、模型服务和工作流程编排，必须精心设计，以使系统可扩展、高效且易于迁移。对于几乎没有 DevOps 经验的机器学习研究人员来说，很难启动和管理分布式训练任务。

许多书籍都涉及机器学习或分布式系统，然而，目前还没有一本书能够同时讨论两者的结合，并弥合它们之间的差距。本书将介绍在分布式环境中大规模机器学习系统的许多模式和最佳实践。

本书还包括一个动手项目，构建了一个端到端分布式机器学习系统，该系统集成了本书中涵盖的许多模式。我们将使用包括 Kubernetes、Kubeflow、TensorFlow 和 Argo 在内的多种最先进技术来实现系统。这些技术是构建原生云分布式机器学习系统的首选，使其非常可扩展和易于迁移。

我在这个领域工作多年，包括维护本书中使用的某些开源工具，并领导团队提供可扩展的机器学习基础设施。在我的日常工作中，设计系统之初或改进现有系统时，这些模式和它们的权衡总是被考虑在内。我希望这本书也能对你有所帮助！

## 致谢

首先，我要感谢我的妻子，文璇。你一直支持我，在我努力完成这本书的过程中，你总是耐心地倾听，总是让我相信我能完成这个项目，并在我在写书的时候帮助照顾孩子。感谢我的三个可爱的孩子，他们在我遇到困难时总是给我带来笑容。我爱你们所有人。

接下来，我想感谢我的前开发编辑 Patrick Barb，感谢你多年来对我的耐心和指导。我还要感谢 Michael Stephens 指导本书的方向，并在我怀疑自己时帮助我度过难关。还要感谢 Karen Miller 和 Malena Selic 提供顺利的过渡，并帮助我快速进入生产阶段。你们对这本书质量的承诺使它对每一位读者都变得更好。还要感谢所有在 Manning 与我一起参与本书生产和推广的人。这确实是一个团队的努力。

还要感谢我的技术编辑 Gerald Kuch，他带来了超过 30 年的行业经验，包括几家大型公司和初创企业以及研究实验室。Gerald 在数据结构和算法、函数式编程、并发编程、分布式系统、大数据、数据工程和数据科学方面的知识和教学经验，使他在手稿开发过程中成为我的宝贵资源。 

最后，我还要感谢在本书开发过程中不同阶段抽出时间阅读我的手稿并提供宝贵反馈的审稿人。感谢 Al Krinker、Aldo Salzberg、Alexey Vyskubov、Amaresh Rajasekharan、Bojan Tunguz、Cass Petrus、Christopher Kottmyer、Chunxu Tang、David Yakobovitch、Deepika Fernandez、Helder C. R. Oliveira、Hongliang Liu、James Lamb、Jiri Pik、Joel Holmes、Joseph Wang、Keith Kim、Lawrence Nderu、Levi McClenny、Mary Anne Thygesen、Matt Welke、Matthew Sarmiento、Michael Aydinbas、Michael Kareev、Mikael Dautrey、Mingjie Tang、Oleksandr Lapshyn、Pablo Roccatagliata、Pierluigi Riti、Prithvi Maddi、Richard Vaughan、Simon Verhoeven、Sruti Shivakumar、Sumit Pal、Vidhya Vinay、Vladimir Pasman 和 Wei Yan，你们的建议帮助我改进了这本书。

## 关于这本书

*《分布式机器学习模式》* 中充满了在云端的分布式 Kubernetes 集群上运行机器学习系统的实用模式。每个模式都是为了帮助解决在构建分布式机器学习系统时面临的常见挑战而设计的，包括支持分布式模型训练、处理意外故障和动态模型服务流量。现实场景提供了如何应用每个模式的清晰示例，以及每种方法的潜在权衡。一旦你掌握了这些前沿技术，你将把它们全部付诸实践，并通过构建一个全面的分布式机器学习系统来完成。

### 适合阅读这本书的人

*《分布式机器学习模式》* 适合熟悉机器学习算法基础以及在生产环境中运行机器学习的数据分析师、数据科学家和软件工程师。读者应熟悉 Bash、Python 和 Docker 的基础知识。

### 本书组织结构：路线图

本书分为三个部分，共涵盖九个章节。

第一部分提供了一些关于分布式机器学习系统的背景和概念。我们将讨论机器学习应用的增长规模和分布式系统的复杂性，并介绍在分布式系统和分布式机器学习系统中常见的一些模式。

第二部分展示了机器学习系统各个组件中涉及的一些挑战，并介绍了一些在行业中广泛采用的成熟模式来解决这些挑战：

+   第二章介绍了数据摄取模式，包括批处理、分片和缓存，以有效地处理大数据集。

+   第三章包括了在分布式模型训练中经常看到的三个模式，涉及参数服务器、集体通信、弹性以及容错性。

+   第四章展示了复制的服务、分片的服务和事件驱动处理在模型服务中的有用性。

+   第五章描述了几个工作流程模式，包括扇入和扇出模式、同步和异步模式以及步骤记忆化模式，这些模式通常用于创建复杂和分布式的机器学习工作流程。

+   第六章以调度和元数据模式结束这一部分，这些模式对于操作可能很有用。

第三部分深入到端到端的机器学习系统，以应用我们之前学到的知识。读者将获得实际经验，实现在这个项目中之前学到的许多模式：

+   第七章介绍了项目背景和系统组件。

+   第八章涵盖了我们将用于项目的技术的根本原理。

+   第九章通过一个端到端的机器学习系统的完整实现来结束本书。

通常情况下，如果读者已经知道什么是分布式机器学习系统，可以跳过第一部分。第二部分的所有章节都可以独立阅读，因为每个章节都涵盖了分布式机器学习系统中的不同视角。第七章和第八章是第九章中我们构建的项目的前提条件。如果读者已经熟悉这些技术，可以跳过第八章。

### 关于代码

您可以从本书的 liveBook（在线）版本中获取可执行的代码片段，网址为[`livebook.manning.com/book/distributed-machine-learning-patterns`](https://livebook.manning.com/book/distributed-machine-learning-patterns)。本书中示例的完整代码可以从 Manning 网站[www.manning.com](http://www.manning.com)和 GitHub 仓库[`github.com/terrytangyuan/distributed-ml-patterns`](https://github.com/terrytangyuan/distributed-ml-patterns)下载。请将任何问题提交到 GitHub 仓库，它将得到积极监控和维护。

### liveBook 讨论论坛

购买《分布式机器学习模式》包括免费访问 Manning 的在线阅读平台 liveBook。使用 liveBook 的独特讨论功能，您可以在全球范围内或针对特定章节或段落附加评论。为自己做笔记、提出和回答技术问题，以及从作者和其他用户那里获得帮助都非常简单。要访问论坛，请访问[`livebook.manning.com/book/distributed-machine-learning-patterns/discussion`](https://livebook.manning.com/book/distributed-machine-learning-patterns/discussion)。您还可以在[`livebook.manning.com/discussion`](https://livebook.manning.com/discussion)了解更多关于 Manning 的论坛和行为准则。

曼宁对读者的承诺是提供一个平台，在这里读者之间以及读者与作者之间可以进行有意义的对话。这并不是对作者参与特定数量活动的承诺，作者对论坛的贡献仍然是自愿的（且未支付报酬）。我们建议您尝试向作者提出一些挑战性的问题，以免他们的兴趣转移！只要这本书有售，论坛和之前讨论的存档将可通过出版社的网站访问。

## 关于作者

![Tang_Author-Photo](img/Tang_Author-Photo.png)

殷唐是 Akuity 的创始人工程师，为开发者构建一个企业级平台。他之前在阿里巴巴和 Uptake 领导数据科学和工程团队，专注于 AI 基础设施和 AutoML 平台。他是 Argo 和 Kubeflow 的项目负责人，TensorFlow 和 XGBoost 的维护者，以及多个开源项目的作者。此外，殷唐还著有三本机器学习书籍和几篇出版物。他是各种会议的常客，并在多个组织担任技术顾问、领导者和导师。

## 关于封面插图

《分布式机器学习模式》封面上的形象是“科孚人”，或“科孚岛人”，取自雅克·格拉塞·德·圣索沃尔的收藏，该收藏于 1797 年出版。每一幅插图都是手工精细绘制和着色的。

在那些日子里，仅凭人们的着装就可以轻易地识别出他们的居住地以及他们的职业或社会地位。曼宁通过基于几个世纪前丰富多样的地区文化的书封面来庆祝计算机行业的创新精神和主动性，这些文化通过像这样的收藏品中的图片被重新带回生活。
