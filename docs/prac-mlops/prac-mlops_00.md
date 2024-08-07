# 前言

# 我们为什么写这本书

我们两个大部分职业生涯都在自动化事务。当我们初次见面时，阿尔弗雷多不懂 Python，诺亚建议每周自动化一个任务。自动化是 MLOps、DevOps 和本书的核心支柱。您应该将本书中的所有示例和观点放在未来自动化的背景下思考。

如果诺亚能总结他在 2000 年至 2020 年间的经历，他几乎自动化了所有能够自动化的事情，从电影制作流水线到软件安装再到机器学习流水线。作为湾区初创公司的工程经理和首席技术官，他从零开始建立了许多数据科学团队。因此，在人工智能/机器学习革命的早期阶段，他看到了许多将机器学习应用到生产环境中的核心问题。

在过去几年中，诺亚一直在杜克大学、西北大学和加州大学戴维斯分校担任兼职教授，主要教授云计算、数据科学和机器学习工程相关课题。这种教学和工作经验使他对机器学习解决方案在现实世界部署中涉及的问题有了独特的视角。

阿尔弗雷多在系统管理员时代就有丰富的运维背景，对自动化有着相似的热情。如果没有一键自动化，是无法建立弹性基础设施的。在灾难发生时，重新运行脚本或流水线以重建崩溃的内容是最令人满足的事情。

当 COVID-19 爆发时，加速了我们共同关注的一个问题：“为什么我们不将更多模型投入到生产中？”诺亚在一篇他为 Forbes 写的 [文章](https://oreil.ly/Qj8ut) 中触及了其中一些问题。该文章的总结前提是，数据科学出现了问题，因为组织没有看到他们投资的回报。

后来在 [O’Reilly 的 “Foo Camp”](https://oreil.ly/ODJvG)，诺亚主持了一个关于“为什么我们不能在生产环境中提高机器学习速度 10 倍？”的讨论会，与包括 Tim O’Reilly、Mike Loukides、Roger Magoulas 在内的众多人进行了深入讨论。讨论的结果是：“是的，我们可以达到 10 倍速度。”因此感谢 Tim 和 Mike 引发了这样一场引人入胜的讨论，并让本书顺利进行。

机器学习在过去几十年中感觉很像其他许多技术。起初，要取得成果需要花费多年时间。史蒂夫·乔布斯谈到，NeXT 希望能够将软件构建速度提高 10 倍（他成功了）。您可以在 [YouTube](https://oreil.ly/mWRoO) 观看该采访。目前机器学习的一些问题是什么？

+   焦点放在“代码”和技术细节上，而非业务问题

+   缺乏自动化

+   HiPPO（最高薪水人士的意见）

+   非云原生

+   缺乏解决可解决问题的紧迫性

引用讨论中 Noah 提出的一点：“我反对一切精英主义。编程是人类的权利。认为只有某些特权阶层才有资格做这件事是错误的。” 与机器学习类似，技术不应只掌握在少数人手中。通过 MLOps 和 AutoML，这些技术可以走进公众的生活。我们可以通过使机器学习和人工智能技术民主化来做得更好。真正的 AI/ML 从业者将模型推向生产环境，在“真实”的未来，如医生、律师、技师和教师等人也将利用 AI/ML 来帮助他们完成工作。

# 本书组织结构

我们设计本书的方式是让你可以把每一章当作独立的部分来使用，旨在为你提供即时帮助。每章末尾都附有旨在促进批判性思维的讨论问题，以及旨在提高你对材料理解的技术练习。

这些讨论问题和练习也非常适合在数据科学、计算机科学或 MBA 课程中使用，以及对自我学习有动力的人。最后一章包含了几个案例研究，有助于作为 MLOps 专家建立工作组合。

本书分为 12 章，我们将在以下部分详细介绍一下。书末还附有一个附录，收录了一些实施 MLOps 的宝贵资源。

## 章节

前几章涵盖了 DevOps 和 MLOps 的理论与实践。其中一个涉及的项目是如何建立持续集成和持续交付。另一个关键主题是 Kaizen，即在各个方面持续改进的理念。

云计算有三章涵盖了 AWS、Azure 和 GCP。作为微软的开发者倡导者，Alfredo 是 Azure 平台 MLOps 知识的理想来源。同样，Noah 多年来一直致力于培训学生云计算，并与 Google、AWS 和 Azure 的教育部门合作。这些章节是熟悉基于云的 MLOps 的绝佳途径。

其他章节涵盖了 MLOps 的关键技术领域，包括 AutoML、容器、边缘计算和模型可移植性。这些主题涵盖了许多具有活跃追踪的前沿新兴技术。

最后，在最后一章中，Noah 讲述了他在社交媒体初创公司的时间以及他们在进行 MLOps 时面临的挑战的一个真实案例研究。

## 附录

附录是一些在完成《Python for DevOps》（O'Reilly）和本书之间几年间出现的文章、想法和宝贵物品的集合。使用它们的主要方法是帮助您做出未来的决策。

## 练习问题

在这本书的练习中，一个有用的启发式方法考虑如何利用它们通过 GitHub 创建一个作品集，并且使用 YouTube 演示你的操作步骤。保持着“图像胜于千言万语”的表达，将一个可重复的 GitHub 项目的 YouTube 链接添加到简历上可能价值 10000 字，并且将简历置于新的职位资格类别之中。

在阅读本书和做练习时，请考虑以下关键思维框架。

## 讨论问题

根据乔纳森·哈伯在《批判性思维》（麻省理工出版社基本知识系列）和非营利组织[批判性思维基金会](https://oreil.ly/FXoTU)的说法，讨论问题是关键的批判性思维组成部分。由于社交媒体上的误信息和浅薄内容的泛滥，世界急需批判性思维。掌握以下技能使个人脱颖而出：

智力谦逊

承认自己知识的局限性。

智力勇气

即使在社会压力面前，也能为自己的信仰辩护的能力。

智力同理

理解他人立场并将自己置于他人思维之中的能力。

智力自主性

在无视他人的情况下独立思考的能力。

智力诚信

以你期望他人对待你的智力标准思考和辩论的能力。

智力毅力

提供支持你立场的证据的能力。

自信的理由

相信有不可辩驳的事实，并且理性是获得知识的最佳解决方案。

公正思维

以诚信的努力对待所有观点的能力。

根据这些标准，评估每章的讨论问题。

## 章节引用的起源

*作者：诺亚*

我在 1998 年底大学毕业，并花了一年时间在美国或欧洲的小联盟训练篮球，同时担任私人教练。我的备胎计划是找 IT 工作。我申请加州理工学院帕萨迪纳分校的系统管理员职位，因机缘巧合获得了 Mac IT 专家职位。我决定低薪职业运动员的风险/回报比不值得，接受了这份工作。

说加州理工改变了我的生活实在是轻描淡写。午餐时，我玩极限飞盘，并听说了 Python 编程语言，我学会了它以便“融入”我的极限飞盘朋友中，他们是加州理工的员工或学生。后来，我直接为加州理工的管理层工作，并成为大卫·巴尔的摩尔博士的个人 Mac 专家，他 30 多岁就获得了诺贝尔奖。我以许多意想不到的方式与许多名人互动，这增强了我的自信心并扩展了我的人脉。

我还与许多后来在人工智能/机器学习领域做出不可思议成就的人们有过像阿甘正传式的随机相遇。有一次，我与斯坦福大学 AI 负责人李飞飞博士和她的男友共进晚餐；我记得被她得她的男友整个夏季都在与他父亲一起编写视频游戏，令我印象深刻。当时我非常赞叹，并想，“谁会做这种事情？”后来，我在著名物理学家戴维·古德斯坦博士的桌子下安装了一个邮件服务器，因为他不断受到 IT 部门对他的邮箱存储限制的苛责。这些经历使我开始对建立“影子基础设施”产生兴趣。因为我直接为管理层工作，所以如果有充分理由，我可以违反规定。

我   我随机遇到的人之一是约瑟夫·博根博士，他是一位神经外科医生，也是加州理工学院的客座教授。在加州理工学院，他对我的生活影响最深远。有一天，我接到一个帮助台呼叫，需要去他家修理他的电脑，后来这演变成了每周与他和他的妻子格伦达在他家吃晚餐的活动。从大约 2000 年开始，直到他去世那天，他一直是我的朋友和导师。

当时，我对人工智能非常感兴趣，我记得加州理工计算机科学教授告诉我这是一个死胡同，我不应该专注于它。尽管如此，我制定了一个计划，到 40 岁时精通许多软件编程语言，并写人工智能程序。果然，我的计划成功了。

如果没有遇见乔·博根，我可以明确地说，我今天不会做我现在在做的事情。当他告诉我他做过第一次半脑切除手术来帮助一名患有严重癫痫的病人时，他让我震惊了。我们会花数小时讨论意识的起源，上世纪七十年代使用神经网络来确定谁将成为空军飞行员，以及你的大脑是否包含“你的两个自我”，一个在每个半球。最重要的是，博根给了我对我的智力的信心。直到那时，我对自己能做什么有严重的怀疑，但我们的对话就像是一个高级思维的硕士学位。作为一名教授，我考虑他对我的生活产生了多大的影响，并且希望向我与之互动的其他学生，无论是正式的教师还是他们所遇到的人，回馈。您可以自己从[博根博士的加州理工个人主页的档案](https://oreil.ly/QPIIi)和他的[传记](https://oreil.ly/EgZQO)阅读这些引用。

# 本书中使用的约定

本书中使用的以下排版约定：

*Italic*

指示新术语、网址、电子邮件地址、文件名和文件扩展名。

`Constant width`

用于程序清单，以及段落内用来指代程序元素如变量或函数名、数据库、数据类型、环境变量、语句和关键字。

**`Constant width bold`**

显示用户应按字面输入的命令或其他文本。

*`Constant width italic`*

显示应替换为用户提供的值或由上下文确定的值的文本。

###### 小贴士

此元素表示提示或建议。

###### 注意

此元素表示一般说明。

###### 警告

此元素表示警告或注意事项。

# Using Code Examples

补充材料（代码示例、练习等）可在[*https://github.com/paiml/practical-mlops-book*](https://github.com/paiml/practical-mlops-book)下载。

如果您有技术问题或在使用代码示例时遇到问题，请发送电子邮件至*bookquestions@oreilly.com*。

本书旨在帮助您完成工作。一般而言，如果此书提供了示例代码，您可以在您的程序和文档中使用它。除非您复制了代码的大部分内容，否则无需联系我们请求权限。例如，编写一个使用本书多个代码块的程序不需要权限。销售或分发 O’Reilly 书籍示例需要权限。引用本书并引用示例代码来回答问题不需要权限。将本书的大量示例代码整合到产品文档中需要权限。

我们感激但通常不要求署名。署名通常包括书名、作者、出版商和 ISBN。例如：“*Practical MLOps* by Noah Gift and Alfredo Deza (O’Reilly). Copyright 2021 Noah Gift and Alfredo Deza, 978-1-098-10301-9.”

如果您认为使用代码示例超出了公平使用范围或上述授权，请随时联系我们，邮件至*permissions@oreilly.com*。

# 致谢

## 来自 Noah

正如前面提到的，如果不是 Mike Loukides 邀请我参加 Foo Camp 并与 Tim O'Reilly 进行了一次很好的讨论，这本书就不会存在。接下来，我要感谢我的合著者 Alfredo。我有幸与 Alfredo 合作写了五本书，其中两本是为 O'Reilly 出版的，另外三本是自我出版的，这主要归功于他接受工作并完成任务的能力。对于努力工作的渴望可能是最好的才能，而 Alfredo 在这方面拥有丰富的技能。

我们的编辑 Melissa Potter 在将事情整理到位方面做了大量工作，而她编辑前后的书几乎是两本不同的书。我感到很幸运能和这样一个才华横溢的编辑合作。

我们的技术编辑，包括 Steve Depp、Nivas Durairaj 和 Shubham Saboo，在提供关于何时要“走弯道”和何时“直道”的出色反馈中发挥了至关重要的作用。许多改进都归功于 Steve 的详细反馈。此外，我还要感谢 Julien Simon 和 Piero Molino，他们用对 MLOps 的实际思考丰富了我们的书籍。

我想感谢我的家人 Liam、Leah 和 Theodore，在疫情期间紧张的截止日期内给我完成这本书的空间。我也期待着未来看到他们写的一些书。另外，我要特别感谢我在西北大学、杜克大学、加州大学戴维斯分校和其他学校教过的所有前学生。他们的许多问题和反馈都融入了这本书中。

最后感谢 Joseph Bogen 博士，他是 AI/ML 和神经科学的早期先驱。如果我们在加州理工学院没有相遇，我绝对不可能成为教授，也不可能有这本书存在。他对我的生活影响如此之大。

## 来自 Alfredo

我在写这本书的过程中，完全感谢我的家人的支持：Claudia、Efrain、Ignacio 和 Alana——你们的支持和耐心对完成这本书至关重要。再次感谢你与我一同工作的所有机会，Noah；这是另一次不可思议的旅程。我珍视我们的友谊和专业关系。

特别感谢 Melissa Potter（毫无疑问是我合作过的最好的编辑）出色的工作。我们的技术编辑做得很好，发现问题并突出需要完善的地方，这总是一件难事。

我也非常感谢 Lee Stott 在 Azure 方面的帮助。没有他，Azure 的内容不会那么好。还要感谢 Francesca Lazzeri、Mike McCoy 和我在 Microsoft 联系过的所有其他人，你们都非常有帮助。
