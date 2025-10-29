# 前言

我们为那些正在构建希望投入生产的机器学习系统/模型的数据工程师和数据科学家而写此书。如果您曾经训练出了一个优秀的模型，却不知如何将其部署到生产环境或在部署后如何保持更新，那么这本书适合您。我们希望这本书能帮助您用相对可靠的方式替换`Untitled_5.ipynb`。

本书不适合作为您第一次接触机器学习的入门书籍。下一节将指出一些资源，如果您刚开始机器学习之旅可能会有所帮助。

# 我们对您的假设

本书假设您要么已经了解如何在本地训练模型，要么正在与了解此过程的人合作。如果以上都不是，那么有许多出色的机器学习入门书籍可以帮助您入门，包括[Aurélien Géron 的《Python 机器学习实战》第二版，使用 Scikit-Learn、Keras 和 TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632)（O’Reilly）。

我们的目标是教会您如何以可重复的方式进行机器学习，并自动化您的模型训练和部署。这个目标涉及到一系列广泛的主题，您可能不太熟悉其中的所有内容是非常合理的。

由于我们无法深入研究每个主题，因此我们希望为您提供一个我们最喜欢的入门指南的简短列表：

+   [*Python 数据分析*](https://learning.oreilly.com/library/view/python-for-data/9781491957653)，第二版，作者：Wes McKinney（O’Reilly）

+   [*从零开始的数据科学*](https://learning.oreilly.com/library/view/data-science-from/9781492041122)，第二版，作者：Joel Grus（O’Reilly）

+   [*使用 Python 进行机器学习入门*](https://www.oreilly.com/library/view/introduction-to-machine/9781449369880)，作者：Andreas C. Müller 和 Sarah Guido（O’Reilly）

+   [*Python 机器学习实战*](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632)，第二版，作者：Aurélien Géron（O’Reilly）

+   [*Kubernetes 入门与实战*](https://learning.oreilly.com/library/view/kubernetes-up-and/9781492046523)，作者：Brendan Burns 等（O’Reilly）

+   [*学习 Spark*](https://learning.oreilly.com/library/view/learning-spark/9781449359034)，作者：Holden Karau 等（O’Reilly）

+   [*机器学习特征工程*](https://learning.oreilly.com/library/view/feature-engineering-for/9781491953235)，作者：Alice Zheng 和 Amanda Casari（O’Reilly）

+   [*构建机器学习流水线*](https://learning.oreilly.com/library/view/building-machine-learning/9781492053187)，作者：Hannes Hapke 和 Catherine Nelson（O’Reilly）

+   *Apache Mahout：超越 MapReduce*，作者：Dmitriy Lyubimov 和 Andrew Palumbo（CreateSpace）

+   [*R 手册*](https://learning.oreilly.com/library/view/r-cookbook-2nd/9781492040675)，第二版，作者 J. D. Long 和 Paul Teetor（O’Reilly）

+   [*机器学习模型的服务*](https://www.oreilly.com/library/view/serving-machine-learning/9781492024095)，作者 Boris Lublinsky（O’Reilly）

+   [“机器学习的持续交付”](https://oreil.ly/y59_n)，作者 Danilo Sato 等

+   [*可解释的机器学习*](https://oreil.ly/hBiw1)，作者 Christoph Molnar（自出版）

+   [“机器学习中概念漂移的简介”](https://oreil.ly/KnJL0)，作者 Jason Brownlee

+   [“模型漂移与确保健康的机器学习生命周期”](https://oreil.ly/q9o6P)，作者 A. Besir Kurtulmus

+   [“模型服务器的崛起”](https://oreil.ly/zvIyU)，作者 Alex Vikati

+   [“现代机器学习中模型可解释性概述”](https://oreil.ly/lo36s)，作者 Rui Aguiar

+   [*Python 机器学习手册*](https://learning.oreilly.com/library/view/machine-learning-with/9781491989371)，作者 Chris Albon（O’Reilly）

+   [机器学习闪卡](https://machinelearningflashcards.com)，作者 Chris Albon

当然，还有许多其他资源，但这些可以帮助你入门。请不要被这个清单吓倒—你并不需要成为这些主题的专家来有效地部署和管理 Kubeflow。实际上，Kubeflow 存在的目的就是简化这些任务。

容器和 Kubernetes 是一个广泛且快速发展的领域。如果你想深入了解 Kubernetes，我们建议查看以下内容：

+   [*云原生基础设施*](https://learning.oreilly.com/library/view/cloud-native-infrastructure/9781491984291)，作者 Justin Garrison 和 Kris Nova（O’Reilly）

+   [*Kubernetes 实战*](https://learning.oreilly.com/library/view/kubernetes-up-and/9781492046523)，作者 Brendan Burns 等（O’Reilly）

# 作为从业者的责任

本书帮助你将机器学习模型投入生产解决现实世界的问题。用机器学习解决现实世界的问题很棒，但在应用你的技能时，请记得考虑其影响。

首先，确保你的模型足够准确非常重要，在 Kubeflow 中有很多强大的工具，这在“训练和部署模型”章节有详细介绍。即使是最好的工具也不能保证你免于所有错误—例如，在同一数据集上进行超参数调整以报告最终交叉验证结果。

即使具有显著预测能力的模型在常规的训练评估阶段可能不会显示出意外的效果和偏见。意外的偏见可能很难发现，但有许多故事（例如，[亚马逊基于机器学习的招聘引擎后来被发现存在严重偏见，决定只招聘男性](https://oreil.ly/VekPG)）显示了我们工作的深远潜在影响。早期不解决这些问题可能会导致不得不放弃整个工作，就像[IBM 决定停止其面部识别程序](https://oreil.ly/WKUXl)以及在警方手中的面部识别中种族偏见影响明显后，行业中类似暂停一样。

即使表面上没有偏见的数据，如原始购买记录，最终也可能存在严重偏见，导致不正确的推荐甚至更糟糕的情况。公开并广泛可用的数据集并不意味着它没有偏见。众所周知的[word embeddings](https://oreil.ly/1dmOV)做法已被证明包含许多种类的偏见，包括性别歧视、反 LGBTQ 和反移民。在查看新数据集时，查找数据中偏见的例子并尽可能减少这些偏见至关重要。对于最流行的公开数据集，研究中经常讨论各种技术，您可以借鉴这些来指导自己的工作。

虽然本书没有解决偏见的工具，但我们鼓励您在投入生产之前，对系统中潜在的偏见进行批判性思考，并探索解决方案。如果您不知道从哪里开始，请查看 Katharine Jarmul 的[出色的入门讲座](https://oreil.ly/fiVYL)。IBM 在其[AI 公正性 360 开源工具包](http://aif360.mybluemix.net)中收集了工具和示例，这可能是开始探索的好地方。减少模型中偏见的关键步骤之一是拥有一个多样化的团队，以便及早发现潜在问题。正如[Jeff Dean](https://oreil.ly/PJNsF)所说：“AI 充满了承诺，有潜力彻底改变现代社会的许多不同领域。为了实现其真正的潜力，我们的领域需要对所有人都友好。但事实并非如此。我们的领域在包容性方面存在问题。”

###### 提示

需要注意的是，在你的结果中消除偏见或验证准确性并不是一蹴而就的事情；模型性能可能会随时间降低，甚至会引入偏见——即使你个人没有做任何改变。¹

# 本书使用的约定

本书使用以下排版约定：

*斜体*

表示新术语、网址、电子邮件地址、文件名和文件扩展名。

`等宽字体`

用于程序清单，以及段落内部引用程序元素，如变量或函数名称、数据库、数据类型、环境变量、语句和关键字。

**`固定宽度粗体`**

显示用户应该按字面意义输入的命令或其他文本。

*`固定宽度斜体`*

显示应用用户提供的值或由上下文确定的值的文本。

###### 提示

此元素表示提示或建议。

###### 注意

此元素表示一般提示。

###### 警告

此元素表示警告或注意事项。

我们将使用警告指出任何可能导致结果管道不可移植的情况，并呼吁您可以使用的可移植替代方法。

# 代码示例

可下载补充材料（例如代码示例等），网址为[*https://oreil.ly/Kubeflow_for_ML*](https://oreil.ly/Kubeflow_for_ML)。这些代码示例根据 Apache 2 许可或下一节中描述的许可证提供。

有其他示例在它们自己的各自许可证下，您可能会发现它们有用。Kubeflow 项目有一个[示例仓库](https://oreil.ly/yslNT)，在撰写本文时可根据 Apache 2 许可获取。Canonical 还为 MicroK8s 用户提供了一套资源，可在[此处](https://oreil.ly/TOt_E)找到。

## 使用代码示例

如果您有技术问题或使用代码示例时遇到问题，请发送电子邮件至*bookquestions@oreilly.com*。

本书旨在帮助您完成工作。通常，如果本书提供了示例代码，您可以在您的程序和文档中使用它。除非您重复使用了本书中的大量代码片段，否则无需联系我们请求许可。出售或分发奥莱利图书的示例代码需要许可。引用本书并引用示例代码回答问题无需许可。将本书中大量示例代码合并到产品文档中需要许可。

更多许可细节可以在存储库中找到。

我们感谢但通常不要求署名。署名通常包括标题、作者、出版商和 ISBN。例如：“*Kubeflow for Machine Learning* by Holden Karau, Trevor Grant, Boris Lublinsky, Richard Liu, and Ilan Filonenko (O’Reilly). Copyright 2021 Holden Karau, Trevor Grant, Boris Lublinsky, Richard Liu, and Ilan Filonenko, 978-1-492-05012-4.”

如果您觉得您使用的代码示例超出了合理使用范围或上述许可，请随时通过*permissions@oreilly.com*与我们联系。

# 如何联系作者

欲提供反馈，请发送邮件至*intro-to-ml-kubeflow@googlegroups.com*。偶尔也会关于 Kubeflow 的随意言论，请在线关注我们：

特雷弗

+   [Twitter](https://twitter.com/rawkintrevo)

+   [博客](https://rawkintrevo.org)

+   [GitHub](https://github.com/rawkintrevo)

+   [Myspace](https://myspace.com/rawkintrevo)

霍尔登

+   [Twitter](http://twitter.com/holdenkarau)

+   [YouTube](https://www.youtube.com/user/holdenkarau)

+   [Twitch](https://www.twitch.tv/holdenkarau)

+   [领英](https://www.linkedin.com/in/holdenkarau)

+   [博客](http://blog.holdenkarau.com)

+   [GitHub](https://github.com/holdenk)

+   [Facebook](https://www.facebook.com/hkarau)

勃里斯

+   [领英](https://www.linkedin.com/in/boris-lublinsky-b6a4a/)

+   [GitHub](https://github.com/blublinsky)

理查德

+   [GitHub](https://github.com/richardsliu)

伊兰

+   [领英](https://www.linkedin.com/in/ifilonenko)

+   [GitHub](https://github.com/ifilonenko)

# 致谢

作者们想要感谢 O’Reilly Media 的所有人，特别是我们的编辑 Amelia Blevins 和 Deborah Baker，以及 Kubeflow 社区使这本书的出版成为可能。来自 [Seldon](https://www.seldon.io) 的 Clive Cox 和 Alejandro Saucedo 在 第八章 中做出了重要贡献，否则这本书将缺少关键部分。我们要感谢 Google Cloud Platform 提供的资源，使我们能够确保示例在 GCP 上运行。也许最重要的是，我们要感谢我们的审阅者，没有他们，这本书就不会以目前的形式存在。这包括 Taka Shinagawa，Pete MacKinnon，Kevin Haas，Chris Albon，Hannes Hapke 等人。感谢所有早期读者和书籍审阅者，感谢你们的贡献。

Holden

想要感谢她的女友 Kris Nóva，在调试她的第一个 Kubeflow PR 方面给予的帮助，以及整个 Kubeflow 社区对她的热情欢迎。她还要感谢她的妻子 Carolyn DeSimone，她的小狗 Timbit DeSimone-Karau（见图 P-1），以及她的玩偶们给予的写作支持。她要感谢 SF General 和 UCSF 的医生，让她能够完成这本书的写作（尽管她希望手不再疼痛），以及在医院和护理院探望她的每一个人。特别感谢 Ann Spencer，第一位教会她如何享受写作的编辑。最后，她要感谢她的约会伴侣 Els van Vessem，在她意外后的康复中给予的支持，尤其是通过阅读故事和提醒她对写作的热爱。

![Timbit](img/kfml_0001.png)

###### 图 P-1\. Timbit the dog

Ilan

想要感谢所有在 Bloomberg 工作的同事们，他们花时间审阅、指导和鼓励他参与开源贡献。名单包括但不限于：Kimberly Stoddard，Dan Sun，Keith Laban，Steven Bower 和 Sudarshan Kadambi。他还要感谢他的家人—Galia，Yuriy 和 Stan，给予他无条件的爱和支持。

Richard

想要感谢 Google Kubeflow 团队，包括但不限于：Jeremy Lewi，Abhishek Gupta，Thea Lamkin，Zhenghui Wang，Kunming Qu，Gabriel Wen，Michelle Casbon 和 Sarah Maddox—没有他们的支持，这一切都不可能实现。他还想感谢他的猫 Tina（见图 P-2）在 COVID-19 期间的支持和理解。

![Tina](img/kfml_0002.png)

###### 图 P-2\. Tina the cat

Boris

想要感谢他在 Lightbend 的同事们，特别是 Karl Wehden，在书写过程中给予的支持，对初版文本的建议和校对，以及他的妻子 Marina，在他长时间写作时的支持和供应。

Trevor

Trevor 想要感谢他的办公室同事 Apache 和 Meowska（见 图 P-3），提醒他午睡的重要性，以及去年聆听他关于 Kubeflow 的演讲的所有人（特别是那些听了糟糕版本，尤其尤其是听了糟糕版本但现在依然在阅读本书的人——你们是最棒的）。他还想要感谢他的妈妈、姐姐和弟弟多年来对他各种古怪行为的包容。

![Apache 和 Meowska](img/kfml_0003.png)

###### 图 P-3\. Apache 和 Meowska

# 抱怨

作者还要感谢 API 变更带来的挑战，这使得写作本书变得如此令人沮丧。如果你也遇到 API 变更的困扰，要知道你并不孤单；它们几乎让每个人都感到恼火。

Holden 还想要感谢 Timbit DeSimone-Karau 曾经在她工作时捣乱挖坑的时刻。我们有一个特别的抱怨要向那个撞到 Holden 的人宣泄，导致本书发布进度减慢。

Trevor 有一个抱怨要向他的女朋友发表，她一直在这个项目中坚持要他求婚，而他一直在“努力”——如果在这本书出版前他还没向她求婚的话：**凯蒂，你愿意嫁给我吗？**

¹ 记得那个通过强化学习成为新纳粹的 Twitter 机器人吗？
