# 前言

在过去的几年里，一个不断增长的工程师和研究人员社区悄悄地改写了计算机与物理世界互动的规则。结果是一种被称为“边缘人工智能”的技术，它承诺颠覆一个世纪的计算机历史，并触及每个人的生活。

借助一个微小的软件更新，边缘人工智能技术可以赋予廉价、节能的处理器——已经存在于从洗碗机到恒温器的各种设备中——感知和理解世界的能力。我们可以赋予日常物品自己的智能，不再依赖于数据饥渴的集中服务器。下一代工具使得这种魔法变得触手可及，从高中学生到保护研究人员皆可如此。

已经有很多边缘人工智能产品在世界上推出。以下是本书将介绍的一些产品：

+   智能设备通过安装在电力塔上，预测电路故障可能发生的时机，帮助预防由电力传输引起的森林火灾

+   戴在身上的带子能够通过警告来确保消防员在热应激和过度劳累时不会受到威胁

+   提供无需互联网连接的技术的语音用户界面

+   智能项圈能够监测野生大象的活动，帮助研究人员了解它们的行为并保护它们免受冲突的影响

+   野生动物摄像机能够识别特定的动物物种，并帮助科学家了解它们的行为

边缘人工智能技术仍然是新鲜的，这些现有的应用程序只是其可能性的一瞥。随着更多人学习如何使用边缘人工智能，他们将创建解决人类活动各个方面问题的应用程序。

*本书的目标是赋予您成为其中一员的能力。* 我们希望帮助您基于自己独特的视角创建成功的边缘人工智能产品。

# 关于本书

本书旨在为工程师、科学家、产品经理和决策者提供指导，他们将推动这场革命。它是对整个领域的高级指南，提供了解决使用边缘人工智能解决现实问题的工作流程和框架。

除其他事项外，我们希望教会您：

+   各种边缘人工智能技术固有的机会、限制和风险

+   使用人工智能和嵌入式机器学习分析问题和设计解决方案的框架

+   一个端到端的实际工作流程，用于成功开发边缘人工智能应用程序

在书的第一部分，初步章节将介绍和讨论关键概念，帮助您了解全局局势。接下来的几章将带您完成实际的流程，帮助您设计和实施自己的应用程序。

在本书的第二部分中，从第十一章开始，我们将通过三个端到端的实例演示如何将您的知识应用于解决科学、工业和消费项目中的实际问题。

到本书结束时，你将自信地通过边缘人工智能的视角看待世界，并拥有一套可靠的工具，可以帮助你构建有效的解决方案。

###### 注意

本书涵盖了许多主题！如果你想了解我们包含的所有内容，请快速查看目录。

# 期望什么

本书不是一本编程书籍或特定工具的教程，所以不要期望有大量逐行代码解释或使用特定软件的逐步指南。相反，你将学习如何应用通用框架来解决问题，使用最适合工作的工具。

话虽如此，这是一个极大受益于具体、互动示例的主题，可以进行探索、定制和构建。在本书的过程中，我们将提供各种你可以探索的工件——从 Git 仓库到免费在线数据集和示例训练管道。

其中许多内容将托管在[Edge Impulse](https://edgeimpulse.com)，这是一个用于构建边缘人工智能应用程序的工程工具。¹ 它基于开源技术和标准最佳实践构建，因此即使在不同平台上进行自己的工作，你也能理解这些原则。本书的作者们都是 Edge Impulse 的忠实粉丝——但也许有些偏见，因为他们是建立它的团队的一部分！

###### 注意

为了保证可移植性，机器学习管道的所有工件都可以在 Edge Impulse 中以开放格式导出，包括数据集、机器学习模型以及任何信号处理代码的 C++实现。

# 你需要已经了解的内容

本书讨论的是在边缘设备上运行的软件构建，因此熟悉嵌入式开发的高级概念将是有帮助的。这可能涉及到资源受限设备，如微控制器或数字信号处理器（DSP），或通用设备，如嵌入式 Linux 计算机。

话虽如此，如果你刚开始接触嵌入式软件，你应该没有困难跟上！我们将保持简单，并在需要时引入新的主题。

此外，并不假设任何特定的知识。由于本书的目标是为整个工程领域提供一个实用的路线图，我们将以高层次涵盖许多主题。如果你对我们提到的任何内容感兴趣——从机器学习的基础知识到机器学习应用设计的基本要素——我们将提供大量我们自己学习中发现有用的资源。

# 负责、道德和有效的人工智能

构建任何类型的应用程序的最重要部分是确保它在现实世界中正常工作。不幸的是，人工智能应用程序特别容易出现一类问题，使它们在表面上*看起来*运行良好，而实际上它们却在很多时候以非常有害的方式失败。

避免这类问题将是本书的核心主题——如果不是*唯一*的核心主题。因为现代 AI 开发是一个迭代过程，仅在工作流的最后测试系统是否有效是不够的。相反，您需要在每一个步骤中考虑潜在的陷阱。您必须理解风险所在，批判性地审查中间结果，并做出考虑到利益相关者需求的知情决策。

在本书的过程中，我们将介绍一个强大的框架，帮助您理解、推理、衡量性能，并基于对构建 AI 应用程序可能出现问题的意识做出决策。这将成为我们整个开发过程的基础，并塑造我们设计应用程序的方式。

这一过程始于项目的最初构思阶段。要构建有效的应用程序，理解我们当前的人工智能方法并不适合某些用例至关重要。在许多情况下，造成的风险——无论是物理、财务还是社会的——都超过了部署 AI 的潜在利益。本书将教您如何识别这些风险，并在探索项目可行性时加以考虑。

作为领域专家，我们有责任确保我们创建的技术得到恰当的使用。没有其他人比我们更适合做这项工作，因此我们有责任做好。本书将帮助您做出正确的决策，创建性能优越、避免伤害并造福更广泛世界的应用程序。

# 进一步资源

一本涵盖从低级实现到高级设计模式的嵌入式 AI 的书将有整整一整个书架大！与其试图把所有内容塞进一卷书中，您正在阅读的书籍将为整个领域提供详细但高层次的路线图。

为了深入研究与您的特定项目相关的细枝末节，《“学习边缘 AI 技能”》推荐了大量进一步的资源。

# 本书中使用的约定

本书使用以下排版约定：

*斜体*

指示新术语、URL、电子邮件地址、文件名和文件扩展名。

`常宽`

用于程序清单，以及段落中引用程序元素，如变量或函数名，数据库，数据类型，环境变量，语句和关键字。

**`常宽粗体`**

显示用户应直接输入的命令或其他文本。

*`常宽斜体`*

显示应由用户提供值或根据上下文确定值的文本。

###### 提示

此元素表示提示或建议。

###### 注意

此元素表示一般性说明。

###### 警告

此元素表示警告或注意。

# 使用代码示例

可以下载补充材料（代码示例、练习等）[*https://github.com/ai-at-the-edge*](https://github.com/ai-at-the-edge)。

如果您有技术问题或使用代码示例时遇到问题，请发送电子邮件至*bookquestions@oreilly.com*。

本书的目的是帮助您完成工作。一般情况下，如果本书提供示例代码，您可以在您的程序和文档中使用它。除非您复制了代码的大部分内容，否则无需征得我们的许可。例如，编写一个使用本书多个代码片段的程序无需许可。销售或分发奥莱利书籍中的示例则需要许可。通过引用本书回答问题并引用示例代码无需许可。将本书大量示例代码合并到产品文档中则需要许可。

我们感谢，但通常不需要，署名。署名通常包括标题、作者、出版商和 ISBN。例如：“*AI at the Edge* by Daniel Situnayake and Jenny Plunkett (O’Reilly)。Copyright 2023 Daniel Situnayake and Jenny Plunkett, 978-1-098-12020-7。”

如果您认为您对代码示例的使用超出了公平使用范围或上述许可，请随时与我们联系*permissions@oreilly.com*。

# 奥莱利在线学习

###### 注释

[*奥莱利媒体*](http://oreilly.com)已提供技术和商业培训、知识和见解超过 40 年，帮助企业取得成功。

我们独特的专家和创新者网络通过书籍、文章和我们的在线学习平台分享他们的知识和专业知识。奥莱利的在线学习平台为您提供按需访问的现场培训课程、深入学习路径、互动编码环境以及来自奥莱利和其他 200 多家出版商的大量文本和视频。更多信息，请访问[*http://oreilly.com*](http://oreilly.com)。

# 如何联系我们

请将有关本书的评论和问题发送至出版商：

+   奥莱利媒体，公司

+   1005 Gravenstein Highway North

+   Sebastopol，CA 95472

+   800-998-9938（在美国或加拿大）

+   707-829-0515（国际或本地）

+   707-829-0104（传真）

我们为本书提供了一个网页，列出勘误、示例和任何其他信息。您可以访问[*https://oreil.ly/ai-at-the-edge*](https://oreil.ly/ai-at-the-edge)。

电子邮件*bookquestions@oreilly.com*以评论或询问本书的技术问题。

欲了解我们的书籍和课程的新闻和信息，请访问[*https://oreilly.com*](https://oreilly.com)。

在 LinkedIn 上找到我们：[*https://linkedin.com/company/oreilly-media*](https://linkedin.com/company/oreilly-media)。

在 Twitter 上关注我们：[*https://twitter.com/oreillymedia*](https://twitter.com/oreillymedia)。

观看我们的 YouTube 频道：[*https://youtube.com/oreillymedia*](https://youtube.com/oreillymedia)。

# 致谢

没有大量人员的辛勤工作和支持，这本书将无法问世，对此我们深表感激。

我们荣幸地得到了[Pete Warden](https://petewarden.com)的序言支持，他不仅是一个具有远见卓识的技术专家，功不可没地推动了这一领域的发展，同时也是一位了不起的人类和伟大的朋友。非常感谢你的支持，Pete！

我们深深感谢[Wiebke (Toussaint) Hutiri](https://wiebketoussaint.com)，她在帮助塑造和提供本书中的负责任 AI 内容方面付出了超乎寻常的努力，包括为"负责任设计与 AI 伦理"贡献了一篇出色的引言。你是你领域的明星。

我们对我们不可思议的技术审阅和顾问小组深表感激，他们的智慧和见解对本书产生了巨大影响。他们是：Alex Elium, Aurélien Geron, Carlos Roberto Lacerda, David J. Groom, Elecia White, Fran Baker, Jen Fox, Leonardo Cavagnis, Mat Kelcey, Pete Warden, Vijay Janapa Reddi, 和 Wiebke (Toussaint) Hutiri。特别感谢 Benjamin Cabé允许我们展示他的人工鼻子项目。任何不准确之处均由作者负责。

我们还要感谢 O'Reilly 的出色团队，特别是 Angela Rufino，在整个写作过程中以最大的理解和关怀指导我们。特别感谢 Elizabeth Faerm, Kristen Brown, Mike Loukides, Nicole Taché, 和 Rebecca Novack。

这本书的存在离不开我们在 Edge Impulse 团队的支持，这是一群无比英勇的明星。特别感谢创始人 Zach Shelby 和 Jan Jongboom，他们相信我们对这本书的愿景，支持我们实现它，并创造了一个可以让思想迸发的空间。对全体团队，包括但不限于：Adam Benzion, Alessandro Grande, Alex Elium, Amir Sherman, Arjan Kamphuis, Artie Beavis, Arun Rajasekaran, Ashvin Roharia, Aurelien Lequertier, Carl Ward, Clinton Oduor, David Schwarz, David Tischler, Dimi Tomov, Dmitry Maslov, Emile Bosch, Eoin Jordan, Evan Rust, Fernando Jiménez Moreno, Francesco Varani, Jed Huang, Jim Edson, Jim van der Voort, Jodie Lane, John Pura, Jorge Silva, Joshua Buck, Juliette Okel, Keelin Murphy, Kirtana Moorthy, Louis Moreau, Louise Paul, Maggi Yang, Mat Kelcey, Mateusz Majchrzycki, Mathijs Baaijens, Mihajlo Raljic, Mike Senese, Mikey Beavis, MJ Lee, Nabil Koroghli, Nick Famighetti, Omar Shrit, Othman Mekhannene, Paige Holvik, Raul James, Raul Vergara, RJ Vissers, Ross Lowe, Sally Atkinson, Saniea Akhtar, Sara Olsson, Sergi Mansilla, Shams Mansoor, Shawn Hanscom, Shawn Hymel, Sheena Patel, Tyler Hoyle, Vojislav Milivojevic, William DeLey, Yan Li, Yana Vibe, 和 Zin Kyaw。你们创造了奇迹。

Jenny 感谢她在德克萨斯州的家人和朋友多年来给予的大力支持，还有她的猫蓝基因和比阿特丽斯作为最好的合作伙伴。她特别感谢她的父亲迈克尔·普朗克特，他鼓励她在奥斯汀德克萨斯大学攻读电气工程，并激发了她对新技术终身的好奇心。

Dan 感谢他的家人和朋友在每次大冒险中的支持。他深深感激 Lauren Ward 在我们所有旅程中的爱和伴侣关系。他还感谢 Minicat 给予他平静的猫咪陪伴，并允许在本书中使用她的照片。

¹ Edge Impulse 在学术论文 [“Edge Impulse: An MLOps Platform for Tiny Machine Learning”](https://oreil.ly/Dyd-Z)（S. Hymel 等人，2022 年）中有详细描述。