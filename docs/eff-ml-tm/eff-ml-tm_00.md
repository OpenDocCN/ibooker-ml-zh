# 前言

> *现在是晚上 9:25，Dana 的电脑屏幕柔和的光芒刺痛了她疲惫的眼睛，当她登录继续修复一个错误——屏幕上满是红色的管道和无数打开的标签页。她已经吃过晚饭并完成了日常的杂事，但她的思维并不在那里——实际上，她的思维分散到几个地方了。*
> 
> *这是一个紧张的一天，在长时间的训练跑步和与支持团队来回沟通客户关于为何模型拒绝他们的贷款申请的问题的消息之间。她不断深入调试为何尽管对数据和模型架构进行了各种调整，但模型性能却无法提高。偶尔出现的堆栈跟踪只会让情况变得更糟。*
> 
> *她很累，本地机器上堆积的未提交代码变更增加了她头脑中潜在的认知负荷。但她必须继续前行——她的团队已经比最初的发布日期迟了四个月，高管们的不耐烦也表现出来了。更糟糕的是，她担心自己的工作可能岌岌可危。在公司，她认识的几位同事中有十分之一在最新一轮的削减成本措施中被裁员。*
> 
> *团队中的每个人都有良好的意图和能力，但他们每天都在琐碎的测试、充满焦虑的生产部署和翻阅难以理解和脆弱的代码的泥沼中陷入困境。几个月的辛苦之后，他们都筋疲力尽。他们尽力而为，但感觉像是在建造一个没有基础的房子——事情一直在崩塌。*

许多人在机器学习（ML）的旅程中开始时势头强劲，很快获得了信心，这要归功于日益完善的工具、技术、教程和 ML 从业者社区生态系统。然而，当我们从教程笔记本和 Kaggle 竞赛的受控环境毕业，进入到现实问题、混乱的数据、互联系统和拥有不同目标的人群中，许多人不可避免地在实践中难以实现 ML 的潜力。

当我们剥去数据科学是最性感职业的华丽宣称时，我们常常看到 ML 从业者陷入繁重的手动工作、复杂和脆弱的代码库以及无法在生产中看到一丝曙光的令人沮丧的 Sisyphean ML 实验中。

在 2019 年，据报道[87%的数据科学项目从未进入生产](https://oreil.ly/xy9Xi)阶段。根据[Algorithmia 2021 年企业 AI/ML 趋势报告](https://oreil.ly/HP6Qh)，即使是那些成功部署了 ML 模型的公司中，64%的调查对象表示部署新模型需要超过一个月的时间，这一比例从 2020 年的 56%增长。Algorithmia 还发现，有 38%的受访组织在模型部署上花费了超过 50%的数据科学家时间。

这些障碍阻碍了——或者在某些情况下甚至阻止了——机器学习从业者将他们的专业知识应用到为客户和企业提供人工智能的价值和承诺中。但好消息是事情并不一定要这样。在过去几年中，我们有幸参与了各种数据和机器学习项目，并与来自多个行业的机器学习从业者合作。尽管存在障碍和困难，正如我们上面所述，但也有更好的路径、实践和工作系统，使得机器学习从业者能够可靠地将支持机器学习的产品交付到客户手中。

这本书讲的就是这个。我们将从我们的经验中汲取，提炼出一套持久的原则和实践，这些原则和实践一直帮助我们有效地在现实世界中交付机器学习解决方案。这些实践之所以有效，是因为它们基于一种全面的方法来构建机器学习系统。它们不仅仅局限于机器学习，还在各个子系统中（如产品、工程、数据、交付流程、团队拓扑）创建了必要的反馈循环，并使团队能够快速安全地失败、快速实验和可靠交付。

# 本书适合谁？

> 无论你认为你可以，还是认为你不行——你都是对的。
> 
> 亨利·福特

无论您是学术界的机器学习从业者，企业、初创公司、规模化公司还是咨询公司，本书中的原则和实践都可以帮助您和您的团队在交付机器学习解决方案方面更加有效。与本书详细介绍的跨功能性机器学习交付技术一致，我们关注团队中多个角色的关切和愿望：

数据科学家和机器学习工程师

过去几年来，数据科学家的工作范围发生了变化。我们不再仅关注建模技术和数据分析，而是看到对一个[全栈数据科学家](https://oreil.ly/jV7EP)的期望（无论是隐性还是显性）：数据整理、机器学习工程、MLOps 和业务案例制定等能力。本书详细阐述了数据科学家和机器学习工程师在设计和交付现实世界中的机器学习解决方案所需的能力。

在过去，我们向数据科学家、机器学习工程师、博士生、软件工程师、质量分析师和产品经理介绍了本书中的原则、实践和实际练习，并且我们一直收到积极的反馈。我们在行业中合作过的机器学习从业者表示，他们受益于诸如自动化测试和重构等实践所带来的反馈周期、流程性和可靠性的改进。我们的结论是*机器学习社区渴望*学习这些技能和实践，而这正是我们尝试扩展这些知识分享的途径。

软件工程师、基础设施和平台工程师、架构师

当我们在这本书中涵盖的主题上开展研讨会时，我们经常遇到软件工程师、基础设施和平台工程师以及从事 ML 领域架构师。虽然来自软件世界的能力（例如基础设施即代码、部署自动化、自动化测试）在设计和交付现实世界中的 ML 解决方案中是必要的，但它们并不足够。要构建可靠的 ML 解决方案，我们需要扩展软件视角，并看看其他原则和实践——例如 ML 模型测试、双轨交付、持续发现和 ML 治理——以应对 ML 独特的挑战。

产品经理、交付经理、工程经理

如果我们认为我们只需要数据科学家和 ML 工程师来构建 ML 产品，我们就设定了自己的失败。相反，我们的经验告诉我们，当团队是跨职能的，并且具备必要的 ML、数据、工程、产品和交付能力时，它们是最有效的。

在本书中，我们详细阐述了如何应用精益交付实践和系统思维来创建结构，帮助团队专注于客户的声音，缩短反馈循环，快速且可靠地进行实验，并循序渐进地构建正确的事物。正如[W. Edwards Deming](https://oreil.ly/eUxHc)曾经说过的那样，“一个糟糕的系统每次都会打败一个好人。” 因此，我们分享了将帮助团队创建结构的原则和实践，优化信息流，减少浪费（例如交接、依赖关系），并提升价值的方法。

如果我们做得对，这本书将邀请您仔细观察 ML 和您的团队中“一直如何做”的事物，反思它们对您的工作效果如何，并考虑更好的替代方案。请以开放的心态阅读本书，并且——对于工程重点的章节——要有一种开放的代码编辑器的态度。正如[Peter M. Senge](https://oreil.ly/9HEwI)在他的书《第五项修炼》（Doubleday）中所说，“吸收信息与真正的学习只有很远的关系。说‘我刚读了一本关于骑自行车的好书——现在我已经学会了’是荒谬的。” 我们鼓励您在团队中尝试这些实践，并希望您第一手地体验到它们在实际项目中带来的价值。

以持续改进的心态来对待这本书，而不是完美主义的心态。没有完美的项目可以完全没有挑战地运行。复杂性和挑战永远存在（我们知道适当的挑战对成长至关重要），但本书中的实践将帮助您减少*偶然*复杂性，使您能够集中精力处理您的 ML 解决方案的*本质*复杂性，并负责地交付价值。

# 本书的组织方式

第一章，“在交付机器学习解决方案中的挑战与更好路径”，是整本书的精华。我们探讨了机器学习项目失败的高层和低层原因。然后，我们提出了通过在产品、交付、机器学习、工程和数据的五个关键学科中采用精益交付实践，为交付机器学习解决方案的价值更可靠的路径。

在其余章节中，我们描述了有效机器学习团队和机器学习从业者的实践。在 第一部分，“产品和交付” 中，我们详细阐述了在其他子系统中实施机器学习解决方案所必需的实践，例如产品思维和精益交付。在 第二部分，“工程” 中，我们涵盖了帮助机器学习从业者在实施和交付解决方案时的实践（例如自动化测试、重构、有效使用代码编辑器、持续交付和 MLOps）。在 第三部分，“团队” 中，我们探索影响机器学习团队效果的动态因素，例如信任、共享进展、多样性，以及帮助您构建高绩效团队的工程效能技术。我们还讨论了组织在超越一个或两个团队时扩展机器学习实践所面临的常见挑战，并分享了关于团队拓扑、交互模式和领导力的技术，帮助团队克服这些扩展挑战。

## 第一部分：产品和交付

第二章，“机器学习团队的产品和交付实践”

我们讨论了产品发现技术，帮助我们快速识别机会、测试市场和技术假设，并收敛于可行的解决方案。通过从最有价值的问题和可行的解决方案开始，我们在交付过程中为成功打下基础。我们还介绍了帮助我们塑造、规模化和顺序化工作的交付实践，以创建稳定的价值流。我们处理了由某些机器学习问题的实验性和高不确定性特性导致的独特挑战，并讨论了诸如双轨交付模型等技术，帮助我们在较短周期内更快地学习。最后，我们介绍了测量机器学习项目关键方面的技术，并分享了识别和管理项目风险的技术。

## 第二部分：工程

章节 3 和 4：有效的依赖管理

在这里，我们描述了原则和实践，以及一个你可以与之并行编码的实际示例，用于创建一致、可重现、安全且类生产的运行时环境来运行你的代码。当我们迅速起步并开始交付解决方案时，你将看到本章节中的实践如何使你和你的团队能够轻松地“检出并运行”，而不是陷入依赖地狱。

章节 5 和 6：ML 系统的自动化测试

这些章节为您提供了测试 ML 解决方案组件的评分表——无论是软件测试、模型测试还是数据测试。我们展示了自动化测试如何帮助我们缩短反馈周期，减少手动测试的繁琐工作，或者更糟糕的是修复在手动测试中漏过的生产缺陷。我们描述了软件测试范式在 ML 模型上的局限性，以及 ML 适应函数和行为测试如何帮助我们扩展 ML 模型的自动化测试。我们还涵盖了全面测试大型语言模型（LLMs）和 LLM 应用程序的技术。

第七章，“用简单技巧加速您的代码编辑器”

我们将向您展示如何配置您的代码编辑器（PyCharm 或 VS Code），以帮助您更有效地编码。在我们完成几个步骤来配置我们的 IDE 后，我们将介绍一系列快捷键，可以帮助您自动化重构、自动检测和修复问题，并在代码库中导航而不会迷失在细节中，等等。

第八章，“重构与技术债务管理”

在本章中，我们借鉴软件设计的智慧，帮助设计可读、可测试、可维护和可演化的代码。在“学以致用”的精神下，您将看到我们如何将有问题、混乱且脆弱的笔记本应用重构技术，逐步改进我们的代码库，使其变为模块化、经过测试和可读的状态。您还将学习技术，帮助您和您的团队使技术债务可见，并采取措施保持其在健康水平。

第九章，“ML 的 MLOps 和持续交付（CD4ML）”

我们将详细阐述 MLOps 和 CI/CD（持续集成和持续交付）的真正内涵。剧透警告：它不仅仅是自动化模型部署和定义 CI 管道。我们提出了适用于 ML 项目的 CI/CD 独特架构的蓝图，并指导您如何设置每个组件，以创建可靠的 ML 解决方案，并解放您的团队成员，摆脱重复和同质化的劳动，让他们专注于其他更有价值的问题。我们还将探讨 CD4ML 如何作为一种风险控制机制，帮助团队维护 ML 治理和负责任 AI 的标准。

## 第三部分：团队

第十章，“构建高效 ML 团队的基础模块”

在本章中，我们超越机械性的方法，理解促进有效团队实践的人际因素。我们将描述有助于创建安全、以人为中心和以增长为导向的团队的原则和实践。我们将探讨信任、沟通、共享目标、有目的的进展以及团队中的多样性等主题。我们会分享一些需要注意的反模式以及您可以使用的策略，来培养协作、有效交付和学习文化。

第十一章，“有效的机器学习组织”

本章介绍了机器学习团队的各种形式，并解决了组织在将其机器学习实践扩展到多个团队时面临的常见挑战。我们借鉴并调整了《团队拓扑》（IT Revolution Press）中讨论的策略，并概述了帮助团队在工作流与专业技能的集中之间找到平衡、促进协作和自治的独特结构、原则和实践。我们评估了这些结构的优势和限制，并为它们在满足组织需求时的发展提供指导。最后，我们讨论了有意识的领导角色及其在塑造敏捷、响应迅速的机器学习组织中所起支持作用。

# 附加思考

在我们结束前，我们想提及四件事情。

首先，我们要承认机器学习不仅仅是监督学习和大语言模型。我们还可以使用其他优化技术（例如，[强化学习](https://oreil.ly/7PjY6)、[运筹学](https://oreil.ly/ZezrC)、[仿真](https://oreil.ly/UVhfB)）来解决数据密集型（甚至数据稀缺）的问题。此外，机器学习并非银弹，有些问题可以在没有机器学习的情况下解决。尽管我们在整本书的代码示例中选择了一个监督学习问题（贷款违约预测）作为锚定示例，但这些原则和实践在监督学习以外也同样有用。例如，自动化测试、依赖管理和代码编辑器的效率章节即使在强化学习中也很有用。在第二章中概述的产品和交付实践对任何产品或问题领域的探索和交付阶段都有用。

其次，随着生成式人工智能（Generative AI）和大语言模型（LLMs）进入公众意识和许多组织的产品路线图，我和我们的同事们有机会与组织合作，思考、塑造和交付利用生成式人工智能的产品。虽然大语言模型已经引领了如何引导或约束模型朝向其期望功能的范式转变，但精益产品交付和工程的基本原则并未改变。事实上，本书中的基本工具和技术帮助我们早期测试假设，快速迭代并可靠地交付——因此，即使处理生成式人工智能和大语言模型固有的复杂性时，仍能保持敏捷性和可靠性。

第三，关于文化的角色：ML 的有效性和本书中的实践不是——也不能是——一个人的努力。这就是为什么我们将这本书题为《有效的机器学习 *团队*》的原因。例如，你不能是唯一编写测试的人。在我们合作的组织中，只有当团队、部门甚至整个组织在这些精益和敏捷实践上达成文化上的一致时，个人才能最有效。这并不意味着你需要和整个组织一起做所有事情；但独自行动是不够的。正如史蒂夫·乔布斯曾经说过的，“在商业上，伟大的事情从来不是由一个人完成的。它们是由一个团队的人完成的。”

最后，本书不是关于生产率（如何尽可能多地发布特性、故事或代码），也不是关于效率（如何以最快的速度发布特性、故事或代码）。相反，它关注的是有效性——如何快速、可靠和负责任地构建正确的产品。本书探讨的是通过移动找到平衡和以有效方式移动。

本书中的原则和实践始终帮助我们成功交付 ML 解决方案，我们相信它们也会对您有同样的帮助。

# 本书使用的约定

本书使用以下排版约定：

*斜体*

表示新术语、URL、电子邮件地址、文件名和文件扩展名。

`恒定宽度`

用于程序清单，以及段落中引用程序元素，如变量或函数名，数据库，数据类型，环境变量，语句和关键字。

**`恒定宽度粗体`**

用于引起对代码块中感兴趣部分的注意。

###### 注意

此元素表示一般注释。

###### 警告

此元素表示警告或注意事项。

# 使用代码示例

补充材料（代码示例，练习等）可下载：

+   [*https://github.com/davified/loan-default-prediction*](https://github.com/davified/loan-default-prediction)

+   [*https://github.com/davified/ide-productivity*](https://github.com/davified/ide-productivity)

+   [*https://github.com/davified/refactoring-exercise*](https://github.com/davified/refactoring-exercise)

如果您对代码示例有技术问题或使用问题，请发送电子邮件至*support@oreilly.com*。

本书旨在帮助您完成工作。一般情况下，如果本书提供示例代码，您可以在您的程序和文档中使用它。除非您要复制大部分代码，否则无需联系我们以获取许可。例如，编写使用本书多个代码片段的程序不需要许可。销售或分发 O’Reilly 书籍中的示例代码需要许可。引用本书回答问题并引用示例代码不需要许可。将本书大量示例代码整合到您产品的文档中需要许可。

我们感激，但通常不要求署名。署名通常包括标题、作者、出版商和 ISBN。例如：“*有效的机器学习团队* 作者大卫·谭、阿达·梁和大卫·科尔斯（O'Reilly）。版权所有 2024 年大卫·谭瑞冠、阿达·梁咏雯和大卫·科尔斯，978-1-098-14463-0。”

如果您认为使用代码示例超出了合理使用范围或上述授权，请随时联系我们，邮件至 *permissions@oreilly.com*。

# 致谢

当我们开始写这本书时，我们旨在分享一系列帮助我们构建机器学习系统的实用方法。但最终，我们得到了一本全面指南，我们坚信这将提升机器学习团队的共同标准，改变团队塑造和交付机器学习产品的方式。这本书没有这些人的示范、言论和行动，将不可能完成。

我们要感谢 O'Reilly 的出色团队，他们帮助将这本书变成现实：Nicole Butterfield、Melissa Potter、Gregory Hyman、Kristen Brown、Nicole Taché、Judith McConville、David Futato、Karen Montgomery、Kate Dullea，以及其他在幕后不断完善这本书的编辑、设计师和工作人员。特别感谢我们的技术审阅者，他们花时间和精力仔细阅读了超过 300 页的内容，并提供了深思熟虑和坦诚的反馈：Hannes Hapke、Harmeet Kaur Sokhi、Mat Kelcey 和 Vishwesh Ravi Shrimali。

## 来自 David Tan

感谢 Nhung 在我为这本书通宵达旦的日子里如此耐心和支持。没有你的支持，我无法完成这本书。

我看见了什么，Jacob 和 Jonas —— 一棵树！永远保持好奇心。

特别感谢 Jeffrey Lau —— 你的指导和鸭肉面并没有白费。

感谢 Thoughtworks 的过去和现在的同事们，他们教会了我很多关于提出问题之美，并且向我展示了踏上新道路是可以的。我尝试过列出你们所有人的名字，但名单会变得太长。你们知道自己是谁——非常感谢你们坦诚、善良，并且在你们所做的事情上表现出色。特别感谢 Sue Visic、Dave Colls 和 Peter Barnes 在写作这本书过程中对我的鼓励和支持。

Neal Ford：当我联系你询问有关写书的一些后勤问题时，你不仅仅回答了我的问题，还分享了你的写作过程，如何测试想法，并介绍了史蒂芬·金和安妮·迪拉德关于写作的想法。你本可以不这样做，但你做了。感谢你成为一个倍增者。

这几乎是不言而喻的，但是非常感谢我的同谋 Ada 和 Dave。你们将这本书的质量和广度提升到了我无法想象的高度，我很高兴看到这本指南通过我们的集体经验帮助机器学习团队和从业者。

## 来自 Ada Leung

我要感谢我的伴侣、朋友和家人。你们知道你们是谁。你们无尽的鼓励和对我合著一本书的钦佩（是的，我知道，对吧？！）让我想起了与一群非常聪明和令人印象深刻的技术人员一起工作是多么酷。

我还要感谢我在路上遇到的 Thoughtworks 同事，我从远处受到启发，并有幸受到指导——你们对于知识分享的激情和慷慨树立了优秀的榜样。没有比*Ubuntu*哲学更适合描述这个社区了：因为我们在一起，所以我存在。

最后，感谢我的共同作者 David 和 Dave：在这段旅程中，感谢你们始终如一的支持。从分享我们的想法，发现我们集体知识的广度和重叠之处，我意识到了我有多么珍视团队合作和同伴关系。这真是一种喜悦和荣幸。

## 来自 David Colls

我要感谢我的家人，在这几个月里，他们容忍着一个作为丈夫和父亲在周末、电影之夜和篮球场边写作、审阅和研究内容的我。

我要感谢全球许多 Thoughtworks 同事在撰写书籍和创造技术转型视角方面的先驱，激励我们坚定地走同样的道路，并展示给我们什么是优秀的样子。更贴近家门口，我要感谢在澳大利亚与我共事超过十年的所有 Thoughtworks 同事，他们拓宽了我的视野，也在职业上和个人成长中丰富了我。

特别感谢我在 Thoughtworks 澳大利亚数据与 AI 实践中有幸合作过的所有成员，我们共同打造了一些新的东西——这本书里都带有你们每个人的一点。我也要感谢我们的客户，他们信任我们为他们最重要的机遇和挑战开发新的解决方案。

最后，我要感谢我的共同作者 David 和 Ada，感谢他们的专业知识和洞察力，对我的想法提供的反馈，以及在这本书中提炼和分享我们知识的结构化方法。能与你们合作是一种乐事。
