# 前言

人工智能涵盖了各种技术和方法，您将在智能手机到工厂生产线中无处不在地找到它。

随着人工智能的进步，这些技术变得更加强大，实施起来也更加复杂。越来越多的最强大的人工智能系统是由非常大型的深度学习模型驱动的，这些模型在大量数据上训练，使用数十亿个参数，然后定制以解决特定问题。

构建和训练这些非常大型模型需要大公司的资源和大量技术专业知识，并且需要大规模基础设施投资来运行它们。OpenAI 开发的 GPT-3 语言生成模型的初始训练成本约为 400 万美元或更多。仅 GPT-3 所使用的 45 TB 数据的初始训练可能需要一个月的连续训练和超过一千个高端图形处理单元（GPU）卡。

这意味着世界上只有少数几个组织能够创建和运行这些非常大的模型，斯坦福人类中心人工智能研究所将其称为基础模型，因为它们对当前人工智能的开发和使用方式非常重要，并且您可以在其基础上构建新的系统。¹

这些包括所谓的大型语言模型，如 GPT-3，以及许多领域中依赖半监督深度学习、自监督预训练、迁移学习和类似方法创建的极其大型机器学习模型，这些模型使用强大硬件训练在庞大数据集上，然后可以应用于各种问题或定制训练以解决特定问题。

即使是具备构建深度学习系统专业知识的企业，可能发现模型训练和生产运行成本过高，尤其是当它们需要更多工作来减少在开放网络数据集中训练模型时的偏差，以负责任的方式实施它们。但是，任何人都可以利用非常大型基础模型，甚至通过使用 Microsoft 在云中提供的 Azure AI 服务根据自己的需求定制它们。

其他技术，如强化学习，刚刚从研究实验室中出现，并且需要显著的专业知识来实施。

利用 Azure AI 服务，您可以依赖公共云带来的开发、训练和部署规模，花费时间来构建一个解决用户问题的应用程序或工作流程。

# 本书适合谁

Azure 提供的云 AI 服务为任何开发人员、甚至是业务用户以及数据科学家和数据工程师带来了广泛的最新发展。

有这么多技术和许多您可能想要使用 AI 的地方，即使是正在招聘和培训数据工程师和数据科学家的组织也希望通过使用 AI 云服务使他们更加高效。云服务还为那些永远不会成为数据工程师或数据科学家的开发人员和业务专家解锁了 AI 的可能性。

微软提供了许多在 Azure 上运行的不同 AI 工具和服务，我们无法深入涵盖它们所有，因此我们选择了四个关键领域。Power 平台帮助业务用户和专业开发人员构建利用 AI 的应用程序和工作流程。Azure 认知服务和应用 AI 服务为开发人员提供了 API、软件开发工具包（SDK）和现成解决方案，可以集成到他们的代码中。Azure 机器学习强大到足以供数据科学家使用，但也有帮助非 AI 专家培训自己模型以解决特定业务问题的选项。

AI 和云服务的世界都在快速发展，因此，本书所能捕捉到的是它们背后的原则，可以使您成功的最佳实践以及您可以实现的示例。特别是，我们将探讨如何负责任地使用 AI：这对许多开发人员来说越来越重要的问题。

我们涵盖了使用一些关键特性的实际操作方面，但请记住，这些服务可能在您使用它们时已经更新。我们描述的步骤可能与您在云服务中看到的略有不同，性能可能已经得到改进，并且您可能会发现额外的功能，让您能够实现更多！

# 如何使用本书

在本书中，我们希望帮助您开始使用云 AI 服务，无论您的背景或熟悉 Azure 的程度如何。如果您不熟悉强大的 AI 技术和机器学习模型已经发展到何种程度，第一章将介绍当前的技术水平，包括 Azure AI 建立在其上的关键研究里程碑。

要了解作为 Microsoft AI 平台一部分提供的全部 AI 工具和服务范围，请转到第二章，我们将探讨您可以做出的许多不同选择，包括在需要在边缘运行时将云 AI 带入您自己的基础设施。

如果您有构建想法以及 Azure AI 服务如何帮助您的想法，您可以直接转到第二部分。

如果您是一名经验丰富的开发者，并准备立即开始构建机器学习模型，请从第三章开始，我们将介绍 Azure 机器学习，这是微软在云端提供的综合机器学习服务，您可以使用像 PyTorch 和 TensorFlow 这样的行业标准框架进行工作。

如果您宁愿调用 API 来利用可以针对常见 AI 任务或整个场景进行微调的预构建模型，并使用多种熟悉的编程语言和您喜爱的开发者工具，请直接进入第四章和第五章，在这里我们涵盖了 Azure 认知服务和 Azure 应用 AI 服务。

但您并不需要是开发人员才能使用 Azure AI。许多 Azure 认知服务可在低代码 Power 平台和无代码 AI Builder 工具中使用。在第六章中，我们将带您了解如何使用机器学习来理解数据和解决问题。

如果您希望退后一步，思考如何使用 AI，您的数据将从哪里获取，以及如何负责地完成所有这些工作，请转至第七章和第八章，在这里我们将探讨道德方法和最佳实践，这将帮助您充分利用云 AI 的潜力。

想确保云 AI 确实能够扩展到处理您的问题，无论其大小和复杂程度？想过是否更合理地运行自己的 AI 基础设施吗？在第九章中，我们深入探讨了 Azure 认知服务的幕后情况，看看在全球范围内运行 24/7 API 平台意味着什么，以及如何在生产中使用数十个机器学习模型，并始终保持其更新。

第 III 部分继续探讨其他组织如何使用 Azure AI 服务描述世界给盲人用户，即使由于大流行导致购买习惯一夜之间改变，也能精确选择推荐的正确产品，并在多语言之间实时翻译语音。在第十章、11 章和 12 章中，我们有一些使用 AI 导向架构构建的实际系统案例研究，集成了不同的 Azure AI 服务，向您展示了可能性。

但使用 Azure AI，真正的限制在于您的想象力——以及您可以为问题带来的数据。继续阅读，了解您可以做什么以及如何入门。

# 本书使用的约定

本书使用以下排印约定：

*斜体*

指示了新术语、网址、电子邮件地址、文件名和文件扩展名。

`等宽字体`

用于程序列表，以及在段落中引用程序元素，例如变量或函数名称、数据库、数据类型、环境变量、语句和关键字。

`**等宽粗体**`

显示用户应直接键入的命令或其他文本。

`*等宽斜体*`

显示应替换为用户提供值或由上下文确定的值的文本。

###### 提示

此元素表示提示或建议。

###### 注意

此元素表示一般提示。

###### 警告

此元素指示警告或注意事项。

# 使用代码示例

您可以从[*https://github.com/Azure-Samples/Azure-AI-Services-O-reilly-book-Companion-repo*](https://github.com/Azure-Samples/Azure-AI-Services-O-reilly-book-Companion-repo)下载本书的补充材料，如代码示例。

本书旨在帮助您充分利用 Azure AI 服务。一般而言，如果本书提供示例代码，您可以在您的程序和文档中使用它。除非您复制了代码的大部分内容，否则无需征得我们的许可。例如，编写一个使用本书多个代码片段的程序不需要许可。销售或分发包含来自 O’Reilly 书籍示例的 CD-ROM 需要许可。引用本书回答问题并引用示例代码不需要许可。将本书大量示例代码整合到产品文档中需要许可。

我们赞赏但不要求署名。一般的署名通常包括标题、作者、出版商和 ISBN。例如：“*Azure AI Services at Scale for Cloud, Mobile, and Edge* by Simon Bisson, Mary Branscombe, Chris Hoder, and Anand Raman (O’Reilly). Copyright 2022 O’Reilly Media, Inc., 978-1-098-10804-5.”

如果您觉得您使用的代码示例超出了合理使用范围或上述许可，请随时通过*permissions@oreilly.com*与我们联系。

# O’Reilly 在线学习

###### 注意

40 多年来，[*O’Reilly Media*](https://oreilly.com)已为公司提供技术和商业培训、知识和见解，以帮助其取得成功。

我们独特的专家和创新者网络通过书籍、文章和我们的在线学习平台分享他们的知识和专长。O’Reilly 的在线学习平台为您提供按需访问的实时培训课程、深入学习路径、交互式编码环境以及来自 O’Reilly 和 200 多家其他出版商的大量文本和视频。更多信息，请访问[*https://oreilly.com*](https://oreilly.com)。

# 如何联系我们

请将关于本书的评论和问题发送至出版商：

+   O’Reilly Media, Inc.

+   加利福尼亚州格拉文斯坦大道北 1005 号

+   加利福尼亚州塞巴斯托波尔市，邮政编码 95472

+   800-998-9938（美国或加拿大境内）

+   707-829-0515（国际或当地）

+   707-829-0104（传真）

我们为这本书制作了一个网页，其中列出了勘误、示例和任何额外信息。您可以访问此页面：[*https://oreil.ly/azure-ai*](https://oreil.ly/azure-ai)。

通过电子邮件*bookquestions@oreilly.com*发表评论或提出关于本书的技术问题。

欲了解有关我们的书籍和课程的新闻和信息，请访问*[`oreilly.com`](https://oreilly.com)*。

在 LinkedIn 上找到我们：*[`linkedin.com/company/oreilly-media`](https://linkedin.com/company/oreilly-media)*

在 Twitter 上关注我们：*[`twitter.com/oreillymedia`](https://twitter.com/oreillymedia)*

观看我们的 YouTube 频道：*[`youtube.com/oreillymedia`](https://youtube.com/oreillymedia)*

# 致谢

作者可能会写下这些文字，但要创作一本书，需要更多的工作，而这本书的完成离不开许多帮助过我们的人们。

尽管所有错误都是我们自己的，但对于特定章节的帮助，我们感激不尽：

+   第三章：普拉尚特·凯特卡尔

+   第六章：阿米尔·内茨、朱斯蒂娜·鲁克尼克、安托万·塞莱利耶、乔·费尔南德斯

+   第七章：萨利玛·阿默希、莎拉·伯德、梅尔努什·萨米基、米哈埃拉·沃尔沃雷亚努

+   第九章：格雷格·克拉克

+   第十章：萨基卜·沙克

+   第十一章：伊沃·拉莫斯、奥利维尔·纳诺

+   第十二章：杰夫·门登霍尔

这本书还受益于 O’Reilly 团队的工作：加里·奥布莱恩、乔纳森·哈塞尔、丽贝卡·诺瓦克、贝丝·凯利、凯特·达莱亚和莎伦·特里普。

### 出自玛丽·布兰斯康姆和西蒙·比松

安德鲁·布雷克、道格·伯格、莉莉·程、卡特雅·霍夫曼、埃里克·霍维茨、查尔斯·拉曼纳、彼得·李、约翰·兰福德、詹姆斯·菲利普斯、马克·鲁辛诺维奇、达尔马·舒克拉、帕特里斯·西马尔德、杰弗里·斯诺弗、约翰·温和许多其他人多年来帮助我们理解机器学习技术和微软服务。还要感谢詹姆斯·多兰，埃塞克斯大学荣休教授和玛丽的硕士论文导师，是她第一次听到“机器学习”这个术语的人。

### 出自克里斯·霍德尔

所有在认知服务和合作伙伴团队中为这些服务和这本书做出贡献并帮助我理解机器学习、应用人工智能以及许多其他宝贵知识的令人惊叹的人们。还要感谢劳伦·霍德尔对我的爱和支持。

### 出自阿南德·拉曼

我的父母和姐妹教会了我今天的这个人所拥有的一切价值观。我的妻子阿努帕玛和孩子阿什雷和阿哈娜每天都给予我灵感、爱和支持。感谢微软的所有同事，在这条路上帮助我理解人工智能/机器学习以及许多其他宝贵的东西。

1 详见[*关于基础模型的机遇与风险*](https://arxiv.org/pdf/2108.07258.pdf)，对这种日益普遍的方法的优势和可能的危险进行了广泛而发人深省的分析。