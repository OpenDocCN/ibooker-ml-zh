# 第十四章：实用的 ML 组织实施示例

组织是复杂的实体，它们的各个不同方面都是相互连接的。组织领导者将面临新的挑战和变化，这是由于采用 ML 所带来的。为了在实践中考虑这些问题，让我们看看三种常见的组织采用结构以及它们如何适用于我们一直在考虑的组织设计问题。

对于每种情况，我们将描述组织领导者选择如何将 ML 集成到组织中以及该选择的影响。总体而言，我们将考虑每种选择的优势和可能的缺陷，但特别地，我们将考虑每种选择如何影响过程、奖励和人员方面（来自第十三章中介绍的星型模型）。组织领导者应该能够在这些实施场景中看到足够的细节，以识别自己组织的方面，并能将其映射到他们自己的组织环境和战略中。

# 情景 1：新的集中式 ML 团队

假设 YarnIt 决定通过雇佣一个单一的 ML 专家将 ML 纳入其堆栈中，并开发一个模型来生成购物推荐。试点成功，并由于发布而增加了销售。现在公司需要就如何在这一成功基础上扩展以及如何（以及多少！）投资于 ML 做出一些决定。YarnIt 的 CEO 决定雇佣一个新的副总裁来建立和运行 ML 卓越中心团队，作为组织的一种新的集中能力。

## 背景和组织描述

这种组织选择有显著的优势。团队可以专注，有充分的合作机会和共同工作的机会，并且领导层可以明确地优先考虑公司内 ML 工作。如果他们的范围限于 ML 系统，可靠性专家可以在同一组织中。集中还创建了一个重要的影响中心：ML 组织的领导者在整个 YarnIt 公司中更有权利为他们的优先事项辩护。

随着团队的壮大和项目的多样化，YarnIt 的更多部门将需要与 ML 组织互动。这就是集中化成为劣势的地方。ML 团队不能与业务其余部分距离太远，否则团队将需要更长的时间来发现机会，深入了解原始数据，并构建良好的模型。如果 ML 团队与个别产品团队隔离开来，那么它成功的可能性就不大，如果没有这些产品团队的支持。更糟糕的是，将这两个功能（ML 和产品开发）完全分开放在组织结构图中可能会促使团队竞争而不是合作。

最后，集中化的组织可能对请求帮助将机器学习整合到其产品中的业务部门的需求没有实质性的响应。当涉及到机器学习的产品化时，业务部门可能不了解机器学习团队的可靠性需求，也不理解为什么要遵循可靠性流程（从而导致交付速度变慢）。

尽管集中化机器学习团队存在这些缺陷，但组织始终可以进化。该场景经历了星形模型，好像它只使用了一个集中化团队，但我们还将在第三种情况下说明一个集中化团队在基础设施中的演变。另一种可能的演变是，集中化团队教育和使其他人增加组织内的机器学习素养。

## 过程

正如我们所提到的，引入机器学习在组织中的影响往往是广泛的。为了解决集中化组织的一些缺点，引入流程可以帮助分散（或分权）决策或知识。这些流程包括以下内容：

主要利益相关者的审查

机器学习团队定期提交的这些审查应确保当前建模结果的批准，以及业务领导对系统适应业务方式的理解。确切地确定这些审查中应包括哪些指标的科学是必要的，但这些指标应是全面的，需要包括任何给定模型实施的改进及其成本。关键的业务利益相关者还可以审查各种机器学习团队工作的优先级以及投资回报率和使用案例。

对变更的独立评估

集中化机器学习团队可能面临的一个问题是，所有变更都变得彼此依赖，并可能被其他变更所阻碍。如果团队改为独立评估变更，确保每个模型在改变其自身的生产中的准确性，那么变更可以更快地可用。通常，一个模型可能会在平均性能上改进，但可能会损害特定子组的性能。判断这些权衡是否值得往往需要进行重要的分析和挖掘，并可能需要基于不易通过简单指标如预测准确性反映的业务目标的判断。

降低变更组合测试的风险

在大型模型开发团队中，通常会并行开发多个改进措施，然后可能一同推出。问题是这些更新是否能良好地配合。这就需要同时测试候选变更的组合效果，而不仅仅是单独地测试。¹ 重要的是建立一个流程来审查候选变更，并确定哪些组合测试是有用的。这可能会通过进行走/不走的会议来讨论模型推出及团队如何进行组合测试来进一步促成测试或决策是否推出。

虽然在集中式机器学习团队中更容易实现，但是引入流程使得来自各业务单位的人员也能够评估变更。对于集中式机器学习团队而言，可能是请求变更或特性的产品/业务团队，或者可能会受到变更影响的支持团队。

## 奖励

为了确保机器学习项目的成功实施，我们需要*奖励业务与模型构建者之间的互动*。员工需要根据他们跨组织合作的有效性和对业务结果的对齐进行评估，而不仅仅是评估他们所在部门、部门或团队完成狭窄任务的有效性。寻求其他组织的意见和正式审查、提供反馈以及沟通计划，这些行为都应该得到奖励。如何奖励这些行为在文化上依赖于每个组织，但它们应该得到认可。

奖励可以是金钱（奖金、休假）但也可以包括晋升和职业机会。在 YarnIt，我们可能会添加“有效影响者”等评价特征作为我们寻找成功员工的标准之一。

## 人员

产品领导者需要*实验心态*，既要欣赏机器学习所创造的价值，也要容忍一些风险。产品领导者需要理解，调整机器学习模型以符合组织目标可能需要一些实验，并且在这个过程中几乎肯定会出现负面影响。

YarnIt 及所有实施机器学习的组织，需要聘请具有*细微差别*心态的人才以确保成功。领导者需要容忍复杂性，并且在超出自身职权范围的组织边界内工作时感到舒适。特别是，领导者还必须有信心向上级层传达这些影响的复杂性，而不是过于简化或粉饰。YarnIt 的 CEO 不需要听到机器学习是魔法并将解决所有问题，而是需要了解如何利用机器学习作为工具实现业务目标。目标不是做机器学习，而是推动业务变革。虽然机器学习很强大，但细微之处在于通过最小化负面影响来创造价值。

如果这些中心团队的人员在这些领域没有专业知识，他们需要接受有关质量、公平、道德和隐私问题的培训。

## 默认实现

一个过于简化的集中模型的默认实现如下：

+   雇佣一位具有机器学习建模和生产技能的新领导。

    +   雇佣机器学习工程人员来构建模型。

    +   雇佣软件工程人员来建立机器学习基础设施。

    +   雇佣机器学习生产工程人员来运行基础设施。

+   与产品团队制定实施计划（数据来源和新模型的集成点）。

+   设立整个计划的常规执行审查。

+   根据成功的实施计划进行补偿，并对机器学习人员和产品领域人员进行补偿。

+   启动隐私、质量、公平和道德计划，建立标准和对这些标准的遵守监控。

# 情景 2：分散的机器学习基础设施和专业知识

YarnIt 可能决定在整个组织中投资几位专家，而不是一个单一的高级领导者。每个部门将必须自行雇佣其自己的数据科学家，包括购物推荐和库存管理团队。基本上，YarnIt 将允许数据科学和机器学习的简单实现出现在任何有足够需求和愿意支付的部门。

## 背景和组织描述

这种方法更快，或者至少更容易开始。每个团队可以根据自己的优先事项雇佣和安排项目人员。随着机器学习专家的雇佣，他们将更接近业务和产品，因此能更好地理解每个组的需求、目标甚至政治因素。

存在风险。缺乏集中的机器学习专业知识，尤其是在管理方面，将使深入理解 YarnIt 需要在机器学习成功方面做的更多工作变得更加困难。管理层将不了解需要什么专门工具和基础设施。很可能会倾向于尝试用现有工具解决机器学习问题。很难理解当机器学习团队主张真正需要的东西（如特定于模型的质量跟踪工具如 TensorBoard），与可能很好但可能并非必需的东西（某些模型类型和大小的 GPU 或提供大规模但成本也很高的云训练服务）之间的区别。此外，每个团队将会重复执行一些相同的工作：创建一个稳健且易于使用的服务系统，可以跨多个模型共享资源，并监控系统以跟踪模型的训练进度并确保完成训练。如果可以避免，所有这些重复可能都是昂贵的。

如果这些团队中有些团队在处理涉及其他产品的工作，而他们很可能会这样做，故障排除和调试就会变得更加困难。当产品或生产问题出现时，YarnIt 需要找出哪个团队的模型出了问题，或者更糟的情况是，让多个团队一起调试彼此模型之间的交互。大量的仪表盘和监控会使这一切变得指数级更加困难。对于任何给定模型变化的影响不确定性将会增加。

最后，YarnIt 将努力确保其采用一致的方法。在 ML 公平性、道德和隐私方面，只要一个糟糕的模型就可能损害他们的用户，并且在公众中损害我们的声誉。YarnIt 还可能会通过这种组织结构复制认证、集成到 IT、日志记录和其他 DevOps 任务。

虽然存在真正的权衡，但对于许多组织来说，这种分散的方法确实是正确的选择。它可以减少启动成本，同时确保组织立即从 ML 中获得有针对性的价值。

## 过程

为了使这种结构有效，组织应专注于能够引入一致性而不会引入过多开销的流程。这些流程包括以下几点：

高级利益相关者的评审

模型开发者仍应该参与高级利益相关者的评审。对于模型开发者来说，创建每个提议的模型开发目标及其发现的写作是一种非常有用的做法。这些内部报告或白皮书，尽管篇幅较短，却能比较详细地记录尝试过的想法以及组织从中学到的东西。随着时间的推移，这些会形成组织的记忆，并强化评估的严谨性，类似于软件工程师代码评审中的严谨性。YarnIt 应该为这些报告创建一个模板，可能由一些最早开始使用 ML 的团队共同合作生成，并为参与人员超出仅仅实施 ML 的组织的小组定期评审制定标准时间表。

分类或 ML 生产会议

ML 模型开发者应每周与生产工程人员及产品开发组的利益相关者会面，以审查 ML 部署的任何变化或意外效果。像所有事情一样，这可以做得好也可以做得不好。一个糟糕的会议版本可能没有所有相关的观点，可能基于偶发性问题而不是被深入理解的系统性问题，可能深入解决问题解决得太多，或者简单地可能持续时间过长。良好的生产会议应该是简短的，专注于分类和优先级确定，分配问题的所有权，并审查过去的任务更新和进展。

技术基础设施的最低标准

YarnIt 应建立这些最低标准，以确保所有模型在投入生产之前都通过了某些测试。这些测试应包括基线测试，例如“模型能否提供单个查询的服务？”以及涉及模型质量的更复杂的测试。即使是像标准化的 URL 这样的简单变化也可以帮助推动内部的一致性（在机器学习快速变化且复杂的世界中，任何有助于记住和行为一致的东西都是有用的）。

## 奖励

为了平衡这种方法的分散效果，YarnIt 高级管理层需要*奖励一致性和公布的质量标准*。例如，领导应奖励及时的撰写和在广泛可用的语料库中进行仔细审查的员工。每个团队都有其本地优先事项，因此有必要奖励平衡增加速度与一致性、技术严谨性和沟通的行为。

在这种情况下需要注意的一个具体因素是，YarnIt 不太可能拥有具有显著机器学习经验的员工。一个有用的奖励是鼓励员工参加（并在）与他们工作相关的会议上发表演讲。

## 人员

在这种部署中，YarnIt 应寻找那些能够*既能本地思考又能全局思考*的人才，平衡本地利益与公司可能的不利因素（或反之）。影响力和跨组织线合作等技能可能是需要明确表达和奖励的。

组织仍然需要关心质量、公平性、道德和隐私问题的人员，并能够影响组织——这在每种部署方案中都是如此。不同之处在于，在这种情况下，这些员工将不得不开发本地实施方案，以实现质量、公平性、道德和隐私，同时还要制定广泛的标准并推动它们在公司内的实施。

## 默认实施

这里是分散结构方案的一个过度简化的默认实施：

+   每个团队在其业务单元内聘请专家：

    +   雇佣机器学习工程人员直接与产品团队一起构建模型。

    +   雇佣或转移软件工程人员来构建机器学习基础设施。

    +   雇佣机器学习人员或转移生产工程人员来运行基础设施。

+   发展内部发现报告的实践，供高级利益相关者审查。

+   建立公司范围内的技术基础设施标准。

+   每周进行分类或机器学习生产会议以审查变更。

+   启动一个关于隐私、质量、公平性和道德的计划，为这些标准建立标准和合规监控。

# 情景 3：混合中心化基础设施/分散建模

YarnIt 开始通过集中模型实施，但随着组织的成熟和机器学习的普及，公司决定重新审视该模型并考虑采用混合结构。在这种情况下，组织将保持一些集中的基础设施团队和一些中央组织的机器学习模型咨询团队，但各个业务单元也可以自由聘请和培养他们自己的机器学习建模专家。

这些分布式机器学习工作人员可能起初会严重依赖中央建模顾问，但随着时间的推移，他们会变得更加独立。然而，所有团队都应该使用并为中央的机器学习生产实施做出贡献。

通过集中投资和基础设施的使用，YarnIt 将继续从效率和一致性中受益。但至少在一定程度上分散机器学习专业知识将增加采用速度，并改善机器学习模型与业务需求之间的对齐。

注意，许多组织都会演变成这种混合模型。作为领导者，计划这种演变可能是明智的选择。

## 背景和组织描述

这种混合实施的缺点源于集中和分散的机器学习组织结构。在分散人员配置和在整个组织中实施机器学习方面可能存在效率低下的问题。业务单元可能对机器学习了解不足，并可能存在特别糟糕的实施。如果失败与隐私或道德有关，这尤其成问题。同时，集中的基础设施可能会对分散建模团队产生摩擦。而且，集中的基础设施可能感觉更复杂和更昂贵。然而，公司存在的时间越长，集中的基础设施模型带来的回报就越多。

## 过程

有一种思考这种混合实施影响的方法是重新考虑在 “大不一定是好的” 中的购物车放弃示例。在这种情况下，如果建模团队居住在网店产品团队中，这些团队成员更有可能快速注意到问题，并重新调整模型的指标，而不仅仅是最大化购物车的大小。他们还会考虑可能的 *新功能*，比如“现在购买一半的毛线，一个月后保留另一半”。在所有这些情况下，机器学习已经引发了关于如何更加顾客友好的讨论。

但假设问题发生在组织部门之间：网店模型给购买带来问题。在这些情况下，解决问题可能会慢得多，因为采购团队试图说服网站团队模型造成了问题。在这些情况下，组织文化必须支持跨团队模型故障排除甚至开发。

考虑推荐给场景 1 和 2 的流程，看看它们是否可以帮助减轻这种实施可能的缺点：

+   对变更组合的无风险测试

+   对变更的独立评估

+   评审：

    +   机器学习团队的发现文档和审查

    +   模型结果的业务评审

    +   ML 生产会议的分类

+   技术基础设施的最低标准

+   质量、公平性、伦理和隐私问题的培训或团队

## 奖励

在混合场景中，YarnIt 的高级管理层应该奖励那些利用集中基础设施的业务单元，以防止它们开发自己的重复基础设施。集中基础设施团队应该因满足其他业务单元的需求而获得奖励。在几乎所有情况下，衡量和奖励采用，同时鼓励使用中央基础设施，都是有意义的。

中央基础设施团队应该有计划识别业务单元中开发的关键技术，并将其扩展到公司的其他部门。从职业发展的角度来看，业务单元的 ML 建模师应该能够在一段时间内轮换到中央基础设施团队，以了解可用的服务及其限制，并为这些团队提供最终用户的视角。

## 人员

要在整个 YarnIt 公司有效运作，所有这些团队都需要有全公司视角的工作。基础设施团队需要建立能够真正有用和受其他公司部门欢迎的基础设施。与业务部门紧密结合的 ML 团队需要有合作是最好的心态，因此他们应该寻找跨部门合作的机会。

## 默认实现

这是集中基础设施/分散建模模型的一个过度简化的默认实现：

+   雇佣一个具备 ML 基础设施和生产技能的集中团队（领导者）：

    +   雇佣软件工程人员建立 ML 基础设施。

    +   雇佣 ML 生产工程人员来运行基础设施。

+   每个产品团队都会雇佣他们业务单元的专家：

    +   雇佣 ML 工程人员直接与产品团队建立模型。

+   开发一种内部发现报告的实践，供高级利益相关者审查。

+   建立全公司技术基础设施标准。

+   根据成功实施计划补偿，并且不仅补偿达到业务目标，还要衡量利用中央基础设施的效率。

+   选择那些有助于跨组织协作的流程，如跨团队 ML 发现审查。

+   启动隐私、质量、公平和伦理计划，建立标准并监督这些标准的合规性。

# 结论

首次将机器学习技术引入组织是困难的，最佳路径必然因组织而异。成功将需要提前考虑确保成功所需的组织变化，包括诚实面对缺失的技能和角色、流程变化，甚至整个缺失的子组织。有时可以通过招聘或提拔合适的新高级领导者并委以实施任务来解决这些问题。但通常所需的组织变化将涵盖整个公司。

表格 14-1 总结了各种组织结构，以及它们在人员、流程和奖励方面的影响和要求。

一些团队，如生产工程或软件工程团队，开始并不需要大量的机器学习技能来变得有效。但他们将受益于学习小组、参加会议及其他专业发展活动。我们还需要在业务领导者中建立机器学习技能。确定一些能理解将机器学习加入基础设施的好处和复杂性的关键领导者。

然而，成功的最大障碍通常是组织领导层对风险、变革和细节的容忍度，以及坚持机器学习可能需要一段时间才能显现回报的态度。为了推动机器学习进展，我们必须愿意承担风险并改变我们的做法。确保团队专注于业务结果将有助于推广。这涉及改变工作良好的流程并改变成功团队和领导者的行为。这通常也意味着冒险损失成功的业务线。为了理解正在承担的风险，领导者必须关心实施的一些细节。领导者需要容忍风险，但明确关心其机器学习实施的最终可靠性。

表格 14-1\. 各种结构和要求的总结

|  | 中心化的机器学习基础设施和专业知识 | 分散化的机器学习基础设施和专业知识 | 中心化基础设施与分散化建模的混合体 |
| --- | --- | --- | --- |

| **人员** | 具有明确专注的专业团队，对机器学习优先事项和投资具有影响力的核心。

需要有实验心态。

领导者/高级成员需要成为其组织外有效的合作者和影响者。

团队需要成为公司内跨团队的机器学习质量、公平性、道德和隐私的倡导者。| 机器学习专业知识分布在各个团队中，通常既重复又稀缺。

领导者/高级成员需要鼓励和执行内部社区。

孤立的决策将导致不良/不一致的客户体验，从而产生显著的业务影响。

所有产品领域的团队需要在机器学习质量、公平性、伦理和隐私方面获得专业知识。 | 集中机器学习基础设施和建模，用于通用/核心业务用例，但鼓励根据具体需求开发个体模型。

避免重复，提高团队效率和一致性，尤其是在大规模情况下。

机器学习质量、公平性、伦理和隐私需要在所有部门的基因中。 |

| **流程** | 需要进行跨职能的协作来做出决策和分享知识。

业务单元的关键利益相关者需要共同审查提案和结果，以及启动计划。

需要去中心化/独立的模型评估，以确保和衡量业务目标和影响。

确保对组合变更进行验证，以避免意外退步，并建立进行/不进行审查会议。 | 需要大量关于最佳实践、知识、评估和启动标准的文档，以保持一致性，*或者*在本地团队范围之外决定不维护任何文档（这对机器学习来说是有问题的）。

业务单元的关键利益相关者需要共同审查提案和结果，以及启动计划。

需要良好结构化和适度主持的进行/不进行会议，以避免延迟。

为机器学习流水线的技术基础设施建立标准。 | 需要跨职能协作和基础设施与个体产品团队在项目/程序基础上的适度文档化。

建立具有明确责任的跨职能团队。

需要在项目/程序级别定期进行跨职能同步。

业务单元的关键利益相关者需要共同审查提案和结果，以及启动计划。

| **奖励** | 除了总体质量和达到业务目标外，个人/团队绩效还应根据跨职能协作的有效性进行评估。

建立机制，共同补偿成功的 AI 功能发布的机器学习和产品团队。 | 除了总体质量和达到业务目标外，个人/团队绩效还应根据一致性、已发布的质量标准以及运营内部机器学习社区进行评估。 | 除了总体质量和达到业务目标外，个人/团队绩效还应根据可重复使用性、共同基础设施演化以及执行速度进行评估。

¹ 在大多数文献中，这被描述为算法或技术问题。当然，它确实是这些问题，但也很大程度上是组织问题。如果我们没有决策和管理框架来分开评估变更并制定部署策略，我们将无法正确优先考虑这项工作。
