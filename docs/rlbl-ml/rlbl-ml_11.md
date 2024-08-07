# 第十章：连续机器学习

到目前为止，我们对机器学习系统的讨论有时集中在这样一个观点上，即模型是我们训练然后部署的东西，几乎像这是一次性的事情。稍微深入一点的看法是要区分那些训练一次然后部署的模型与那些以更连续的方式训练的模型，我们将其称为*连续机器学习系统*。典型的连续机器学习系统以流式或周期性批处理的方式接收新数据，并使用这些数据触发训练更新的模型版本，然后将其推送到服务环境中。

显然，从 MLOps 的角度来看，训练一次与连续更新模型之间存在重大差异。转向连续机器学习增加了自动验证的风险。它引入了围绕反馈环路和模型对外部世界变化的反应可能导致的头痛。管理连续数据流、响应模型失败和数据损坏，甚至看似琐碎的任务，如引入新特性以供模型训练，都增加了系统的复杂性。

事实上，从表面上看，创建一个连续机器学习系统可能是个糟糕的主意。毕竟，在这样做的过程中，我们将系统暴露给一组因为真实外部世界潜在变化而无法预先知晓的变化，从而可能导致意外或不希望的系统行为。如果我们记得在机器学习中，数据就是代码，那么连续机器学习的理念就是接受相当于不断涌入的新代码，这些代码可以改变我们生产系统的行为。唯一做出这种改变的理由是当不断更新的系统的好处超过成本时。在许多情况下，好处确实是相当可观的，因为拥有一个能够学习并适应世界新趋势的系统可以实现复杂的业务和产品目标。

本章的目标是审视成本可能累积和问题可能出现的领域，以期在保留好处的同时尽可能降低连续机器学习系统的总体成本。这些观察包括以下几点：

+   外部世界事件可能影响我们的系统。

+   模型可能通过反馈环路影响其未来的训练数据。

+   时间效应可以出现在几个时间尺度上。

+   危机响应必须实时进行。

+   新的启动需要分阶段的逐步推进和稳定的基线。

+   模型必须进行管理，而不是仅仅交付。

这些观点每一个都总结了一系列潜在的复杂性，我们将在本章的大部分内容中深入探讨。当然，技术挑战并不是故事的终点。除了连续机器学习在我们的机器学习开发和部署过程中引入的实际和技术挑战之外，它还创造了组织上的机会和复杂性。

###### 小贴士

持续改进的模型需要能够管理这种持续改进的组织。

我们需要框架来生成和追踪对模型改进的想法。我们需要一种评估我们模型各个版本在长时间内表现的方法，而不是仅仅关注我们推出一个模型以替代另一个模型的时间点。我们需要将建模视为一个长期持续的、产生价值的程序，它具有成本和风险，但也有巨大的潜在好处，并在本章末讨论这些需求对组织的影响。

# 持续 ML 系统的解剖学

在我们详细探讨持续 ML 系统的影响之前，让我们花一些时间浏览典型的 ML 生产堆栈，并看看与非持续设置相比，事物如何变化。在高层次上，持续 ML 系统定期从世界中获取数据流，用于更新模型，然后经过适当验证，推出更新版本的模型以服务新数据。

## 训练示例

在持续 ML 系统中，训练数据不再是一组固定的不可变数据，而是以稳定的数据流形式存在。这可能包括来自一组可能的纱线产品的推荐产品集合，以及生成这些推荐的查询。在高容量应用中，训练示例的数据流可能类似于消防栓，每秒从全球各地收集大量数据。需要进行重要的数据工程工作，以确保这些数据流被有效和可靠地处理。

## 训练标签

我们示例的训练标签也是以数据流形式来自世界。有趣的是，这个数据流的来源很可能与训练示例本身的数据流不同。例如，假设我们希望使用用户是否购买了某种纱线产品作为训练标签。我们知道在查询时用户展示了哪些产品，并且可以记录它们发送给用户的情况。然而，购买行为不能在查询时知晓——我们必须等待一段时间来看他们是否选择购买——这些信息可能来自完全不同于整体服务基础设施的购买处理系统。在其他情况下，当这些标签由人类专家提供时，我们可能会看到训练标签的延迟。

将示例与其正确标签结合起来需要不可避免的延迟，并且可能涉及一些相对复杂的基础设施，以便高效可靠地进行处理。实际上，这种结合是生产关键的。试想一下，如果由于故障而导致标签信息不可用，并且未标记的示例被发送到模型进行训练，会带来多大的麻烦¹。

## 过滤坏数据

每当我们允许我们的模型直接从世界上的行为中学习时，我们就面临着这样的风险：世界会发送一些我们希望我们的模型不必从中学习的行为。例如，垃圾邮件发送者或欺诈者可能会试图通过发出许多虚假查询而不购买，试图使我们的羊毛产品预测模型看起来比实际上受用户欢迎度低。或者一些不良行为者可能会试图通过反复输入侮辱性文本到聊天窗口来使我们有用的*yarnit.ai*聊天机器人学习粗鲁的行为。必须检测并处理这些攻击。不那么恶意但同样具有破坏力的坏数据形式可能由管道中断或错误引起。在所有情况下，重要的是在训练之前从管道中去除这些坏数据形式，以确保模型训练不受影响。

有效地删除垃圾或损坏数据是一项困难的任务。这需要自动化异常检测方法以及主要目的是检测坏数据的模型。通常会出现一种类似于坏行为者试图不适当影响模型和运维团队试图检测这些尝试并过滤它们的竞争。有效的组织通常专门设有全职团队来解决从连续 ML 管道中过滤坏数据的问题。

## 特征存储和数据管理

在典型的生产 ML 系统中，原始数据被转换为特征，除了对学习有用外，还更紧凑以便存储。许多生产系统使用*特征存储*来以这种方式存储数据，它本质上是一个增强型数据库，管理输入流，知道如何将原始数据转换为特征，高效存储它们，允许在项目之间共享，并支持模型训练和服务。² 对于高容量应用程序，通常需要从整体数据流中进行一定数量的抽样，以减少存储和处理成本。在许多情况下，这种抽样不会是均匀的。例如，我们可能希望保留所有（罕见的）正例，但仅保留（非常普遍的）负例的一部分，这意味着我们需要跟踪这些抽样偏差，并将其与适当的加权合并到训练中。

尽管特征化数据更紧凑且通常更有用，但几乎总是需要保留一定数量的日志原始数据。这对于开发新特征以及测试和验证特征提取和转换代码路径的正确性非常重要。

## 更新模型

在持续的机器学习系统中，通常更倾向于使用允许增量更新的训练方法。基于随机梯度下降（SGD）的训练方法可以在持续环境中无需任何修改地使用。（回想一下，SGD 构成了大多数深度学习训练平台的基础。）要在持续环境中使用 SGD，我们基本上只需闭上眼睛假装模型展示的数据流是随机顺序的。如果我们的数据流实际上是相对混乱的顺序，这完全没问题。

实际上，数据流通常具有基于时间的相关性，这并不真的是随机的。因此，我们需要担心这种假设破坏我们实际应用中的影响。如果我们的数据绝对最坏的非随机方式是，我们有一个上游的批处理作业，例如，所有的正例在一个批次中出现，然后所有的负例在另一个批次中出现。在这种数据上，SGD 方法会遭遇到严重问题，我们需要对数据进行中间混洗，以帮助使 SGD 在更安全、更随机的基础上进行。

一些流水线强制执行严格的策略，只对给定示例进行一次训练，按照它在数据流中时间上的顺序。这种策略从基础设施、模型分析和可重复性的角度来看，简化了许多事情，在数据充足时并没有真正的缺点。然而，在数据匮乏的应用程序中，可能需要多次访问单个示例才能收敛到一个好的模型，因此不总是能够按顺序仅访问每个示例一次。

## 推送更新的模型到服务端

在大多数持续的机器学习环境中，我们将模型的重大变更称为*启动*。重大变更可能包括模型架构的更改、某些特征的添加或移除、超参数设置（如学习率）的更改，或其他需要在将其作为生产模型启动之前完全重新评估模型性能的变更。小的变更，例如基于新进数据的内部模型权重的微小修改，则被称为*更新*。

随着模型的更新，我们会定期保存当前模型状态的检查点。这些检查点会被推送到服务端，但它们在灾难恢复中也非常重要。可以将将新的模型检查点推送到服务端看作是一个小型的自动化模型启动过程。如果我们每小时推送新的检查点四次，那么我们每天几乎会进行一百次小型的自动化模型启动。

在主要模型发布时可能出现的所有问题，在推送新检查点到服务时也可能出现。模型可能某种方式被损坏——也许是由于错误，也许是由于训练在坏数据上。模型文件本身可能有缺陷，也许是由于写入错误、硬件故障，甚至（是的，真的）宇宙射线。如果模型是深度学习模型，在最近的训练步骤中可能会“爆炸”，内部模型参数包含 `NaN`，或者我们可能遇到消失梯度问题，有效地停止进一步学习。如果我们的检查点和推送过程是自动化的，可能会存在系统中的错误。如果我们的检查点和推送过程不是自动化的，而是依赖于手动操作，那么这个系统可能还没有准备好以连续模式运行。

当然，如果我们不定期根据新数据推送模型更新，我们的系统可能会出现很多问题，因此重点并不是避免更新模型，而是指出在将模型检查点推送到服务之前进行模型检查的重要性。典型的策略使用分阶段验证。首先，我们使用可以在离线状态下执行而不影响生产系统的测试，例如在沙盒环境中加载检查点并对一组黄金集数据进行评分。第五章 中讨论的所有离线评估方法都适用于这里。然后，我们将新的检查点加载到一个 *canary*——一个我们仔细观察以查看其是否出现故障的单个实例中，并允许其提供少量流量，只要监控保持稳定，我们就会逐步增加更新版本提供的流量量，直到最终提供 100% 的数据为止。

# 连续 ML 系统的观察

现在您对连续 ML 管道与非连续兄弟的不同之处有了一些了解，我们可以深入探讨它们的独特特征和挑战。

## 外部世界的事件可能会影响我们的系统

当我们查看广泛使用的类或对象的 API，比如来自 C++ 标准模板库的 `vector` 或 Python 的 `dictionary`，它们通常不包括一条醒目的文档行，例如：“警告：在世界杯期间行为未定义。” 谢天谢地它们没有，也不必有。

相比之下，连续 ML 系统有——或者应该有——确切的警告形式。它可能会写成这样：

###### 警告

生产中模型输入分布的任何变化可能会导致系统行为异常或不可预测，因为大多数 ML 模型的理论保证实际上仅适用于 IID 设置。³

这些变化的来源可能非常多样化和意外。体育赛事、选举夜、自然灾害、夏令时调整、恶劣天气、良好天气、交通事故、网络故障、大流行病、新产品发布——所有这些都是我们数据流变化的潜在来源，因此也会影响我们的系统行为。在所有情况下，我们可能几乎没有关于事件本身的警告，尽管第九章中描述的监控策略可能帮助您考虑及时警报所需的内容。

外部事件可能引发哪些情况？以下是一个例子。对于我们的毛线店来说，让我们想象一下，一位重要政治人物在寒冷的天气里穿着手工编织的棕色羊毛手套出现在全国电视上的影响。突然间，“棕色羊毛”搜索和购买量激增。稍作延迟后，模型根据这些新的搜索和购买数据进行更新，并学会为棕色羊毛产品预测更高的价值。我们的模型使用一种 SGD 形式进行训练，导致它变得过于自信，并为这些产品制定极高的评分。由于突然的高评分，这些产品几乎展示给了所有用户，现有库存迅速售罄。一旦所有库存售罄，就不再进行购买，但几乎所有搜索仍显示棕色羊毛产品，因为它们在我们的模型中获得了高分。

下一批数据显示，没有用户购买任何产品，而模型过度补偿，但由于“棕色羊毛”产品已经在如此广泛的查询和用户中展示，模型现在学会为几乎所有产品给出较低的评分，导致所有用户查询的结果为空或垃圾结果。这加强了没有用户购买任何东西的趋势，系统陷入螺旋下降状态，直到我们的 MLOps 团队识别出问题，将模型回滚到之前良好运行的版本，并在重新启用训练之前从我们的训练数据存储中过滤异常数据。

显然，这个例子可以通过多个层次的潜在修复和监控来解决，但它说明了系统动态可能产生的难以预料的后果。

知道各种世界事件可能导致意外系统行为的一个微妙危险是，我们可能会过快地解释观察到的指标变化或监控。知道今天阿根廷和巴西要进行一场重要的足球比赛可能会让我们假设这是观察到的系统不稳定的根本原因，并错过排除管道错误或其他系统错误的可能性。

什么样的持续机器学习系统能够完全抵御分布变化的影响？基本上，我们需要一种方法来自适应地加权训练数据，使其分布不会随着世界变化而改变。一种做法是创建一个模型来进行*倾向评分*，显示在给定时间内某个示例在我们的训练数据中出现的可能性。然后，我们需要按照这些倾向评分的倒数来加权我们的训练数据，以便罕见的示例被赋予更大的权重。当世界事件导致某些示例突然比过去更可能发生时，倾向评分需要被迅速更新以相应地减小其权重。最重要的是，我们需要确保所有示例的倾向评分非零，以避免在进行倒数倾向评分时出现除以零的问题。

我们需要避免倾向评分过小，以免倒数倾向加权由于少数拥有巨大权重的示例而被放大。这可以通过限制权重来实现，但我们还需要这些评分在统计上是无偏的，而限制权重会引起偏差。相反，我们可以使用广泛的随机化来确保没有示例的被包括概率太低，但这可能会意味着向用户暴露随机或无关的数据，或者使我们的模型建议可能不希望的随机行动。总之，在理论上实现这样的设置是可能的，但在实践中极其困难。

###### 提示

现实情况是，我们可能需要找到管理由于分布变化而导致的不稳定性的方法，而不是完全解决这些问题。

## 模型可以影响它们自己的训练数据

对于我们的持续机器学习系统，最重要的问题之一是模型和其训练数据之间是否存在*反馈循环*。显然，所有训练过的模型都受到用于训练的数据流的影响，但某些模型也反过来影响收集到的数据。

要帮助理解这些问题，考虑一下一些模型系统对重新训练收集的数据流没有影响。天气预测模型就是一个很好的例子。无论气象站想让我们相信什么，预测明天将是个晴天并不会真正影响大气条件。这类系统在这种意义上是干净的，因为它们没有反馈循环，我们可以对模型进行更改而不必担心我们可能会影响到明天的实际降雨概率。

其他模型确实会影响它们的训练数据的收集，特别是当这些模型向用户提供建议或决定影响它们可以学习的内容时。这些创建了隐式反馈循环，就像任何听过麦克风尖叫反馈的人知道的那样，反馈循环可能会产生意想不到的有害效果。

作为一个简单的例子，一些依赖于反馈循环来了解新趋势的系统，如果从未有机制允许它们首先尝试新事物，可能会完全错过它们。我们可以通过尝试让孩子第一次尝试新食物的经验来思考这种效应。作为一个更具体的例子，考虑一个模型，帮助推荐给用户展示羊毛产品；该模型可能会根据用户对这些选择产品的反应进行训练，例如点击、页面浏览或购买。模型将收到关于展示给用户的产品的反馈，但*不会*收到关于未被选择的产品的反馈。⁴ 很容易想象，例如有一种新的羊毛产品，如有机羊驼毛的新颜色，可能是用户非常喜欢购买的东西，但模型没有之前的数据。在这种情况下，模型可能会继续推荐以前的非有机产品，并对其遗漏视而不见。

不是发现新事物是坏事，但更糟糕的行为可能会发生。想象一下，有一个股市预测模型负责根据市场数据选择买入和卖出股票。如果一个外部实体错误地进行了大规模的卖出，模型可能会观察到这一点，并预测市场即将下跌，建议大多数持有也应该卖出。如果这次卖出足够大，将会导致市场下跌，这可能会使模型更加积极地想要卖出。有趣的是，其他模型——可能来自完全不同的组织——也可能在市场上看到这个信号，并决定进行卖出操作，从而形成一个反馈循环，导致整体市场崩盘。

股票预测场景虽然是一个极端案例，但事实上确实发生过。⁵ 然而，我们并不需要存在在竞争模型的广泛市场中才能体验到这些影响。例如，想象一下，在我们的*yarnit.ai*商店中，有一个模型负责向用户推荐产品，另一个模型负责确定何时给用户提供折扣或优惠券。如果产品推荐依赖于购买行为作为训练的信号，而折扣的存在影响购买行为，那么就存在一个反馈循环将这两个模型联系起来，而且对一个模型的更改或更新可能会影响另一个模型。

反馈回路是另一种分布变化形式，我们描述的倾向加权方法可能会有所帮助，尽管要完全正确地实现它们并不容易。通过记录模型版本信息以及其他训练数据，将这些信息作为模型的特征之一，可以在一定程度上减少反馈回路的影响。这至少让我们的模型有机会区分观察到的数据突然变化是由于现实世界的变化——比如假期结束，而在一月初没有人想购买羊毛（或者基本上任何东西）——还是模型学习状态的变化。

## 时间效应可以在多个时间尺度上出现

当我们关心数据随时间变化的方式时，我们创建连续的机器学习系统。其中一些变化对基础产品需求至关重要，比如引入新型合成羊毛或创建适用于家庭使用的自动编织机。尽快整合关于这些新趋势的数据对我们 *yarnit.ai* 商店非常重要。

其他时间效应是周期性的，周期至少会发生在三个主要时间尺度上：

季节性

许多连续的机器学习系统会经历深刻的季节性影响。对于像 *yarnit.ai* 这样的在线商务网站，随着冬季假期的临近，购买行为可能会出现显著的变化和增加，随后在一月初会突然下降。温暖的天气月份和凉爽的天气月份可能会有非常不同的趋势——甚至可能因地理区域或南半球与北半球的差异而显著变化。处理季节性影响的最有效方式是确保我们的模型训练了过去一整年的数据（如果我们有幸拥有），并且时间信息作为训练数据的特征信号包含在内。

每周

数据不仅会随季节变化，还会根据周周期循环，基于每周的具体日期。对某些情况来说，周末的使用量可能显著增加，比如我们 *yarnit.ai* 商店中面向爱好者的部分，或者在面向企业间销售的部分则可能显著减少。地域因素在这里起着重要作用，因为周末的情况可能因国家而异，而时区影响也很重要，例如在东京可能是星期一，而在旧金山仍然是星期天。

每天

当我们看到基于一天中不同时间的日常效应时，事情就开始变得复杂了。乍一看，显而易见的是，许多系统在一天中的不同时间可能会体验到不同的数据——午夜的行为可能与清晨或工作日的行为不同。显然地，地域因素在这里至关重要，因为时区效应。

日常周期的微妙之处在于，当我们考虑到大多数连续机器学习系统实际上是在现实之后持续运行时，就会出现问题，这是因为管道和数据流等待训练标签（如点击或购买行为可能需要一些时间才能确定）的固有延迟，以及过滤坏数据、更新模型、验证检查点，并将检查点推送到服务端并完全升级。事实上，这些延迟可能累计达到 6 甚至 12 小时。因此，我们的模型可能与现实相差甚远，服务的模型版本认为现在是午夜，实际上却是工作日的中心。

幸运的是，通过记录每天的时间信息以及其他训练信号，并将它们用作我们模型的输入，可以相对容易地修复这些问题。但这也突显了一个重要问题，即我们在服务中加载的模型版本可能已经过时，或者在实际情况下得到了错误信息。

## 紧急响应必须实时完成

到了这一点，应该清楚了，虽然连续机器学习系统可以提供巨大的价值，但它们也具有广泛的系统级脆弱性。可以说，连续机器学习系统继承了所有大型复杂软件系统的生产脆弱性，这些系统本身就存在大量的可靠性问题，并且还添加了一整套可能导致未定义或不可靠行为的额外问题。在大多数情况下，我们无法依赖于理论保证或正确性证明。

当连续机器学习系统出现此类问题时，不仅需要修复，而且需要实时修复（或减轻）。这其中有几个原因。首先，我们的模型可能是使命关键的，如果生产中出现问题，可能会影响我们组织的分钟到分钟的服务能力。其次，我们的模型可能与自身存在反馈循环，这意味着如果我们不及时解决问题，输入数据流也可能被破坏，需要小心修复。第三，我们的模型可能是更大生态系统的一部分，很难重置到已知状态。当我们的模型与其他模型存在反馈循环，或者模型的糟糕预测造成长期伤害并且难以撤销时，这种情况就会发生，比如原本有帮助的*yarnit.ai*聊天机器人突然对用户发出粗鲁的咒骂。

实时危机响应首先需要快速检测问题，这意味着从组织的角度来看，确定是否准备好进行连续 ML 的一个良好试金石是检查我们的监控和警报策略的彻底性和及时性。由于管道延迟和系统逐渐在新数据上学习的缓慢变化，数据缺陷产生下游有害影响的全面效果可能需要一些时间。这使得拥有简单的金丝雀指标特别重要，这些指标在输入分布发生变化时发出警报，而不是等待模型输出发生变化。

仅当有责任在警报触发时响应的人员支持时，监控或调整策略才会在实时中有所帮助。运作良好的 MLOps 组织设定了关于警报必须多快响应的具体服务水平协议，并设置了像电话值班轮换这样的机制，以确保警报在正确的时间被正确的人看到。拥有一个在多个时区拥有队友的全球团队，如果你处于这种幸运的位置，可以极大地帮助避免人们因警报在凌晨 3 点而被吵醒。

一旦收到警报，我们需要有一个良好记录的应对手册，任何 MLOps 团队成员都可以执行，并且需要一个进一步行动的升级路径。

###### 提示

对于连续 ML 系统，我们有一组针对任何给定危机的基本立即响应措施。这些是停止训练，回退，回滚，删除坏数据和通过滚动。

并非每一次危机都需要所有这些步骤，选择哪种响应最合适取决于问题的严重性以及我们能够诊断和治愈根本原因的速度。让我们看看连续 ML 系统的每个基本危机响应步骤，然后讨论选择响应策略的因素。

### 停止训练

有人说，第一条挖洞的法则是这样的：*当你发现自己在一个洞里时，停止挖掘*。类似地，当我们发现我们的数据流以某种方式被损坏，也许是由于一个糟糕的模型，或者一个停机，或者系统中某处的代码级错误时，一个有用的反应可以是停止模型训练，并停止推送任何新的模型版本到服务中。这是一个短期的响应，至少有助于确保问题不会在我们决定采取缓解或修复措施之前变得更糟。确保 MLOps 团队有一种简单的方法来停止他们负责的任何模型的训练是有意义的。自动化系统在这里是有帮助的，但当然需要足够的警报，以免发现模型已经在三周前静默停止训练了。

###### 提示

总是有用的有一个类似于大红按钮的东西，可以在发现紧急情况时手动停止训练。

### 回退

在持续的机器学习系统中，有一个备用策略非常重要，可以用来替代我们的生产模型，提供可接受的（即使是非最佳的）结果。这可以是一个远比较不连续训练的更简单的模型，或者是一个包含最常见响应的查找表，甚至只是一个返回所有查询的中位数预测的小函数。关键是，如果我们的持续机器学习系统遇到突然的大规模故障——我们可能描述为“失控状态”——我们有一个超可靠的方法可以作为临时替代品，而不至于使整个产品变得完全无法使用。备用策略通常在整体性能上比我们的主要模型不太可靠（否则，我们首先不会使用机器学习模型），因此备用策略非常适合作为短期响应，允许在系统的其他部分采取紧急响应。

### 回滚

如果我们的持续机器学习系统当前处于危机状态，将系统恢复到危机之前的状态并查看一切是否正常是有意义的。我们危机的根本原因可能来自两个基本领域：糟糕的代码或糟糕的数据。

如果我们相信我们问题的根本原因是糟糕的代码，由于最近引入的错误，那么将我们的生产二进制文件回滚到以前已知的良好版本可能会在短期内修复问题。当然，任何回滚到以前的生产二进制文件都必须在逐步增加的阶段进行，以防存在使旧版本二进制文件不再可用的新兼容性问题或其他缺陷。无论如何，在需要时保持一组完全编译的先前二进制文件非常重要，以便可以快速有效地进行回滚。

如果我们认为问题的根本原因是糟糕的数据导致模型训练自身进入糟糕状态，那么将模型版本回滚到以前已知的良好版本是有意义的。同样，保持我们训练的生产模型的检查点非常重要，这样我们就有一组以前的版本可以选择。例如，想象一下，美国的黑色星期五销售导致用户向我们的*yarnit.ai*商店发出大量购买请求，使系统的欺诈检测部分开始将所有购买标记为无效，使我们的模型看起来所有产品都极不可能被购买。回滚到一个在黑色星期五日期一周之前检查点的模型版本，至少可以让模型提供合理的预测，同时修复更大系统的其余部分。

### 删除糟糕的数据

当我们的系统中有不良数据时，我们需要一种简单的方法将其移除，以免模型受其污染。在前面的例子中，我们希望移除由于错误的欺诈检测系统而受损的数据。否则，当我们重新启用训练继续进行时，这些数据将被我们回滚的模型遇到，因为它在训练数据中向前移动，这些坏数据会再次对其造成污染。每当我们认为数据本身极不典型且不太可能为模型提供有用的新信息，并且坏数据的根本原因是临时的，可能是由于外部世界事件或系统中的错误而导致的时候，移除坏数据是一个有用的策略。

### 穿越

如果我们已经停止了连续 ML 系统的训练，在某些时候我们需要祈祷并启用它来恢复训练。通常情况下，我们会在清除了不良数据并确保修复了所有错误后进行此操作。然而，如果由于外部事件检测到了危机，有时候最好的响应策略就是祈祷并穿越，让模型在非典型数据上进行训练，然后在事件结束时自行恢复。事实上，遗憾的是，这个世界很少有没有政治事件、重大体育赛事或其他重大新闻灾难发生的日子，确保我们的模型充分接触来自不同全球地区的这类非典型数据可能是确保模型总体上健壮性的重要方式之一。

### 选择响应策略

我们如何选择停止、回滚和移除哪些事件，以及穿越哪些事件？要回答这个问题，我们需要观察我们的模型对类似历史事件的响应，这在我们按顺序训练模型的历史数据时最容易实现。另一个重要问题是，我们目前看到的指示危机的度量是否是由于模型错误或世界处于非典型状态造成的。换句话说，是我们的模型出了问题，还是它只是在目前被要求处理更困难的请求？判断这一点的一种方法是观察黄金数据集上的离线指标，出于这个原因，应该经常重新计算我们模型的黄金数据集结果。如果模型确实出现问题，黄金数据集的结果可能会显示性能急剧下降，这时穿越可能不是正确的方法。

### 组织考虑事项

当当前正发生危机时，这可能是学习新技能、在团队内确定角色或决定如何实施各种响应和缓解策略的困难时期。现实世界的消防员经常一起训练，完善最佳实践，并确保他们需要响应警报的所有基础设施都处于良好状态，并随时准备投入行动。同样，我们不知道我们的持续机器学习系统何时需要危机响应，但我们可以有信心地说，这将会发生，并且我们需要做好充分准备。创建有效的危机响应团队是创建和维护持续机器学习系统成本的一部分，在我们朝这个方向迈进时必须加以考虑。这在“持续组织”中详细讨论。

## 新推出需要分阶段增长和稳定基线。

当我们在持续的机器学习系统中运行模型一段时间后，最终会希望推出新版本的模型，以在各个方面实现改进。也许我们想要使用一个更大的模型版本来提高质量，现在有能力处理它，或者可能模型开发者创建了几个新特性，显著提高了预测性能，或者我们发现了一种更高效的模型架构，以重要方式降低了服务成本。在这些情况下，我们需要明确地推出新版本的模型来替换旧版本。

推出新模型往往涉及一定程度的不确定性，这是因为离线测试和验证的限制。正如我们在第十一章中描述的那样，离线测试和验证可以为我们提供关于新模型版本在生产环境中表现良好的有用指导，但通常无法给出完整的画面。当我们的持续机器学习模型是反馈环路的一部分时，情况尤其如此，因为我们先前训练的数据很可能是由先前的模型版本选择的，而离线数据的评估仅限于基于先前模型做出的行动或建议的数据。我们可以将这种情况想象成是学员驾驶员的情况，他们首先通过坐在副驾驶座位上评估，被要求对教练驾驶汽车的行为给出意见。仅仅因为他们对教练的行动完全同意，并不意味着他们在首次有机会自己驾驶汽车时不会做出一些不良决策。

因此，新模型的推出需要在生产环境中进行一定量的 *测试*，作为最终验证的形式。我们需要让新模型展示它有能力成为驾驶座位上的模型。但这并不意味着我们只是交给它钥匙，然后期望一切都完美无缺。相反，我们通常会使用分阶段逐步增加的策略，首先允许模型仅服务整体数据的一小部分，并且只有在观察到长期良好表现后才增加这一部分。这种策略通常被称为 *A/B 测试*：我们测试我们的新模型 A 与旧模型 B 的性能，以类似控制科学实验的格式，帮助验证新模型在我们的最终业务指标上是否表现得合适（这可能与离线评估指标如准确性不同）。

模型发布和理想的 A/B 测试之间的区别在于，在科学实验中，A 和 B 是独立的，彼此不会互相影响。例如，如果我们在科学环境中进行实验，以确定棉质毛衣（A）是否像天然羊毛毛衣（B）一样保暖，那么穿羊毛毛衣的人不太可能使穿棉毛衣的人报告感觉更暖或更冷。然而，如果穿羊毛毛衣的人感到非常温暖和愉快，以至于他们去为穿棉毛衣的人煮茶和热汤，那肯定会破坏我们的实验。

对于比较连续 ML 模型的 A/B 实验，事实证明当我们的模型处于反馈循环的一部分时，A 和 B 可能会互相影响。例如，想象一下，我们的新模型 A 在向 *yarnit.ai* 用户推荐有机羊毛产品方面做得很好，而我们之前的模型 B 从未这样做过。最初的 A/B 实验可能会显示 A 模型在这方面要好得多，但随着 A 生成的训练数据包括了许多有机羊毛的推荐和购买，B 模型（也在持续更新）可能也会学习到这些产品受用户喜欢，并开始推荐它们，使得这两个模型随着时间的推移看起来变得相似。如果这些影响更加分散，那么很难说 A 的好处是否消失了，因为它实际上从未比 B 更好，或者是因为 B 本身已经改进。

我们可以尝试通过限制 A 和 B 只在它们自己服务的数据上进行训练来解决这个问题。当每个模型提供相同数量的数据时，比如总流量的 50%，这种策略可以很好地工作，但在其他情况下可能会导致比较出现缺陷。如果 A 在早期看起来不好，是因为模型不好，还是因为它只有 1% 的训练数据，而 B 有 99%？

另一种策略是尝试创建某种稳定的基线，这可以帮助作为比较的参考点，以便我们可以弄清楚 A 和 B 之间的比较是因为 A 变得更糟还是因为 B 变得更好，或者确实，两者都在同步变得显著更糟。稳定的基线是一个不受 A 或 B 影响的模型 C，并且允许提供一定量的流量，以便我们可以将这些结果用作比较的依据。基本思想是然后观察(A-C)对(B-C)而不是直接对比 A 对 B，这将使我们能够更清楚地看到任何变化。

创建稳定基线的四种一般策略具有不同的优缺点：

作为基准的回退策略

当我们有一个合理的回退策略，不涉及持续的重新训练时，这不仅对危机响应有用，还作为一个潜在的独立数据点。如果回退策略的质量与主要生产模型的差异不太大，则这可以很好地工作。然而，如果差异非常大，统计噪声可能会压倒使用此作为参考点的 A 和 B 之间的任何比较。

停止训练器

如果我们有我们的生产模型 B 的一个副本并停止对其进行训练，那么根据定义，它将不会受到 A 或 B 的任何未来行为的影响。如果我们允许其提供少量流量服务，则可以提供一个有用的稳定基线 C，但“停止训练器”模型的整体性能会随时间缓慢下降。进行独立实验以观察可以预期的退化程度以及该策略是否有用是有益的。

延迟训练器

如果我们预计整体启动过程需要，比如说，两周时间，那么一个合理的选择可以是运行我们的生产模型的一个副本，设置为持续更新，但延迟两周。这相比于停止训练器有一个优势，即相对性能不太可能下降，但缺点是在运行时间等于其延迟的长度之后，它将开始受到 A 和 B 的影响，并且失去其效用。因此，两周延迟训练器模型将在两周后变得无用。

并行宇宙模型

一个保持与 A 和 B 严格独立但没有有限使用寿命的方法是并行宇宙模型，该模型允许为总体数据的一小部分提供服务，并且仅在其自身提供的数据上学习。A 和 B 不在这些数据上进行训练，从而使这些数据宇宙完全分离。

为什么这很有用？想象一下，将 B 投入生产的行为会以某种方式改变整体生态系统。我们可以想象股票预测市场模型是这样运作的——也许在某些特殊情况下推动整体市场的上涨或下跌。在这种情况下，A 和 B 都可能大幅增加或减少其中位预测，但 A-B 的差异可能很小且看似稳定。有了第三个观点 C，我们能够检测到模型之间的变化是由于 A 和 B 本身的差异，还是由于更广泛的影响。

平行宇宙模型在建立后通常需要时间来稳定，因为训练数据量有限且整体分布会发生变化。但经过这个初始期后，它们可以提供一个有用的独立评估点——当比较评估指标时，要考虑统计噪声的限制。

## 模型必须进行管理，而不是简单地投入使用。

总体而言，模型的发布需要特别注意，因为在这些时候，我们的系统最容易遇到危机。如果我们在 A 和 B 均大约占一半流量的模型发布过程中，我们刚刚增加了潜在的错误来源，并且需要处理任何可能出现的紧急情况的工作量翻倍。像危机响应一样，模型的发布在依赖于沟通良好、经过良好实践的流程时效果最佳。

有些产品就像砖墙：它们需要大量的规划和努力来完成，但一旦完成，基本上就完成了，只需要偶尔维护。默认状态是它们正常工作。连续的机器学习系统则处于相反的极端，并且需要每天关注。连续 ML 模型出现的问题可能难以以完全或永久的方式解决。例如，如果我们整体产品的一个问题是希望更好地向温暖气候的用户推荐羊毛产品，这可能需要采用多种方法，而这些方法的效用可能会随着季节和年份的变化而变化。

以这种方式，持续机器学习系统需要定期管理。有效管理模型需要每日访问报告模型预测性能的指标。这可以通过仪表板和相关工具完成，使模型管理员能够了解今天的情况、趋势可能如何变化以及可能出现的问题所在。任何仪表板的效用都取决于对其付出的关注程度，因此需要明确的负责人定期花时间关注。一个有用的比喻是，每天和我们的模型喝一杯咖啡是有益的，通过使用仪表板了解今天它的表现。就像人员经理定期进行绩效评估一样，模型所有者应定期向组织高层提交有关模型性能的报告，以分享知识和可见性。

当我们从我们的模型中学到一些有用的东西时，一个强大的最佳实践是以简短的写作形式记录下来。这样的写作，可能只是几段文字，附带有仪表板截图或类似的支持证据，可以帮助积累组织知识，当观察伴随有关于模型行为意义的简短总结时效果最佳。这样的写作历史上被证明非常有用，不仅有助于指导未来的模型发展，还有助于理解和调试危机中出现的意外行为。

最后，在我们遇到危机时，从组织角度来看，通过创建事后分析文档尽可能多地从经验中吸取教训是非常重要的。这些文档详细描述了事件的发生过程，问题的诊断方法，造成的损害，应用的缓解策略以及其成功程度，最后提出了改进建议，无论是减少问题再次发生的频率还是增强未来更有效的应对能力。在短期内，创建这些事后分析文档有助于识别修复措施，而长期来看，它们作为组织知识和经验的库存同样有用，随时可供参考。

# 持续的组织

到了这一点，应该清楚的是，一个致力于持续机器学习系统的组织正在承担长期的责任。像小狗一样，持续机器学习系统需要日常关注、照料、喂养和训练。为了确保我们的组织能够有效地处理持续机器学习系统的管理责任，需要建立多种有效的结构。

确定评估策略是一项关键的领导责任。评估策略使我们能够评估我们模型的健康和质量，无论是从长期的业务目标还是从短期决策来看，比如是否将某个特定特征包含或排除在模型之外。如第九章所述，把这个问题简化为确定度量指标可能是诱人的，但是一个给定的度量指标（如收入、点击、精度、召回率甚至延迟）如果没有参照点、基准或分布，就毫无意义。在持续的机器学习环境中做决策常常需要进行某种反事实推理，思考反馈环路的影响，或是应对噪声和不确定性，这些都使有效的决策变得具有挑战性。通过制定清晰定义并记录评估标准和流程，我们可以在一定程度上帮助减少这些挑战的难度。

在组织层面，做出投资决策同样具有挑战性。我们应该投入多少资源来创建更大、潜力更强的模型，这种投资是否能够通过改善产品结果来回本，相对于机会成本来说？我们如何在投资模型质量改进和机器学习系统级可靠性之间进行权衡？在组织层面，如何有效地引导有限的人力专家的时间和精力，以最大程度地使整体使命受益？这些都是根本性难题，其中一部分原因是我们组织的不同部分可能具有不同甚至是不兼容的优先事项。

我们有两种主要策略来处理这些问题。第一种是确保我们有足够广泛的组织领导能够有效权衡改善模型监控和危机响应处理的不同需求，与提高模型准确性的需求。确保每个组织部分的经验教训能够在整个组织内充分传播，可以帮助不同部门理解彼此的挑战和痛点。最好的方式是定期分享事故事后总结，并通过主动的“事前分析”讨论识别潜在的弱点和故障模式来实现。第二种策略是投资于基础设施，确保警报和其他警告在生产者和消费者的完整链条中得到有效传播。这可能需要组织的认真承诺，但长期来看可以在减少复杂系统内部验证的人力负担方面收益。

在组织上，理解持续机器学习系统依赖于持续的数据流来确定系统行为，显然表明数据管道本身需要严肃、专注的监督和管理。除了简单地确保数据流畅和管道功能良好外，关键问题还包括应收集哪些类型的数据、数据存储多长时间以及我们的数据管道应如何与上游生产者和下游消费者进行交互，这些都是组织领导者必须解决的关键战略问题。隐私、伦理、数据包容性和公平性等更深层次的问题同样起着重要作用，并且必须成为整体组织战略的一部分。

正如我们所指出的，机器学习系统的启动过程及进一步改进在持续机器学习设置中必然需要一个分阶段的逐步启动过程。领导层的关键角色是对每个阶段的结果进行监督和评审，并作出是否进入下一个阶段的批准决定，或者确定我们尚未准备好继续前进，甚至需要降低规模，如果事情的发展不如预期。这些决策需要高层监督，因为后果可能影响深远，并且可能与多个生产者或消费者系统发生交互，特别是在存在反馈循环或其他复杂的系统对系统交互点时。在进行各个启动阶段的广泛影响评估并确保稳定之前，必须确立并严格遵循一套流程，这对于长期有效的持续机器学习组织至关重要。

最后，当事件或危机时刻发生时，我们需要建立一个有效的响应流程。对于持续的机器学习系统，我们在处理事件时有优势和劣势。优势在于，我们几乎总是有一个稍微旧一些的当前模型版本，可以在我们检查出错原因的同时回滚到服务中，这在我们需要迅速应对当前模型出现严重问题时非常有帮助。劣势在于整个系统不断演化，难以隔离根本原因或变更中的断裂点。我们模型的变更可能是由系统中的其他并发变更（如添加新数据或使用模型的新集成）驱动或需要的。在这种情况下，要精确地排查问题变得更加困难，尤其是在持续的机器学习环境中。关于这一点的具体例子，请参阅第十一章。

运行连续 ML 系统的组织最重要的前期工作是预先协商停机的后果和处理方式。这不仅仅是提前确定谁将担任哪些角色的问题 — 虽然这也是必须做的。ML 生产工程师不应在处理中确定解决特定事件的紧急程度，模型开发者也不应在停机事件发生时猜测成本和后果。每个人都应清楚自己的角色、权限以及向他人升级决策的途径，因为这些应该事先制定好。但是在可能的情况下，整个组织还应就一些一般的服务可靠性标准达成共识。例如：

+   即使我们通常希望其基于不超过大约 1 小时的数据，此模型可以达到 12 小时的老化而没有严重后果。

+   如果此模型的质量指标低于已知特定阈值，批准的回退方法是回滚到旧模型。

+   如果模型低于预定的额外阈值，唤醒以下业务领导是适当的…

依此类推。总体思路是预先确定各种类型的停机参数，以便事件响应者能够最大程度地采取行动，并尽量减少对如何应对的决策延迟。许多组织通常不会在事先做出这些决策，直到经历了一些停机事件后，才会认为预先决定在遇到严重 ML 问题时应采取何种措施是非常合理的。

# 重新思考非连续 ML 系统

本章节我们讨论了连续 ML 系统的一系列问题。除了提供希望能为面对困难时创建健壮可靠系统的有用缓解策略之外，我们还想提出一个更广泛的建议：

> 所有生产 ML 系统都应被视为连续 ML 系统。

我们应该把每个作为生产系统关键部分的模型都视为连续训练的模型，即使它实际上并非每分钟、每小时，甚至每天或每周都会基于新数据进行更新。

为什么我们要做出这样的建议？毕竟，连续 ML 系统充满了复杂性和故障向量。一个原因是，如果我们将连续 ML 系统的标准和最佳实践应用于所有生产级 ML 系统，我们将确保我们的技术基础设施、模型开发以及 MLOps 或危机响应团队都能够应对挑战。如果我们假设我们的 ML 系统是一个连续 ML 系统并相应地进行规划，我们将处于一个良好的位置。

这是否过度？如果一个模型只被训练一次，那么应用连续机器学习的标准和最佳实践可能会被视为资源浪费。但实际上，没有任何生产模型只会被训练一次。根据我们的经验，我们发现每个生产级机器学习模型最终都会重新训练或推出新版本——可能在几个月后，或明年，随着新数据的出现或模型的发展。这可以通过不定期的方式进行，每隔几周或几个月一次，但这种不规律的方法很可能导致失败和疏忽。我们强烈建议，有效的 MLOps 团队确保他们的模型按照规定的时间表进行更新，无论是每天、每周还是每月，以便验证程序和检查表能够成为组织文化的一部分。从这个角度来看，连续机器学习系统的建议适用于每一个机器学习系统。

# 结论

在本章中，我们提出了一套程序和实践方法，可作为组织管理连续机器学习系统时的基础手册。这些系统提供了广泛的好处，能够使模型随时间适应新数据，并允许响应式学习系统与用户、市场、环境和世界进行交互。

很明显，任何既具有高影响力又容易受影响的系统都需要进行深入监督。根据我们的经验，在这些情况下，监督需求远远超出个人能力范围，不能依赖直觉或即兴处理。任何管理连续机器学习系统的组织都需要把这视为一项持续的高优先级任务，特别是在系统发布或重大更新时，但也需要进行持续的监控，并制定应急预案，以便对新出现的危机能够快速响应。

我们总结了关于连续机器学习系统的六个基本观点，希望您能从中受益：

+   外部世界事件可能会影响我们的系统。

+   模型可能通过反馈循环影响其未来的训练数据。

+   时间效应可能在多个时间尺度上出现。

+   危机响应必须实时进行。

+   新发布需要分阶段的推进和稳定的基线。

+   模型必须进行管理而非简单部署。

最后，我们认为所有机器学习系统最好都被视为连续机器学习系统，因为所有模型最终都会重新训练，建立强大的标准将使任何组织长期受益。

¹ 详见“5\. 广告点击预测：数据库与现实”，这里有一个相关案例。

² 详见“特征存储”以深入了解特征存储。

³ 在统计学和机器学习术语中，IID 假设是数据是从相同分布独立抽取的，也就是说，我们的测试集和训练集以相同的方式随机抽取自同一来源。这在第五章中有更详细的介绍。

⁴ 熟悉机器学习不同子领域的人士会注意到，这在技术上是一个情境感知的强化学习设置，因此存在探索与利用的权衡。在这种情境下的主要思想是，当我们的系统只学习它有意选择的内容时，有时随机改变选择以更多地探索世界，并确保我们不会将系统锁定在自我永续信念的循环中是至关重要的。探索过少会导致错失最佳机会；过多则是时间和资源的浪费。当然，这与我们自己生活中的任何选择有什么相似之处，完全是巧合。

⁵ ​​参见[*闪崩：一场全新的解构*](https://oreil.ly/sAkYH)，详细讨论导致闪崩事件的事件，包括触发事件及促成市场条件的考量。从这次讨论中可以得出一个结论，即在涉及多个模型相互作用的系统中进行根本原因分析是多么困难。
