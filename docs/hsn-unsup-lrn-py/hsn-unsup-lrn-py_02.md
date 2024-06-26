# 第一章：机器学习生态系统中的无监督学习

> 大多数人类和动物的学习都是无监督学习。如果智能是一个蛋糕，无监督学习将是蛋糕，监督学习将是蛋糕上的糖衣，而强化学习将是蛋糕上的樱桃。我们知道如何制作糖衣和樱桃，但我们不知道如何制作蛋糕。在我们甚至考虑真正的 AI 之前，我们需要解决无监督学习问题。
> 
> 伊恩·拉坤

在本章中，我们将探讨基于规则的系统与机器学习、监督学习与无监督学习之间的区别，以及每种方法的相对优势和劣势。

我们还将介绍许多流行的监督学习算法和无监督学习算法，并简要探讨半监督学习和强化学习如何融入其中。

# 基础机器学习术语

在我们深入探讨不同类型的机器学习之前，让我们先看一个简单且常用的机器学习示例，以帮助我们更具体地理解我们介绍的概念：电子邮件垃圾过滤器。我们需要构建一个简单的程序，输入电子邮件并正确地将它们分类为“垃圾邮件”或“非垃圾邮件”。这是一个直接的分类问题。

这里是一些机器学习术语的复习：这个问题的*输入变量*是电子邮件的文本。这些输入变量也被称为*特征*或*预测变量*或*独立变量*。我们试图预测的*输出变量*是标签“垃圾邮件”或“非垃圾邮件”。这也被称为*目标变量*、*依赖变量*或*响应变量*（或*类*，因为这是一个分类问题）。

AI 训练的示例集被称为*训练集*，每个单独的示例称为训练*实例*或*样本*。在训练过程中，AI 试图最小化其*成本函数*或*错误率*，或者更积极地说，最大化其*价值函数*—在本例中，是正确分类的电子邮件比例。AI 在训练期间积极优化以达到最小的错误率。它的错误率是通过将 AI 预测的标签与真实标签进行比较来计算的。

然而，我们最关心的是 AI 如何将其训练推广到以前从未见过的电子邮件上。这将是 AI 的真正测试：它能否使用在训练集示例中学到的知识正确分类它以前从未见过的电子邮件？这种*泛化误差*或*样外误差*是我们用来评估机器学习解决方案的主要指标。

这组以前从未见过的示例被称为*测试集*或*保留集*（因为这些数据被保留在训练之外）。如果我们选择有多个保留集（也许在训练过程中评估我们的泛化误差是明智的），我们可能会有用于评估我们进展的中间保留集，这些中间保留集称为*验证集*。

将所有这些结合起来，AI 在训练数据（*经验*）上进行训练，以提高在标记垃圾邮件（*任务*）中的错误率（*性能*），最终成功的标准是其经验如何推广到新的、以前从未见过的数据上（*泛化误差*）。

# 基于规则与机器学习

使用基于规则的方法，我们可以设计一个垃圾邮件过滤器，通过明确的规则捕捉垃圾邮件，比如标记使用“u”代替“you”，“4”代替“for”，“BUY NOW”等的电子邮件。但是随着坏人改变他们的垃圾邮件行为以逃避规则，这种系统在时间上会很难维护。如果我们使用基于规则的系统，我们将不得不经常手动调整规则，以保持系统的最新状态。而且，设置这种系统将非常昂贵——想象一下我们需要创建多少规则才能使其正常运行。

与基于规则的方法不同，我们可以使用机器学习来训练电子邮件数据，并自动创建规则以正确标记恶意电子邮件为垃圾邮件。这种基于机器学习的系统也可以随着时间的推移自动调整。这种系统的培训和维护成本要低得多。

在这个简单的电子邮件问题中，我们可能可以手工制定规则，但是对于许多问题来说，手工制定规则根本不可行。例如，考虑设计自动驾驶汽车——想象一下为汽车在每一个遇到的情况下如何行为制定规则，这是一个棘手的问题，除非汽车可以根据自己的经验学习和适应。

我们还可以将机器学习系统作为探索或数据发现工具，以深入了解我们尝试解决的问题。例如，在电子邮件垃圾邮件过滤器的示例中，我们可以学习哪些单词或短语最能预测垃圾邮件，并识别新出现的恶意垃圾邮件模式。

# 监督学习与非监督学习

机器学习领域有两个主要分支——*监督学习*和*无监督学习*——以及许多桥接这两者的子分支。

在监督学习中，AI 代理可以访问标签，这些标签可以用来改善其在某些任务上的表现。在电子邮件垃圾邮件过滤问题中，我们有一个包含每封电子邮件中所有文本的数据集。我们还知道哪些邮件是垃圾邮件或非垃圾邮件（所谓的*标签*）。这些标签在帮助监督学习 AI 区分垃圾邮件和其他邮件方面非常有价值。

在无监督学习中，没有标签可用。因此，AI 代理的任务并不是明确定义的，性能也不能如此清晰地衡量。考虑电子邮件垃圾邮件过滤器问题——这次没有标签。现在，AI 代理将尝试理解电子邮件的基本结构，将电子邮件数据库分成不同的组，使得组内的电子邮件彼此相似但与其他组的电子邮件不同。

这个无监督学习问题比监督学习问题的定义不太明确，对 AI 代理来说更难解决。但是，如果处理得当，解决方案将更为强大。

原因在于：无监督学习 AI 可能会发现几个后来标记为“垃圾邮件”的组，但 AI 也可能会发现后来标记为“重要”的组，或者归类为“家庭”、“专业”、“新闻”、“购物”等。换句话说，由于问题没有严格定义的任务，AI 代理可能会发现我们最初未曾寻找的有趣模式。

此外，这种无监督系统在未来数据中发现新模式的能力优于监督系统，使得无监督解决方案在前进时更加灵活。这就是无监督学习的力量。

## 监督学习的优势和劣势

监督学习在定义良好的任务和充足标签的情况下优化性能。例如，考虑一个非常大的对象图像数据集，其中每个图像都有标签。如果数据集足够大，并且我们使用正确的机器学习算法（即卷积神经网络）并且使用足够强大的计算机进行训练，我们可以构建一个非常好的基于监督学习的图像分类系统。

当监督学习 AI 在数据上进行训练时，它将能够通过比较其预测的图像标签与我们文件中的真实图像标签来测量其性能（通过成本函数）。AI 将明确尝试将这个成本函数最小化，使其在以前未见过的图像（从留存集）上的错误尽可能低。

这就是为什么标签如此强大——它们通过提供错误度量来指导 AI 代理。AI 使用这个错误度量随着时间的推移来提高其性能。没有这样的标签，AI 不知道它在正确分类图像方面有多成功（或不成功）。

然而，手动标记图像数据集的成本很高。即使是最好的策划图像数据集也只有数千个标签。这是一个问题，因为监督学习系统在对具有标签的对象图像分类方面表现非常出色，但在对没有标签的对象图像分类方面表现不佳。

尽管监督学习系统非常强大，但它们在将知识推广到以前未见过的实例上的能力也受到限制。由于世界上大多数数据都没有标签，因此使用监督学习时，AI 将其性能扩展到以前未见过的实例的能力是相当有限的。

换句话说，监督学习擅长解决狭义 AI 问题，但在解决更有雄心、定义不太明确的强 AI 类型问题时表现不佳。

## 无监督学习的优势和劣势

在狭义定义的任务中，有着明确定义的模式并且随时间变化不大以及具有充足可用的标记数据集时，监督学习将在效果上胜过无监督学习。

然而，对于那些模式未知或不断变化，或者我们没有足够大的标记数据集的问题，无监督学习确实表现出色。

无监督学习不依赖标签，而是通过学习其训练的数据的基本结构来工作。它通过试图用比数据集中可用示例数量显著较小的一组参数来表示其训练的数据来实现这一点。通过执行这种表示学习，无监督学习能够识别数据集中的不同模式。

在图像数据集示例中（这次没有标签），无监督学习的 AI 可能能够根据它们彼此的相似性以及与其余图像的不同性将图像识别并分组。例如，所有看起来像椅子的图像将被分组在一起，所有看起来像狗的图像将被分组在一起，依此类推。

当然，无监督学习的 AI 本身无法将这些组标记为“椅子”或“狗”，但现在相似的图像被分组在一起后，人类的标记任务变得简单得多。人类可以手动标记所有不同的组，标签将应用于每个组内的所有成员。

经过初步训练后，如果无监督学习的 AI 发现了不属于任何已标记组的图像，AI 将为未分类的图像创建单独的组，触发人类标记新的、尚未标记的图像组。

无监督学习使以前棘手的问题更易解决，并且在找到历史数据和未来数据中隐藏模式方面更为灵活。此外，我们现在有了一种处理世界上存在的大量未标记数据的 AI 方法。

尽管无监督学习在解决特定、狭义定义的问题方面不如监督学习熟练，但在解决更为开放的强 AI 类型问题和推广这种知识方面表现更佳。

同样重要的是，无监督学习可以解决数据科学家在构建机器学习解决方案时遇到的许多常见问题。

# 使用无监督学习来改善机器学习解决方案

机器学习的最近成功是由大量数据的可用性、计算硬件和基于云的资源的进步以及机器学习算法的突破推动的。但这些成功主要出现在狭义 AI 问题，如图像分类、计算机视觉、语音识别、自然语言处理和机器翻译领域。

要解决更雄心勃勃的 AI 问题，我们需要发挥无监督学习的价值。让我们探讨数据科学家在构建解决方案时面临的最常见挑战，以及无监督学习如何帮助解决这些挑战。

### 标记不足的数据

> 我认为 AI 就像建造一艘火箭。你需要一个巨大的引擎和大量的燃料。如果你有一个巨大的引擎和少量的燃料，你无法进入轨道。如果你有一个微小的引擎和大量的燃料，你甚至无法起飞。要建造一艘火箭，你需要一个巨大的引擎和大量的燃料。
> 
> Andrew Ng

如果机器学习是一艘火箭，数据就是燃料——没有大量数据，火箭是无法飞行的。但并非所有数据都是平等的。要使用监督算法，我们需要大量标记数据，这在生成过程中是困难且昂贵的。¹

使用无监督学习，我们可以自动标记未标记的示例。这里是它的工作原理：我们会对所有示例进行聚类，然后将标记示例的标签应用于同一聚类中的未标记示例。未标记的示例将获得它们与之最相似的已标记示例的标签。我们将在第五章中探讨聚类。

### 过拟合

如果机器学习算法根据训练数据学习了一个过于复杂的函数，它在从保留集（例如验证集或测试集）中获得的以前未见实例上可能表现非常糟糕。在这种情况下，算法过度拟合了训练数据——从数据中提取了太多的噪声，并且具有非常差的泛化误差。换句话说，该算法是在记忆训练数据，而不是学习如何基于其泛化知识。²

为了解决这个问题，我们可以将无监督学习引入作为*正则化器*。*正则化*是一种用来降低机器学习算法复杂度的过程，帮助其捕捉数据中的信号而不是过多地调整到噪声。无监督预训练就是这种正则化的形式之一。我们可以不直接将原始输入数据馈送到监督学习算法中，而是馈送我们生成的原始输入数据的新表示。

这种新的表示捕捉了原始数据的本质——真正的底层结构——同时在过程中减少了一些不太代表性的噪声。当我们将这种新的表示输入监督学习算法时，它需要处理的噪声较少，捕捉到更多的信号，从而改善其泛化误差。我们将在第七章探讨特征提取。

### 维度诅咒

尽管计算能力有所提升，大数据对机器学习算法的管理仍然颇具挑战性。一般来说，增加更多实例并不太成问题，因为我们可以利用现代的映射-减少解决方案（如 Spark）并行操作。然而，特征越多，训练就越困难。

在非常高维空间中，监督算法需要学习如何分离点并构建函数逼近，以做出良好的决策。当特征非常多时，这种搜索变得非常昂贵，无论是从时间还是计算资源的角度来看。在某些情况下，可能无法快速找到一个好的解决方案。

这个问题被称为*维度诅咒*，无监督学习非常适合帮助管理这一问题。通过降维，我们可以找到原始特征集中最显著的特征，将维度减少到一个更易管理的数量，同时在过程中几乎不丢失重要信息，然后应用监督算法来更有效地执行寻找良好函数逼近的搜索。我们将在第三章涵盖降维技术。

### 特征工程

特征工程是数据科学家执行的最关键任务之一。如果没有合适的特征，机器学习算法将无法在空间中有效分离点，从而不能在以前未见的示例上做出良好的决策。然而，特征工程通常非常耗时，需要人类创造性地手工设计正确类型的特征。相反，我们可以使用无监督学习算法中的表示学习来自动学习适合解决手头任务的正确类型的特征表示。我们将在第七章探索自动特征提取。

### 异常值

数据的质量也非常重要。如果机器学习算法在稀有的、扭曲的异常值上进行训练，其泛化误差将低于忽略或单独处理异常值的情况。通过无监督学习，我们可以使用降维技术进行异常检测，并分别为异常数据和正常数据创建解决方案。我们将在第四章构建一个异常检测系统。

### 数据漂移

机器学习模型还需要意识到数据中的漂移。如果模型用于预测的数据在统计上与模型训练时的数据不同，那么模型可能需要在更能代表当前数据的数据上重新训练。如果模型不重新训练或者没有意识到这种漂移，那么模型在当前数据上的预测质量将会受到影响。

通过使用无监督学习构建概率分布，我们可以评估当前数据与训练集数据的差异性——如果两者差异足够大，我们可以自动触发重新训练。我们将探讨如何构建这些数据判别器类型的内容在第十二章中。

# 对监督算法的更详细探讨

在我们深入研究无监督学习系统之前，让我们先看看监督学习算法及其工作原理。这将有助于我们理解无监督学习在机器学习生态系统中的位置。

在监督学习中，存在两种主要类型的问题：*分类*和*回归*。在分类中，AI 必须正确地将项目分类为两个或更多类别之一。如果只有两个类别，则该问题称为*二元分类*。如果有三个或更多类别，则该问题被归类为*多类分类*。

分类问题也被称为*离散*预测问题，因为每个类别都是一个离散的群体。分类问题也可能被称为*定性*或*分类*问题。

在回归中，AI 必须预测一个*连续*变量而不是离散变量。回归问题也可能被称为*定量*问题。

监督式机器学习算法涵盖了从非常简单到非常复杂的整个范围，但它们的目标都是最小化与数据集标签相关的某个成本函数或错误率（或最大化某个值函数）。

正如前面提到的，我们最关心的是机器学习解决方案在前所未见的情况下的泛化能力。选择监督学习算法非常重要，可以最大程度地减少这种泛化误差。

为了达到尽可能低的泛化误差，算法模型的复杂性应该与数据底层真实函数的复杂性相匹配。我们不知道这个真实函数究竟是什么。如果我们知道，我们就不需要使用机器学习来创建模型了——我们只需解决函数以找到正确答案。但由于我们不知道这个真实函数是什么，我们选择机器学习算法来测试假设，并找到最接近这个真实函数的模型（即具有尽可能低的泛化误差）。

如果算法模拟的内容比真实函数复杂度低，我们就*欠拟合*了数据。在这种情况下，我们可以通过选择能够模拟更复杂函数的算法来改善泛化误差。然而，如果算法设计了一个过于复杂的模型，我们就*过拟合*了训练数据，并且在以前从未见过的情况下表现不佳，增加了我们的泛化误差。

换句话说，选择复杂算法而不是简单算法并不总是正确的选择——有时简单才是更好的。每种算法都有其一系列的优点、弱点和假设，知道在给定你拥有的数据和你试图解决的问题时何时使用何种方法对于掌握机器学习非常重要。

在本章的其余部分中，我们将描述一些最常见的监督学习算法（包括一些实际应用），然后再介绍无监督算法。³

## 线性方法

最基本的监督学习算法模拟了输入特征与我们希望预测的输出变量之间的简单线性关系。

### 线性回归

所有算法中最简单的是*线性回归*，它使用一个模型假设输入变量（x）与单个输出变量（y）之间存在线性关系。如果输入与输出之间的真实关系是线性的，并且输入变量之间不高度相关（称为*共线性*），线性回归可能是一个合适的选择。如果真实关系更为复杂或非线性，线性回归将会欠拟合数据。⁴

因为它非常简单，解释算法模型的关系也非常直接。*可解释性* 对于应用机器学习非常重要，因为解决方案需要被技术和非技术人员在工业中理解和实施。如果没有可解释性，解决方案就会变成不可理解的黑匣子。

优点

线性回归简单、可解释，并且难以过拟合，因为它无法模拟过于复杂的关系。当输入和输出变量之间的基础关系是线性的时，它是一个极好的选择。

弱点

当输入和输出变量之间的关系是非线性的时，线性回归将欠拟合数据。

应用

由于人类体重与身高之间的真实基础关系是线性的，因此线性回归非常适合使用身高作为输入变量来预测体重，或者反过来，使用体重作为输入变量来预测身高。

### 逻辑回归

最简单的分类算法是 *逻辑回归*，它也是一种线性方法，但预测结果经过逻辑函数转换。这种转换的输出是*类别概率*——换句话说，实例属于各个类别的概率，每个实例的概率之和为一。然后将每个实例分配给其最有可能属于的类别。

优势

与线性回归类似，逻辑回归简单且可解释。当我们尝试预测的类别不重叠且线性可分时，逻辑回归是一个很好的选择。

弱点

当类别不是线性可分时，逻辑回归会失败。

应用场景

当类别大部分不重叠时，例如年幼儿童的身高与成年人的身高，逻辑回归效果很好。

## 基于邻居的方法

另一组非常简单的算法是基于邻居的方法。基于邻居的方法是*惰性学习器*，因为它们学习如何根据新点与现有标记点的接近程度来标记新点。与线性回归或逻辑回归不同，基于邻居的模型不会学习一个固定的模型来预测新点的标签；相反，这些模型仅基于新点到预先标记点的距离来预测新点的标签。惰性学习也称为*基于实例的学习*或*非参数方法*。

### k 近邻算法

最常见的基于邻居的方法是 *k 近邻算法 (KNN)*。为了给每个新点贴上标签，KNN 查看 *k* 个最近的已标记点（其中 *k* 是整数），并让这些已标记的邻居投票决定如何给新点贴标签。默认情况下，KNN 使用欧氏距离来衡量最近的点。

*k* 的选择非常重要。如果 *k* 设置得非常低，KNN 变得非常灵活，可能会绘制非常微妙的边界并可能过度拟合数据。如果 *k* 设置得非常高，KNN 变得不够灵活，绘制出过于刚性的边界，可能会欠拟合数据。

优势

不同于线性方法，KNN 非常灵活，能够学习更复杂、非线性的关系。尽管如此，KNN 仍然简单且可解释。

弱点

当观测数和特征数量增加时，KNN 的表现较差。在这种高度密集且高维的空间中，KNN 变得计算效率低下，因为它需要计算新点到许多附近已标记点的距离，以预测标签。它无法依靠具有减少参数数量的高效模型进行必要的预测。此外，KNN 对 *k* 的选择非常敏感。当 *k* 设置过低时，KNN 可能过拟合；当 *k* 设置过高时，KNN 可能欠拟合。

应用场景

KNN 经常被用于推荐系统，比如用来预测电影品味（Netflix）、音乐喜好（Spotify）、朋友（Facebook）、照片（Instagram）、搜索（Google）和购物（Amazon）。例如，KNN 可以帮助预测用户会喜欢什么，基于类似用户喜欢的东西（称为*协同过滤*）或者用户过去喜欢的东西（称为*基于内容的过滤*）。

## 基于树的方法

而不是使用线性方法，我们可以让 AI 构建*决策树*，在这些实例中所有的实例都被*分割*或*分层*成许多区域，这些区域由我们的标签引导。一旦完成这种分割，每个区域对应于一个特定的标签类别（用于分类问题）或预测值范围（用于回归问题）。这个过程类似于让 AI 自动构建规则，其明确目标是做出更好的决策或预测。

### 单一决策树

最简单的基于树的方法是*单一决策树*，在这种方法中，AI 一次通过训练数据，根据标签创建数据分割规则，并使用这棵树对从未见过的验证或测试集进行预测。然而，单一决策树通常在将其在训练期间学到的内容推广到从未见过的情况时表现不佳，因为它通常在其唯一的训练迭代期间过拟合训练数据。

### 装袋

要改进单一决策树，我们可以引入*自助聚合*（更常被称为*装袋*），其中我们从训练数据中取多个随机样本实例，为每个样本创建一个决策树，然后通过平均这些树的预测来预测每个实例的输出。通过*随机化*样本和对多个树的预测结果进行平均——这种方法也被称为*集成方法*——装袋将解决由单一决策树导致的过拟合问题的一些方面。

### 随机森林

我们可以通过对预测变量进行采样来进一步改善过拟合。通过*随机森林*，我们像在装袋中那样从训练数据中取多个随机样本实例，但是，在每个决策树的每次分割中，我们基于*预测变量的随机样本*而不是所有预测变量进行分割。每次分割考虑的预测变量数量通常是总预测变量数量的平方根。

通过这种方式对预测变量进行采样，随机森林算法创建的树与彼此更少相关（与装袋中的树相比），从而减少过拟合并改善泛化误差。

### 提升法

另一种称为*提升*的方法用于创建多棵树，类似于装袋法，但是*顺序构建树*，使用 AI 从前一棵树学到的知识来改进后续树的结果。每棵树保持相当浅，只有几个决策分裂点，并且学习是逐步进行的，树与树之间逐步增强。在所有基于树的方法中，*梯度提升机*是表现最佳的，并且常用于赢得机器学习竞赛。⁵

优点

基于树的方法是预测问题中表现最佳的监督学习算法之一。这些方法通过逐步学习许多简单规则来捕捉数据中的复杂关系。它们还能够处理缺失数据和分类特征。

弱点

基于树的方法很难解释，特别是如果需要许多规则来做出良好的预测。随着特征数量的增加，性能也成为一个问题。

应用

梯度提升和随机森林在预测问题上表现出色。

## 支持向量机

我们可以使用算法在空间中创建超平面来分隔数据，这些算法由我们拥有的标签引导。这种方法被称为*支持向量机（SVMs）*。 SVMs 允许在这种分隔中存在一些违规情况——并非超空间中的所有点都必须具有相同的标签——但某一标签的边界定义点与另一标签的边界定义点之间的距离应尽可能最大化。此外，边界不一定是线性的——我们可以使用非线性核来更灵活地分隔数据。

## 神经网络

我们可以使用神经网络来学习数据的表示，神经网络由输入层、多个隐藏层和输出层组成。⁶ 输入层使用特征，输出层试图匹配响应变量。隐藏层是一个嵌套的概念层次结构——每个层（或概念）都试图理解前一层如何与输出层相关联。

使用这种概念层次结构，神经网络能够通过将简单的概念组合起来来学习复杂的概念。神经网络是函数逼近中最强大的方法之一，但容易过拟合且难以解释，我们将在本书后面更详细地探讨这些缺点。

# 深入探讨无监督算法

现在我们将注意力转向没有标签的问题。无监督学习算法将尝试学习数据的潜在结构，而不是尝试进行预测。

## 降维

一类算法——称为*降维算法*——将原始高维输入数据投影到低维空间，滤除不那么相关的特征并保留尽可能多的有趣特征。降维允许无监督学习 AI 更有效地识别模式，并更高效地解决涉及图像、视频、语音和文本的大规模计算问题。

### 线性投影

维度的两个主要分支是线性投影和非线性降维。我们将首先从线性投影开始。

#### 主成分分析（PCA）

学习数据的基本结构的一种方法是确定在完整特征集中哪些特征对解释数据实例之间变异性最重要。并非所有特征都是相等的——对于某些特征，数据集中的值变化不大，这些特征在解释数据集方面不那么有用。对于其他特征，其值可能会有显著变化——这些特征值得更详细探讨，因为它们将更有助于我们设计的模型分离数据。

在*PCA*中，该算法在保留尽可能多的变化的同时找到数据的低维表示。我们得到的维度数量远远小于完整数据集的维度数（即总特征数）。通过转移到这个低维空间，我们会失去一些方差，但数据的基本结构更容易识别，这样我们可以更有效地执行诸如聚类之类的任务。

PCA 有几种变体，我们将在本书后面探讨。这些包括小批量变体，如*增量 PCA*，非线性变体，如*核 PCA*，以及稀疏变体，如*稀疏 PCA*。

#### 奇异值分解（SVD）

学习数据的基本结构的另一种方法是将原始特征矩阵的秩降低到一个较小的秩，使得可以用较小秩矩阵中某些向量的线性组合来重建原始矩阵。这就是*SVD*。为了生成较小秩矩阵，SVD 保留具有最多信息（即最高奇异值）的原始矩阵向量。较小秩矩阵捕捉了原始特征空间的最重要元素。

#### 随机投影

类似的降维算法涉及将高维空间中的点投影到远低于其维度的空间中，以保持点之间的距离比例。我们可以使用*随机高斯矩阵*或*随机稀疏矩阵*来实现这一点。

### 流形学习

PCA 和随机投影都依赖于将数据从高维空间线性投影到低维空间。与线性投影不同，执行数据的非线性变换可能更好——这被称为*流形学习*或*非线性降维*。

#### Isomap

*Isomap*是一种流形学习方法。该算法通过估计每个点及其邻居之间的*测地线*或*曲线距离*而不是欧氏距离来学习数据流形的内在几何结构。Isomap 将此用于将原始高维空间嵌入到低维空间。

#### t-分布随机近邻嵌入（t-SNE）

另一种非线性降维方法——称为*t-SNE*——将高维数据嵌入到仅具有两个或三个维度的空间中，使得转换后的数据可以可视化。在这个二维或三维空间中，相似的实例被建模为更接近，而不相似的实例被建模为更远。

#### 字典学习

一种被称为*字典学习*的方法涉及学习底层数据的稀疏表示。这些代表性元素是简单的二进制向量（零和一），数据集中的每个实例都可以重构为代表性元素的加权和。这种无监督学习生成的矩阵（称为*字典*）大多数由零填充，只有少数非零权重。

通过创建这样一个字典，该算法能够有效地识别原始特征空间中最显著的代表性元素——这些元素具有最多的非零权重。不太重要的代表性元素将具有较少的非零权重。与 PCA 一样，字典学习非常适合学习数据的基本结构，这对于分离数据和识别有趣的模式将会有所帮助。

### 独立分量分析

无标签数据的一个常见问题是，许多独立信号被嵌入到我们所获得的特征中。使用*独立分量分析（ICA）*，我们可以将这些混合信号分离成它们的个体组成部分。分离完成后，我们可以通过将我们生成的个体组成部分的某种组合相加来重构任何原始特征。ICA 在信号处理任务中通常用于（例如，识别繁忙咖啡馆音频剪辑中的个别声音）。

### 潜在狄利克雷分配

无监督学习还可以通过学习为什么数据集的某些部分相互类似来解释数据集。这需要学习数据集中的未观察元素——一种被称为*潜在狄利克雷分配（LDA）*的方法。例如，考虑一个文本文档，其中有许多词。文档内的这些词并非纯粹随机；相反，它们呈现出一定的结构。

这种结构可以建模为称为主题的未观察元素。经过训练后，LDA 能够用一小组主题解释给定的文档，每个主题都有一小组经常使用的单词。这是 LDA 能够捕捉的隐藏结构，帮助我们更好地解释以前结构不清晰的文本语料库。

###### 注意

降维将原始特征集合减少到仅包含最重要的特征集合。然后，我们可以在这些较小的特征集上运行其他无监督学习算法，以发现数据中的有趣模式（参见下一节关于聚类的内容），或者如果有标签，我们可以通过向这些较小的特征矩阵输入来加快监督学习算法的训练周期，而不是使用原始特征矩阵。

## 聚类

一旦我们将原始特征集减少到一个更小、更易处理的集合，我们可以通过将相似的数据实例分组来找到有趣的模式。这被称为聚类，可以使用各种无监督学习算法来实现，并且可用于市场细分等现实应用中。

### k-means

要进行良好的聚类，我们需要识别出不同的群组，使得群组内的实例彼此相似，但与其他群组内的实例不同。其中一种算法是*k-means 聚类*。使用这种算法，我们指定所需的群组数量*k*，算法将每个实例分配到这*k*个群组中的一个。它通过最小化*群内变异性*（也称为*惯性*）来优化分组，使得所有*k*个群组内的群内变异性之和尽可能小。

为了加速这一聚类过程，*k*-means 随机将每个观测分配到*k*个群组中的一个，然后开始重新分配这些观测，以最小化每个观测与其群组中心点或*质心*之间的欧氏距离。因此，不同运行的*k*-means（每次都从随机起点开始）将导致略有不同的观测聚类分配。从这些不同的运行中，我们可以选择具有最佳分离性能的运行，即所有*k*个群组内的总群内变异性之和最低的运行。⁷

### 层次聚类

另一种聚类方法——不需要预先确定特定群组数量的方法被称为*层次聚类*。层次聚类的一种版本称为*聚合聚类*，使用基于树的聚类方法，并构建所谓的*树状图*。树状图可以以图形方式呈现为倒置的树，其中叶子位于底部，树干位于顶部。

在数据集中，最底部的叶子是个体实例。然后，按照它们彼此的相似程度，层次聚类将这些叶子连接在一起——随着我们在颠倒的树上向上移动。最相似的实例（或实例组）会更早地连接在一起，而不那么相似的实例则稍后连接。通过这个迭代过程，所有实例最终链接在一起，形成了树的单一主干。

这种垂直描绘非常有帮助。一旦层次聚类算法运行完成，我们可以查看树状图并确定我们想要切割树的位置——我们在树干上切割得越低，留下的个体分支就越多（即更多的簇）。如果我们想要更少的簇，我们可以在树状图的更高处切割，靠近这颠倒树的顶部的单一主干。这种垂直切割的位置类似于在*k*-means 聚类算法中选择*k*簇的数量。⁸

### DBSCAN

另一个更强大的聚类算法（基于点的密度）称为*DBSCAN*（具有噪声的基于密度的空间聚类应用程序）。给定我们在空间中的所有实例，DBSCAN 会将那些紧密聚集在一起的实例分组在一起，其中紧密定义为必须存在一定距离内的最小数量的实例。我们同时指定所需的最小实例数和距离。

如果一个实例在指定的距离内接近多个簇，则将其与其最密集的簇分组。任何不在另一个簇指定距离内的实例都被标记为异常值。

不像*k*-means 那样，我们不需要预先指定簇的数量。我们还可以拥有任意形状的簇。DBSCAN 在数据中典型的由异常值引起的扭曲问题上要少得多。

## 特征提取

通过无监督学习，我们可以学习数据原始特征的新表示——一个称为*特征提取*的领域。特征提取可用于将原始特征的数量减少到更小的子集，从而有效地执行降维。但是特征提取也可以生成新的特征表示，以帮助在监督学习问题上提高性能。

### Autoencoders

要生成新的特征表示，我们可以使用前馈、非循环神经网络进行表示学习，其中输出层中的节点数量与输入层中的节点数量相匹配。这种神经网络被称为*自编码器*，有效地重构原始特征，利用隐藏层之间的学习新的表示。⁹

自编码器的每个隐藏层学习原始特征的表示，后续层基于前面层学习的表示构建。逐层，自编码器从简单表示中学习越来越复杂的表示。

输出层是原始特征的最终新学习表示。这个学习表示然后可以用作监督学习模型的输入，目的是改善泛化误差。

### 使用前馈网络的监督训练进行特征提取

如果有标签，另一种特征提取方法是使用前馈非递归神经网络，其中输出层试图预测正确的标签。就像自编码器一样，每个隐藏层学习原始特征的表示。

然而，在生成新表示时，该网络明确地*由标签引导*。为了从这个网络中提取原始特征的最终新学习表示，我们提取倒数第二层——即输出层之前的隐藏层。然后，可以将这个倒数第二层用作任何监督学习模型的输入。

## 无监督深度学习

在深度学习领域，无监督学习执行许多重要功能，其中一些我们将在本书中探讨。这个领域被称为*无监督深度学习*。

直到最近，深度神经网络的训练在计算上是棘手的。在这些神经网络中，隐藏层学习内部表示来帮助解决手头的问题。这些表示会随着神经网络在每次训练迭代中如何使用误差函数的梯度来更新各个节点的权重而不断改进。

这些更新计算成本很高，过程中可能会出现两种主要类型的问题。首先，误差函数的梯度可能变得非常小，由于*反向传播*依赖于这些小权重的乘积，网络的权重可能更新非常缓慢，甚至根本不更新，从而阻止网络的正确训练。¹⁰ 这被称为*梯度消失问题*。

相反，另一个问题是误差函数的梯度可能变得非常大；通过反向传播，网络中的权重可能会大幅度地更新，使得网络的训练非常不稳定。这被称为*梯度爆炸问题*。

### 无监督预训练

为了解决训练非常深、多层神经网络的困难，机器学习研究人员采用多阶段训练神经网络的方法，每个阶段涉及一个浅层神经网络。一个浅层网络的输出被用作下一个神经网络的输入。通常，这个流水线中的第一个浅层神经网络涉及无监督神经网络，但后续的网络是有监督的。

这个无监督部分被称为*贪婪逐层无监督预训练*。2006 年，Geoffrey Hinton 展示了成功应用无监督预训练来初始化更深神经网络管道的情况，从而开启了当前的深度学习革命。无监督预训练使得 AI 能够捕获原始输入数据的改进表示，随后监督部分利用这些表示来解决手头的具体任务。

这种方法被称为“贪婪”，因为神经网络的每个部分都是独立训练的，而不是联合训练。 “逐层”指的是网络的各层。在大多数现代神经网络中，通常不需要预训练。相反，所有层都使用反向传播联合训练。主要的计算机进步使得梯度消失问题和梯度爆炸问题变得更加可管理。

无监督预训练不仅使监督问题更容易解决，还促进了*迁移学习*。迁移学习涉及使用机器学习算法将从解决一个任务中获得的知识存储起来，以更快速且需要更少数据的方式解决另一个相关任务。

### 受限玻尔兹曼机

无监督预训练的一个应用例子是*受限玻尔兹曼机（RBM）*，一个浅层的双层神经网络。第一层是输入层，第二层是隐藏层。每个节点与另一层的每个节点相连接，但节点与同一层的节点不连接——这就是约束的地方。

RBMs 可以执行无监督任务，如降维和特征提取，并作为监督学习解决方案的有用无监督预训练的一部分。RBMs 类似于自动编码器，但在某些重要方面有所不同。例如，自动编码器有一个输出层，而 RBM 则没有。我们将在本书的后续部分详细探讨这些及其他差异。

### 深度信念网络

RBMs 可以连接在一起形成多阶段神经网络管道，称为*深度信念网络（DBN）*。每个 RBM 的隐藏层被用作下一个 RBM 的输入。换句话说，每个 RBM 生成数据的表示，然后下一个 RBM 在此基础上构建。通过成功地链接这种表示学习，深度信念网络能够学习更复杂的表示，通常用作*特征检测器*。¹¹

### 生成对抗网络

无监督深度学习的一个重大进展是*生成对抗网络（GANs）*的出现，由 Ian Goodfellow 及其蒙特利尔大学的同事于 2014 年引入。GANs 有许多应用，例如，我们可以使用 GANs 创建接近真实的合成数据，如图像和语音，或执行异常检测。

在 GAN 中，我们有两个神经网络。一个网络——生成器——基于其创建的模型数据分布生成数据，该模型数据是通过接收的真实数据样本创建的。另一个网络——鉴别器——区分生成器创建的数据和真实数据分布的数据。

简单类比，生成器是伪造者，鉴别器是试图识别伪造品的警察。这两个网络处于零和博弈中。生成器试图欺骗鉴别器，让其认为合成数据来自真实数据分布，而鉴别器则试图指出合成数据是假的。

GAN（生成对抗网络）是无监督学习算法，因为生成器可以在没有标签的情况下学习真实数据分布的潜在结构。GAN 通过训练过程学习数据中的潜在结构，并使用少量可管理的参数高效捕捉这种结构。

这个过程类似于深度学习中的表征学习。生成器神经网络中的每个隐藏层通过从简单开始捕捉底层数据的表示，随后的层通过建立在较简单前层的基础上，捕捉更复杂的表示。

使用所有这些层，生成器学习数据的潜在结构，并利用所学，尝试创建几乎与真实数据分布相同的合成数据。如果生成器已经捕捉到真实数据分布的本质，合成数据将看起来是真实的。

## 使用无监督学习处理顺序数据问题

无监督学习也可以处理时间序列等顺序数据。一种方法涉及学习*马尔可夫模型*的隐藏状态。在*简单马尔可夫模型*中，状态完全可观察且随机变化（换句话说，随机）。未来状态仅依赖于当前状态，而不依赖于先前状态。

在*隐藏马尔可夫模型*中，状态仅部分可观察，但与简单马尔可夫模型一样，这些部分可观察状态的输出是完全可观察的。由于我们的观察不足以完全确定状态，我们需要无监督学习帮助更充分地发现这些隐藏状态。

隐藏马尔可夫模型算法涉及学习给定我们所知的先前发生的部分可观察状态和完全可观察输出的可能下一个状态。这些算法在涉及语音、文本和时间序列的顺序数据问题中具有重要的商业应用。

# 使用无监督学习进行强化学习

强化学习是机器学习的第三大主要分支，其中一个*代理人*根据它收到的*奖励*反馈，决定其在*环境*中的最佳行为（*actions*）。这种反馈称为*强化信号*。代理人的目标是随时间最大化其累积奖励。

尽管强化学习自 1950 年代以来就存在，但直到近年来才成为主流新闻头条。2013 年，现为谷歌所有的 DeepMind 应用强化学习实现了超越人类水平的表现，玩转多种不同的 Atari 游戏。DeepMind 的系统仅使用原始感官数据作为输入，并且没有关于游戏规则的先验知识。

2016 年，DeepMind 再次吸引了机器学习社区的想象力——这一次，基于强化学习的 AI 代理 AlphaGo 击败了李世石，世界顶级围棋选手之一。这些成功奠定了强化学习作为主流 AI 主题的地位。

如今，机器学习研究人员正在应用强化学习来解决许多不同类型的问题，包括：

+   股市交易中，代理人买卖股票（*actions*），并获得利润或损失（*rewards*）作为回报。

+   视频游戏和棋盘游戏中，代理人做出游戏决策（*actions*），并赢得或输掉（*rewards*）。

+   自动驾驶汽车中，代理人指导车辆（*actions*），并且要么保持在路线上，要么发生事故（*rewards*）。

+   机器控制中，代理人在其环境中移动（*actions*），并且要么完成任务，要么失败（*rewards*）。

在最简单的强化学习问题中，我们有一个有限问题——环境的状态有限，任何给定环境状态下可能的动作有限，并且奖励的数量也是有限的。在给定当前环境状态下，代理人采取的行动决定了下一个状态，代理人的目标是最大化其长期奖励。这类问题称为有限的*马尔可夫决策过程*。

然而，在现实世界中，事情并不那么简单——奖励是未知的和动态的，而不是已知的和静态的。为了帮助发现这个未知的奖励函数并尽可能地逼近它，我们可以应用无监督学习。利用这个近似的奖励函数，我们可以应用强化学习解决方案，以增加随时间累积的奖励。

# 半监督学习

尽管监督学习和无监督学习是机器学习的两个明显不同的主要分支，但每个分支的算法可以作为机器学习流水线的一部分混合在一起。¹² 通常，在我们想充分利用少数标签或者想从无标签数据中找到新的未知模式以及从标记数据中已知的模式时，我们会混合使用监督和无监督学习。这些类型的问题通过一种称为半监督学习的混合方式来解决。我们将在本书后续章节详细探讨这一领域。

# 无监督学习的成功应用

在过去的十年中，大多数成功的商业应用来自监督学习领域，但情况正在改变。无监督学习应用变得越来越普遍。有时，无监督学习只是改善监督应用的手段。其他时候，无监督学习本身就实现了商业应用。以下是迄今为止两个最大的无监督学习应用的更详细介绍：异常检测和群体分割。

## 异常检测

进行降维可以将原始的高维特征空间转化为一个转换后的低维空间。在这个低维空间中，我们找到了大多数点密集分布的地方。这部分被称为*正常空间*。远离这些点的点被称为*离群点*或*异常*，值得更详细地调查。

异常检测系统通常用于诸如信用卡欺诈、电汇欺诈、网络欺诈和保险欺诈等欺诈检测。异常检测还用于识别罕见的恶意事件，如对互联网连接设备的黑客攻击，对飞机和火车等关键设备的维护故障，以及由恶意软件和其他有害代理引起的网络安全漏洞。

我们可以将这些系统用于垃圾邮件检测，例如我们在本章前面使用的电子邮件垃圾过滤器示例。其他应用包括寻找如恐怖主义资金、洗钱、人口和毒品贩运以及军火交易等活动的不良行为，识别金融交易中的高风险事件，以及发现癌症等疾病。

为了使异常分析更加可管理，我们可以使用聚类算法将相似的异常分组在一起，然后基于它们所代表的行为类型手动标记这些聚类。通过这样的系统，我们可以拥有一个能够识别异常、将它们聚类到适当组中，并且利用人类提供的聚类标签向业务分析师推荐适当行动的无监督学习人工智能。

通过异常检测系统，我们可以将一个无监督问题逐步转换为半监督问题，通过这种集群和标记的方法。随着时间的推移，我们可以在未标记数据上运行监督算法，并行进行无监督算法。对于成功的机器学习应用程序，无监督系统和监督系统应该同时使用，相辅相成。

监督系统以高精度找到已知模式，而无监督系统发现可能感兴趣的新模式。一旦这些模式被无监督 AI 揭示，人类会对这些模式进行标记，将更多数据从未标记转换为已标记。

### 群体分割

通过聚类，我们可以根据行为相似性在市场营销、客户保持、疾病诊断、在线购物、音乐收听、视频观看、在线约会、社交媒体活动和文档分类等领域中对群体进行分割。在这些领域产生的数据量非常庞大，且数据只有部分被标记。

对于我们已经了解并希望加强的模式，我们可以使用监督学习算法。但通常我们希望发现新的模式和感兴趣的群体——对于这一发现过程，无监督学习是一个自然的选择。再次强调，这一切都是关于协同作用。我们应该同时使用监督和无监督学习系统来构建更强大的机器学习解决方案。

# 结论

在本章中，我们探讨了以下内容：

+   基于规则的系统和机器学习的区别

+   监督学习和无监督学习的区别

+   无监督学习如何帮助解决训练机器学习模型中的常见问题

+   常见的监督、无监督、强化和半监督学习算法

+   无监督学习的两个主要应用——异常检测和群体分割

在第二章中，我们将探讨如何构建机器学习应用程序。然后，我们将详细讨论降维和聚类，逐步构建异常检测系统和群体分割系统。

¹ 有像 Figure Eight 这样明确提供*人在循环*服务的初创企业。

² 欠拟合是在构建机器学习应用程序时可能出现的另一个问题，但这更容易解决。欠拟合是因为模型过于简单——算法无法构建足够复杂的函数逼近来为当前任务做出良好的决策。为了解决这个问题，我们可以允许算法增加规模（增加参数、执行更多训练迭代等）或者应用更复杂的机器学习算法。

³ 这个列表并非详尽无遗，但包含了最常用的机器学习算法。

⁴ 可能有其他潜在问题会使得线性回归成为一个不好的选择，包括异常值、误差项相关性以及误差项方差的非常数性。

⁵ 想要了解机器学习竞赛中梯度提升的更多信息，请查阅 Ben Gorman 的[博客文章](http://bit.ly/2S1C8Qy)。

⁶ 想要了解更多关于神经网络的信息，请参阅 Ian Goodfellow、Yoshua Bengio 和 Aaron Courville 的《*深度学习*》（MIT Press）。

⁷ *k*-均值聚类的快速变体包括小批量*k*-均值，我们稍后在书中进行介绍。

⁸ 分层聚类默认使用欧几里得距离，但也可以使用其他相似度度量，比如基于相关的距离，我们稍后在书中详细探讨。

⁹ 有几种类型的自编码器，每种都学习不同的表示。这些包括去噪自编码器、稀疏自编码器和变分自编码器，我们稍后在书中进行探讨。

¹⁰ 反向传播（也称为*误差反向传播*）是神经网络使用的基于梯度下降的算法，用于更新权重。在反向传播中，首先计算最后一层的权重，然后用于更新前面层的权重。这个过程一直持续到第一层的权重被更新。

¹¹ 特征检测器学习原始数据的良好表示，帮助分离不同的元素。例如，在图像中，特征检测器帮助分离鼻子、眼睛、嘴等元素。

¹² Pipeline 指的是一种机器学习解决方案的系统，这些解决方案依次应用以实现更大的目标。
