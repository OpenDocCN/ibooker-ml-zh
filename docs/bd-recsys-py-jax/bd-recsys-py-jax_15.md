# 第十二章：排名的训练

典型的 ML 任务通常预测单一结果，例如分类任务中属于正类的概率，或者回归任务中的期望值。而排名则提供物品集合的相对排序。这种任务是搜索结果或推荐中常见的，其中呈现的物品顺序很重要。在这些问题中，物品的分数通常不会直接显示给用户，而是通过物品的序数排名—可能是隐式地—呈现：列表顶部的物品编号低于下一个物品。

本章介绍了 ML 算法在训练过程中可以使用的各种损失函数。这些分数应该估计列表排序，使得与训练数据集中观察到的相关性排序相比，结果更接近的集合。在这里，我们将重点介绍概念和计算，这些内容将在下一章中投入使用。

# 推荐系统中的排名在哪里适用？

在我们深入讨论排名的损失函数细节之前，我们应该谈谈排名在整个推荐系统中的位置。典型的大规模推荐系统有一个检索阶段，在此阶段使用廉价函数将大量候选项收集到候选集中。通常，此检索阶段仅基于物品。例如，候选集可能包括用户最近消费或喜欢的物品相关的物品。或者如果新鲜度很重要，比如对于新闻数据，该集合可能包括最新的热门和相关物品。物品被收集到候选集后，我们对其物品应用排名。

另外，由于候选集通常比整个物品语料库小得多，我们可以使用更昂贵的模型和辅助特征来帮助排名。这些特征可以是用户特征或上下文特征。用户特征可以帮助确定物品对用户的有用性，例如最近消费物品的平均嵌入。上下文特征可以指示当前会话的详细信息，例如一天中的时间或用户最近键入的查询，这些特征区别于其他会话，并帮助确定相关物品。最后，我们有物品本身的表示，可以是从内容特征到学习嵌入的任何东西。

然后将用户、上下文和物品特征连接成一个特征向量，我们将用它来表示物品；然后一次评分所有候选项并对其排序。然后可能对排序集合应用额外的过滤业务逻辑，例如删除近似重复项或使排序集合中显示的物品类型更多样化。

在以下示例中，我们假设所有项目都可以由用户、上下文和项目特征的连接特征向量表示，并且模型可以简化为具有权重向量`W`的线性模型，该向量与项目向量点乘以获取用于排序项目的分数。这些模型可以推广为深度神经网络，但最终层的输出仍然是用于排序项目的标量。

现在我们已经为排序设定了上下文，让我们考虑通过向量表示的一组项目的排名方式。

# 学习排序

*学习排序*（LTR）是一种根据它们的相关性或重要性对排序列表进行评分的模型类型。这种技术是如何从检索的潜在原始输出到基于它们的相关性的排序列表的方法。

LTR 问题有三种主要类型：

点对点

模型将单独的文档视为孤立的并分配给它们一个分数或等级。任务变成了一个回归或分类问题。

逐对

模型在损失函数中同时考虑文档对。目标是尽量减少错误排序的对数。

列表

模型在损失函数中考虑整个文档列表。目标是找到整个列表的最佳排序。

# 训练 LTR 模型

LTR 模型的训练数据通常包括项目列表，每个项目都有一组特征和一个标签（或地面真实值）。这些特征可能包括有关项目本身的信息，而标签通常表示其相关性或重要性。例如，在我们的推荐系统中，我们有项目特征，在训练数据集中，标签将显示项目是否与用户相关。此外，LTR 模型有时会使用查询或用户特征。

训练过程是通过使用这些特征和标签来学习排名函数。然后将这些排名函数应用于检索到的项目。

让我们看一些这些模型是如何训练的例子。

## 用于排序的分类

将排名问题建模为多标签任务的一种方法是一种方法。训练集中出现的与用户关联的每个项目都是正面例子，而那些在外部的则是负面的。这实际上是在项目集合的规模上的多标签方法。网络可能具有每个项目特征作为输入节点的架构，然后还有一些用户特征。输出节点将与您希望标记的项目对应。

使用线性模型，如果<math alttext="upper X"><mi>X</mi></math>是项目向量，<math alttext="upper Y"><mi>Y</mi></math>是输出，我们学习<math alttext="upper W"><mi>W</mi></math>，其中<math alttext="s i g m o i d left-parenthesis upper W upper X right-parenthesis equals 1"><mrow><mi>s</mi><mi>i</mi><mi>g</mi><mi>m</mi><mi>o</mi><mi>i</mi><mi>d</mi><mo>(</mo><mi>W</mi><mi>X</mi><mo>)</mo><mo>=</mo><mn>1</mn></mrow></math>如果<math alttext="upper X"><mi>X</mi></math>是正面集中的项目；否则<math alttext="s i g m o i d left-parenthesis upper W upper X right-parenthesis equals 0"><mrow><mi>s</mi><mi>i</mi><mi>g</mi><mi>m</mi><mi>o</mi><mi>i</mi><mi>d</mi><mo>(</mo><mi>W</mi><mi>X</mi><mo>)</mo><mo>=</mo><mn>0</mn></mrow></math>。这对应于在 Optax 中的[二元交叉熵损失](https://oreil.ly/5Rd14)。

不幸的是，在这种设置中没有考虑项目的相对排序，因此由于每个项目都有一个 sigmoid 激活函数的损失函数，它不会很好地优化排名度量标准。实际上，这种排名仅仅是一个下游的*相关性模型*，只有帮助过滤在先前步骤中检索到的选项。

这种方法的另一个问题是，我们已经将训练集之外的所有内容标记为负面，但是用户可能从未看过一个新项目，这个新项目可能与查询相关，因此将这个新项目标记为负面是不正确的，因为它只是未观察到。

您可能已经意识到，排名需要考虑列表中的相对位置。让我们接着考虑这个问题。

## 用于排名的回归

排列一组项目最朴素的方法是简单地回归到类似 NDCG 或我们其他的个性化度量的排名。

在实践中，这通过将项目集条件化为一个查询来实现。例如，我们可以将问题表述为回归到 NDCG，将查询作为排名的上下文。此外，我们可以将查询作为嵌入上下文向量提供给一个前馈网络，该网络与集合中项目的特征串联并回归到 NDCG 值。

查询作为上下文是必需的，因为一组项目的排序可能依赖于查询。例如，键入搜索栏中的查询**`flowers`**。然后，我们期望一组最能代表花卉的项目出现在前面的结果中。这表明查询是评分函数的重要考虑因素。

对于线性模型，如果 <math alttext="upper X"><mi>X</mi></math> 是项目向量，<math alttext="upper Y"><mi>Y</mi></math> 是输出，则我们学习 <math alttext="upper W"><mi>W</mi></math> ，其中 <math alttext="upper W upper X left-parenthesis i right-parenthesis equals upper N upper D upper C upper G left-parenthesis i right-parenthesis"><mrow><mi>W</mi> <mi>X</mi> <mo>(</mo> <mi>i</mi> <mo>)</mo> <mo>=</mo> <mi>N</mi> <mi>D</mi> <mi>C</mi> <mi>G</mi> <mo>(</mo> <mi>i</mi> <mo>)</mo></mrow></math> 而 <math alttext="upper N upper D upper C upper G left-parenthesis i right-parenthesis"><mrow><mi>N</mi> <mi>D</mi> <mi>C</mi> <mi>G</mi> <mo>(</mo> <mi>i</mi> <mo>)</mo></mrow></math> 是项目 <math alttext="i"><mi>i</mi></math> 的 NDCG。在 Optax 中，可以使用 [L2 loss](https://oreil.ly/IHw-Z) 进行回归学习。

最终，这种方法旨在尝试学习导致个性化指标中得分更高的项目的潜在特征。不幸的是，这也未明确考虑项目的相对排序。这是一个相当严重的限制，我们稍后将考虑。

另一个考虑因素是：对于在前 *k* 个训练项目之外未排名的项目，我们该怎么办？我们给它们分配的排名基本上是随机的，因为我们不知道要给它们分配什么数字。因此，这种方法需要改进，我们将在下一节中探讨。

## 排名的分类和回归

假设我们有一个网页，比如一个在线书店，用户必须浏览并点击项目才能购买它们。对于这样的漏斗，我们可以将排名分为两部分。第一个模型可以预测在展示的一组项目中点击项目的概率。第二个模型可以在点击后进行条件化，并且可以是一个回归模型，估计项目的购买价格。

然后，一个完整的排名模型可以是两个模型的乘积。第一个模型计算在一组竞争项目中点击项目的概率。第二个模型计算被点击后购买的预期值。请注意，第一个和第二个模型可能具有不同的特征，这取决于用户所处的漏斗阶段。第一个模型可以访问竞争项目的特征，而第二个模型可能考虑到可能改变项目价值的运费和应用的折扣。因此，在这种情况下，利用不同的模型对漏斗的每个阶段进行建模是有利的，以便利用每个阶段存在的最多信息。

# WARP

介绍一种随机生成排名损失的可能方式在[“WSABIE: Scaling Up to Large Vocabulary Image Annotation”](https://oreil.ly/bagf-)由 Jason Weston 等人提出。该损失被称为*weighted approximate rank pairwise*（WARP）。在这个方案中，损失函数被分解为看起来像是一对一的损失。更确切地说，如果一个排名更高的项目没有一个分数大于较低排名项目的边界（任意选取为 1），我们对这对项目应用*hinge loss*。这看起来像下面这样：

<math alttext="m a x left-parenthesis 0 comma 1 minus s c o r e left-parenthesis p o s right-parenthesis plus s c o r e left-parenthesis n e g right-parenthesis right-parenthesis" display="block"><mrow><mi>m</mi> <mi>a</mi> <mi>x</mi> <mo>(</mo> <mn>0</mn> <mo>,</mo> <mn>1</mn> <mo>-</mo> <mi>s</mi> <mi>c</mi> <mi>o</mi> <mi>r</mi> <mi>e</mi> <mo>(</mo> <mi>p</mi> <mi>o</mi> <mi>s</mi> <mo>)</mo> <mo>+</mo> <mi>s</mi> <mi>c</mi> <mi>o</mi> <mi>r</mi> <mi>e</mi> <mo>(</mo> <mi>n</mi> <mi>e</mi> <mi>g</mi> <mo>)</mo> <mo>)</mo></mrow></math>

使用线性模型时，如果<math alttext="upper X Subscript p Baseline o s"><mrow><msub><mi>X</mi> <mi>p</mi></msub> <mi>o</mi> <mi>s</mi></mrow></math>是正向项目向量，而<math alttext="upper X Subscript n Baseline e g"><mrow><msub><mi>X</mi> <mi>n</mi></msub> <mi>e</mi> <mi>g</mi></mrow></math>是负向项目向量，则我们学习<math alttext="upper W"><mi>W</mi></math>，其中<math alttext="upper W upper X Subscript p Baseline o s minus upper W upper X Subscript n Baseline e g greater-than 1"><mrow><mi>W</mi> <msub><mi>X</mi> <mi>p</mi></msub> <mi>o</mi> <mi>s</mi> <mo>-</mo> <mi>W</mi> <msub><mi>X</mi> <mi>n</mi></msub> <mi>e</mi> <mi>g</mi> <mo>></mo> <mn>1</mn></mrow></math>。这种情况下的损失是[hinge loss](https://oreil.ly/88zk3)，其中预测输出为<math alttext="upper W upper X Subscript p Baseline o s minus upper W upper X Subscript n Baseline e g"><mrow><mi>W</mi> <msub><mi>X</mi> <mi>p</mi></msub> <mi>o</mi> <mi>s</mi> <mo>-</mo> <mi>W</mi> <msub><mi>X</mi> <mi>n</mi></msub> <mi>e</mi> <mi>g</mi></mrow></math>，目标为 1。

然而，为了弥补一个未观察到的项目可能不是真负向的事实，而只是未观察到的东西，我们计算了从负向集合中抽样找到违反所选对排序的次数。也就是说，我们计算了需要查找的次数，找到这样的情况：

<math alttext="s c o r e left-parenthesis n e g right-parenthesis greater-than s c o r e left-parenthesis p o s right-parenthesis minus 1" display="block"><mrow><mi>s</mi> <mi>c</mi> <mi>o</mi> <mi>r</mi> <mi>e</mi> <mo>(</mo> <mi>n</mi> <mi>e</mi> <mi>g</mi> <mo>)</mo> <mo>></mo> <mi>s</mi> <mi>c</mi> <mi>o</mi> <mi>r</mi> <mi>e</mi> <mo>(</mo> <mi>p</mi> <mi>o</mi> <mi>s</mi> <mo>)</mo> <mo>-</mo> <mn>1</mn></mrow></math>

然后，我们构建一个随着我们从未看过的项目（减去正向项目）中抽样找到违反负向的次数而单调递减的函数，并查找此次数的权重，并将损失乘以它。如果很难找到违反负向的情况，梯度应该较低，因为要么我们已经接近一个好的解决方案，要么该项目以前从未被用户显示为查询结果。

注意，当 CPU 是主要的计算形式来训练机器学习模型时，WARP 损失被开发出来。因此，使用了一个排名的近似值来获得负项的排名。*近似排名*被定义为在我们找到一个负项其分数比正项大的任意常数边界 1.0 之前，从项目宇宙中（减去正例）抽样的次数。

要构建成对损失的 WARP 权重，我们需要一个函数，将负项的近似排名转换为 WARP 权重。计算这个相对简单的代码片段如下：

```py
import numpy as np

def get_warp_weights(n: int) -> np.ndarray:
  """Returns N weights to convert a rank to a loss weight."""

  # The alphas are defined as values that are monotonically decreasing.
  # We take the reciprocal of the natural numbers for the alphas.
  rank = np.arange(1.0, n + 1, 1)
  alpha = 1.0 / rank
  weights = alpha

  # This is the L in the paper, defined as the sum of all previous alphas.
  for i in range(1, n):
    weights[i] = weights[i] + weights[i -1]

  # Divide by the rank.
  weights = weights / rank
  return weights

print(get_warp_weights(5))
[1.         0.75       0.61111111 0.52083333 0.45666667]
```

如您所见，如果我们立即找到一个负样本，那么 WARP 权重为 1.0，但是如果很难找到违反间距的负样本，那么 WARP 权重将很小。

此损失函数大致优化了 precision@*k*，因此是改进检索集中排名估计的良好步骤。更好的是，通过采样，WARP 在计算上是高效的，因此更节省内存。

# k 阶统计量

有没有办法改进 WARP 损失和直接成对铰链损失？事实证明，有整个一系列方法。在[“使用 k 阶统计量损失学习排序推荐”](https://oreil.ly/afphG)，Jason Weston 等人（包括本书的其中一位合著者）展示了如何通过探索铰链损失和 WARP 损失之间的变体来完成这一点。本文的作者在各种语料库上进行了实验，并展示了在优化单个成对与选择像 WARP 这样的更难的负样本之间的权衡如何影响包括平均排名和 precision 和 recall 在*k*上的度量。

关键的一般化是，在梯度步骤期间不是考虑单个正项目，而是使用所有正项目。

再次回顾，随机选择一个正样本和一个负样本对优化 ROC 或 AUC。这对排名不是很好，因为它不会优化列表的顶部。另一方面，WARP 损失会优化单个正项目的排名列表的顶部，但不会指定如何选择正项目。

可以使用几种备选策略来对列表顶部进行排序，包括优化均值最大排名，该策略试图将正项目分组，使得得分最低的正项目尽可能靠近列表顶部。为了允许这种排序，我们提供了一个概率分布函数，用于解释我们如何选择正样本。如果概率偏向于正项目列表的顶部，我们会得到类似于 WARP 损失的损失。如果概率是均匀的，我们会得到 AUC 损失。如果概率偏向于正项目列表的末尾，那么我们将优化最坏情况，就像均值最大排名一样。NumPy 函数`np.random.choice`提供了一种从分布中进行采样的机制<math alttext="upper P"><mi>P</mi></math> 。

我们还有一个优化考虑： <math alttext="upper K"><mi>K</mi></math> ，用于构建正样本集的正样本数量。如果 <math alttext="upper K equals 1"><mrow><mi>K</mi> <mo>=</mo> <mn>1</mn></mrow></math> ，我们只从正样本集中随机选择一个正样本；否则，我们按分数对样本进行排序，并使用概率分布 <math alttext="upper P"><mi>P</mi></math> 从大小为 <math alttext="upper K"><mi>K</mi></math> 的正列表中采样。这种优化在 CPU 时代是有意义的，因为计算成本昂贵，但在 GPU 和 TPU 时代可能不再那么合理，接下来我们会在下面的警告中讨论这一点。

# 随机损失和 GPU

关于上述随机损失需要注意的一点。它们是为早期 CPU 时代开发的，那时对样本进行抽样并且在发现负样本时退出是廉价且简单的。而在现代 GPU 时代，做出类似这样的分支决策更加困难，因为 GPU 核心上的所有线程必须并行运行相同的代码，但在不同的数据上。这通常意味着分支的两侧都会在一个批次中执行，因此这些早期退出的计算节省效果较少。因此，像 WARP 和*k*阶统计损失这样的近似随机损失的分支代码看起来不那么高效。

我们该怎么办？我们将在第十三章中展示如何在代码中近似这些损失。长话短说，由于像 GPU 这样的向量处理器通常通过并行均匀地处理大量数据来工作，我们必须找到一种适合 GPU 的方式来计算这些损失。在下一章中，我们通过生成大批负样本并且要么将它们全部评分低于负样本，要么寻找最明显违反负样本，或者两者混合作为损失函数的一部分来近似负采样。

# BM25

尽管这本书的大部分内容都是针对向用户推荐物品，但搜索排名是一个紧密相关的研究领域。在信息检索或文档搜索排名空间中，*最佳匹配 25*（BM25）是一个必不可少的工具。

BM25 是信息检索系统中用于根据其与给定查询的相关性对文档进行排名的算法。这种相关性是通过考虑诸如 TF-IDF 等因素来确定的。它是一个基于词袋模型的检索函数，根据每个文档中出现的查询词来对一组文档进行排名。它还是概率相关性框架的一部分，并且源自概率检索模型。

BM25 排名函数根据查询为每个文档计算一个分数。得分最高的文档被认为与查询最相关。

这是 BM25 公式的简化版本：

<math alttext="left-brace score right-brace left-parenthesis upper D comma upper Q right-parenthesis equals sigma-summation Underscript i equals 1 Overscript n Endscripts left-brace IDF right-brace left-parenthesis q Subscript i Baseline right-parenthesis asterisk StartStartFraction f left-parenthesis q Subscript i Baseline comma upper D right-parenthesis asterisk left-parenthesis k Baseline 1 plus 1 right-parenthesis OverOver f left-parenthesis q Subscript i Baseline comma upper D right-parenthesis plus k 1 asterisk left-parenthesis 1 minus b plus b asterisk StartFraction StartAbsoluteValue upper D EndAbsoluteValue Over left-brace avgdl right-brace EndFraction right-parenthesis EndEndFraction" display="block"><mrow><mtext>score</mtext> <mrow><mo>(</mo> <mi>D</mi> <mo>,</mo> <mi>Q</mi> <mo>)</mo></mrow> <mo>=</mo> <munderover><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>n</mi></munderover> <mtext>IDF</mtext> <mrow><mo>(</mo> <msub><mi>q</mi> <mi>i</mi></msub> <mo>)</mo></mrow> <mo>*</mo> <mfrac><mrow><mi>f</mi><mrow><mo>(</mo><msub><mi>q</mi> <mi>i</mi></msub> <mo>,</mo><mi>D</mi><mo>)</mo></mrow><mo>*</mo><mrow><mo>(</mo><mi>k</mi><mn>1</mn><mo>+</mo><mn>1</mn><mo>)</mo></mrow></mrow> <mrow><mi>f</mi><mrow><mo>(</mo><msub><mi>q</mi> <mi>i</mi></msub> <mo>,</mo><mi>D</mi><mo>)</mo></mrow><mo>+</mo><mi>k</mi><mn>1</mn><mo>*</mo><mrow><mo>(</mo><mn>1</mn><mo>-</mo><mi>b</mi><mo>+</mo><mi>b</mi><mo>*</mo><mfrac><mrow><mo>|</mo><mi>D</mi><mo>|</mo></mrow> <mtext>avgdl</mtext></mfrac><mo>)</mo></mrow></mrow></mfrac></mrow></math>

这个公式的要素如下：

+   <math alttext="upper D"><mi>D</mi></math> 代表一个文档。

+   <math alttext="upper Q"><mi>Q</mi></math> 是由词组成的查询，包括单词 <math alttext="StartSet q 1 comma q 2 comma period period period comma q Subscript n Baseline EndSet"><mfenced close="}" open="{" separators=""><msub><mi>q</mi> <mn>1</mn></msub> <mo>,</mo> <msub><mi>q</mi> <mn>2</mn></msub> <mo>,</mo> <mo>.</mo> <mo>.</mo> <mo>.</mo> <mo>,</mo> <msub><mi>q</mi> <mi>n</mi></msub></mfenced></math> 。

+   <math alttext="f left-parenthesis q i comma upper D right-parenthesis"><mrow><mi>f</mi> <mo>(</mo> <mi>q</mi> <mi>i</mi> <mo>,</mo> <mi>D</mi> <mo>)</mo></mrow></math> 是查询项 <math alttext="q Subscript i"><msub><mi>q</mi> <mi>i</mi></msub></math> 在文档 <math alttext="upper D"><mi>D</mi></math> 中的频率。

+   <math alttext="StartAbsoluteValue upper D EndAbsoluteValue"><mrow><mo>|</mo> <mi>D</mi> <mo>|</mo></mrow></math> 是文档 <math alttext="upper D"><mi>D</mi></math> 的长度（单词数）。

+   <math alttext="a v g Subscript d Baseline l"><mrow><mi>a</mi> <mi>v</mi> <msub><mi>g</mi> <mi>d</mi></msub> <mi>l</mi></mrow></math> 是集合中的平均文档长度。

+   <math alttext="k 1"><msub><mi>k</mi> <mn>1</mn></msub></math> 和 <math alttext="b"><mi>b</mi></math> 是超参数。 <math alttext="k 1"><msub><mi>k</mi> <mn>1</mn></msub></math> 是正调节参数，用于校准文档词频的缩放。 <math alttext="b"><mi>b</mi></math> 是通过文档长度决定缩放的参数： <math alttext="b equals 1"><mrow><mi>b</mi> <mo>=</mo> <mn>1</mn></mrow></math> 对应完全按文档长度缩放词项权重，而 <math alttext="b equals 0"><mrow><mi>b</mi> <mo>=</mo> <mn>0</mn></mrow></math> 则表示不进行长度归一化。

+   <math alttext="upper I upper D upper F left-parenthesis q Subscript i Baseline right-parenthesis"><mrow><mi>I</mi> <mi>D</mi> <mi>F</mi> <mo>(</mo> <msub><mi>q</mi> <mi>i</mi></msub> <mo>)</mo></mrow></math> 是查询项 <math alttext="q Subscript i"><msub><mi>q</mi> <mi>i</mi></msub></math> 的逆文档频率，用于衡量单词在所有文档中提供的信息量（无论其在文档中是常见还是罕见）。BM25 应用了一种变体的 IDF，可以计算如下：

    <math alttext="left-brace IDF right-brace left-parenthesis q Subscript i Baseline right-parenthesis equals log left-parenthesis StartFraction upper N minus n left-parenthesis q Subscript i Baseline right-parenthesis plus 0.5 Over n left-parenthesis q Subscript i Baseline right-parenthesis plus 0.5 EndFraction right-parenthesis" display="block"><mrow><mtext>IDF</mtext> <mrow><mo>(</mo> <msub><mi>q</mi> <mi>i</mi></msub> <mo>)</mo></mrow> <mo>=</mo> <mo form="prefix">log</mo> <mfenced close=")" open="(" separators=""><mfrac><mrow><mi>N</mi><mo>-</mo><mi>n</mi><mo>(</mo><msub><mi>q</mi> <mi>i</mi></msub> <mo>)</mo><mo>+</mo><mn>0</mn><mo>.</mo><mn>5</mn></mrow> <mrow><mi>n</mi><mo>(</mo><msub><mi>q</mi> <mi>i</mi></msub> <mo>)</mo><mo>+</mo><mn>0</mn><mo>.</mo><mn>5</mn></mrow></mfrac></mfenced></mrow></math>

    这里，<math alttext="upper N"><mi>N</mi></math> 是集合中的文档总数，<math alttext="n left-parenthesis q Subscript i Baseline right-parenthesis"><mrow><mi>n</mi> <mo>(</mo> <msub><mi>q</mi> <mi>i</mi></msub> <mo>)</mo></mrow></math> 是包含查询项 <math alttext="q Subscript i"><msub><mi>q</mi> <mi>i</mi></msub></math> 的文档数。

简单来说，BM25 结合了词项频率（术语在文档中出现的频率）和逆文档频率（术语提供的唯一信息量大小）来计算相关性分数。它还引入了文档长度归一化的概念，惩罚过长的文档，并防止它们在简短文档面前占据主导地位，这是简单 TF-IDF 模型中常见的问题。自由参数 <math alttext="k 1"><msub><mi>k</mi> <mn>1</mn></msub></math> 和 <math alttext="b"><mi>b</mi></math> 允许根据文档集的特定特性进行调整。

在实践中，BM25 为大多数信息检索任务提供了强大的基线，包括即时关键字搜索和文档相似性。BM25 被许多开源搜索引擎（如 Lucene 和 Elasticsearch）使用，并成为通常所称的*全文搜索*的事实标准。

那么我们如何将 BM25 集成到本书中讨论的问题中呢？BM25 的输出是根据给定查询排名的文档列表，然后 LTR 发挥作用。您可以将 BM25 分数作为 LTR 模型中的一个特征，以及您认为可能影响文档与查询相关性的其他特征。

将 BM25 与 LTR 结合进行排名的一般步骤如下：

1.  *检索候选文档列表*。给定一个查询，使用 BM25 来检索候选文档列表。

1.  *为每个文档计算特征*。计算 BM25 分数作为其中一个特征，以及其他潜在的特征。这可能包括各种文档特定特征、查询-文档匹配特征、用户交互特征等。

1.  *训练/评估 LTR 模型*。使用这些特征向量及其对应的标签（相关性评判）来训练您的 LTR 模型。或者，如果您已经有了训练好的模型，可以使用它来评估和排名检索到的文档。

1.  *排名*。LTR 模型为每个文档生成一个分数。根据这些分数对文档进行排名。

通过使用 BM25 进行检索和 LTR 进行排名的组合，您可以首先从可能非常庞大的文档集合中缩小潜在的候选文档范围（其中 BM25 表现突出），然后利用一个可以考虑更复杂特征和交互的模型来微调这些候选文档的排名（其中 LTR 表现突出）。

值得一提的是，BM25 分数可以为文本文档检索提供强大的基线，取决于问题的复杂性和您拥有的训练数据量，LTR 可能会或者不会提供显著的改进。

# 多模态检索

让我们重新审视一下这种检索方法，因为我们可以找到一些强大的优势。回想一下第八章：我们构建了一个共现模型，展示了在其他文章中共同引用的文章如何共享含义和相互相关性。但是，您如何将搜索集成到这个过程中呢？

你可能会想，“哦，我可以搜索文章的名称。” 但这并没有充分利用我们的共现模型；它未充分利用我们发现的联合含义。一个经典的方法可能是在文章标题或文章上使用类似 BM25 的东西。更现代的方法可能会对查询和文章标题进行向量嵌入（使用类似 BERT 或其他变换器模型的东西）。然而，这两者都没有真正捕捉到我们寻找的两面。

考虑改用以下方法：

1.  通过 BM25 使用初始查询进行搜索，以获取初始的“锚点”集。

1.  通过您的潜在模型以每个锚点作为查询进行搜索。

1.  训练一个 LTR 模型来聚合和排名搜索的并集。

现在我们正在使用真正的多模式检索，利用多个潜在空间！这种方法的一个额外亮点是查询通常在基于编码器的潜在空间中与文档的分布不同。这意味着当你输入**`谁是莫桑比克的领导人？`**时，这个问题看起来与文章标题（莫桑比克）或 2023 年夏季相关的句子（“萨莫拉·马歇尔总统领导下的新政府建立了一个基于马克思主义原则的一党制国家。”）相当不同。

当嵌入根本不是文本时，这种方法变得更加强大：考虑输入文本搜索服装项目，并希望看到与之搭配的整套服装。

# 摘要

将事物放入正确顺序是推荐系统中的一个重要方面。到目前为止，您知道排序并不是全部内容，但它是管道中的一个关键步骤。我们已经收集了我们的物品并将它们放在正确的顺序中，剩下的就是把它们发送给用户。

我们从最基本的概念开始，学习排名，并将其与一些传统方法进行比较。然后我们通过 WARP 和 WSABIE 进行了大幅升级。这使我们最终使用了*k*-order 统计量，这涉及到更谨慎的概率抽样。最后，我们以 BM25 作为文本设置中强大的基线结束。

在我们征服服务之前，让我们把这些要素放在一起。在下一章中，我们将加大音量，并创建一些播放列表。这将是最密集的一章，所以去拿杯饮料伸展一下。我们有些工作要做。
