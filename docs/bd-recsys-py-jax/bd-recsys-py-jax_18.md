# 第十四章：业务逻辑

到现在为止，你可能会想，“是的，我们的算法排名和推荐已经到位了！通过潜在理解为每个用户进行个性化是我们经营业务的方式。”不幸的是，业务很少会这么简单。

让我们来看一个非常直接的例子，一个食谱推荐系统。考虑一个简单讨厌西柚的用户（本书的一位作者*确实*如此），但是可能喜欢与西柚搭配得很好的一系列其他配料：芦笋、鳄梨、香蕉、黄油、腰果、香槟、鸡肉、椰子、蟹肉、鱼、姜、榛子、蜂蜜、柠檬、酸橙、甜瓜、薄荷、橄榄油、洋葱、橙子、山核桃、菠萝、覆盆子、朗姆酒、鲑鱼、海藻、虾、八角茴香、草莓、龙蒿、番茄、香草、葡萄酒和酸奶。这些配料是与西柚*最*受欢迎的搭配，而用户几乎都喜欢这些配料。

推荐系统应该如何处理这种情况？这似乎是协同过滤（CF）、潜在特征或混合推荐可以捕捉到的内容。然而，如果用户喜欢所有这些共享的口味，基于项目的 CF 模型可能无法很好地捕捉到这一点。同样，如果用户真正*讨厌*西柚，潜在特征可能不足以真正避免它。

在这种情况下，简单的方法是一个伟大的选择：*硬避免*。在本章中，我们将讨论业务逻辑与推荐系统输出交汇时的一些复杂性。

与试图将异常作为模型在做出推荐时使用的潜在特征的一部分学习相比，通过确定性逻辑将这些业务规则作为外部步骤集成更一致和简单。例如：模型可以移除所有西柚鸡尾酒，而不是试图学习将它们排名较低。

# 硬排名

当你开始考虑类似于我们的西柚场景的情况时，你可以想出许多这些现象的例子。*硬排名*通常指的是两种特殊排名规则之一：

+   明确地在排名之前从列表中移除一些项目。

+   使用分类特征按类别对结果进行排名。（请注意，这甚至可以针对多个特征进行操作，以实现层次化的硬排名。）

你有没有观察到以下任何一种现象？

+   用户购买了一个沙发。尽管他们未来五年不需要沙发，系统继续向这个用户推荐沙发。

+   用户为一个对园艺感兴趣的朋友购买了生日礼物。然后电子商务网站继续推荐园艺工具，尽管用户对此不感兴趣。

+   父母想给孩子买个玩具。但是当父母去他们通常购买玩具的网站时，网站推荐了几款给比孩子小几岁的孩子的玩具—自孩子那个年龄起，父母就没从这个网站购物过。

+   一名跑步者经历了严重的膝盖疼痛，决定不能再进行长跑。他们转而选择了对关节冲击较小的骑行。然而，他们当地的社交聚会推荐仍然全是跑步相关的。

所有这些情况可以通过确定性逻辑相对容易处理。对于这些情况，我们更倾向于*不*通过机器学习来学习这些规则。我们应该假设对于这些类型的场景，我们将得到关于这些偏好的低信号：负面的隐式反馈通常相关性较低，而且许多列出的情况都是由您希望系统彻底学习的细节所代表的。此外，在之前的一些示例中，如果未能尊重用户的偏好，可能会影响或损害与用户的关系。

这些偏好的名称为*避免项*，有时也称为约束、覆盖或硬规则。您应该将它们视为系统的显式期望：“不要显示带有葡萄柚的食谱”，“不要再显示沙发”，“我不喜欢园艺”，“我的孩子现在已经超过 10 岁了”，以及“不要显示越野跑”。

# 学习到的避免项

并非所有的业务规则都是从显式用户反馈中导出的明显避免项，有些是来自于与特定项目无直接关联的显式反馈。在考虑服务推荐时，包含广泛的避免项是非常重要的。

为了简单起见，假设您正在构建一个时尚推荐系统。更为微妙的避免示例包括以下几种情况：

已拥有的物品

这些是用户真正只需要购买一次的物品，例如通过您平台购买过或者已告知您已拥有的服装。创建一个*虚拟衣橱*可能是一种让用户告知您他们拥有什么的方式，以帮助避免这些情况。

不喜欢的特征

这些是用户可以表示不感兴趣的物品特征。在入职问卷期间，您可以询问用户是否喜欢波点或者是否有喜欢的颜色调色板。这些都是可以用来避免的明确表达的反馈。

忽略的类别

这是一个用户不感兴趣的物品类别或组。这可能是隐式学习的，但是超出了主要推荐模型。也许用户从未点击过您电子商务网站上的连衣裙类别，因为他们不喜欢穿它们。

低质量物品

随着时间的推移，您会了解到某些物品对大多数用户来说质量较低。您可以通过高退货率或买家低评分来检测这一点。这些物品最终应从库存中移除，但与此同时，重要的是将它们作为避免项包含在除了最强匹配信号之外的所有情况中。

这些额外的避免行为可以在服务阶段轻松实现，甚至可以包括简单的模型。训练线性模型来捕捉其中一些规则，然后在服务阶段应用它们可能是提高排名的一种有用且可靠的机制。请注意，小型模型执行推理非常快，因此通常将它们包含在管道中通常不会产生太大的负面影响。对于更大规模的行为趋势或高阶因素，我们期望我们的核心推荐模型能够学习到这些想法。

# 手动调整权重

在避免的另一极端是*手动调整排名*。这种技术在搜索排名的早期时期很受欢迎，当时人类会使用分析和观察来确定他们认为排名中最重要的特征，然后制定多目标排名器。例如，花店可能在五月初排名较高，因为许多用户在寻找母亲节礼物。由于可能有许多变量要跟踪，这些方法不太容易扩展，并且在现代推荐排名中已经大大减少了重视。

然而，手动调整排名在某种程度上可以作为*避免*的一种极其有用的方式。尽管技术上它不是一种避免，但我们有时仍然会这样称呼它。实践中的一个例子是知道新用户喜欢从价格较低的物品开始，因为他们在学习您的运输是否可靠时。一个有用的技术是在第一次订单之前将价格较低的物品提升排名。

虽然考虑构建手动调整排名可能会让人感到不舒服，但重要的是不要排除这种技术。它有一个位置，通常是一个很好的起点。这种技术的一个有趣的人机交互应用是专家手动调整排名。回到我们的时尚推荐器，一个时尚专家可能知道今年夏天流行的颜色是紫红色，特别是在年轻一代中。如果专家将这些紫红色物品为适合的用户提升排名，可能会积极影响用户满意度。

# 库存健康

硬排名的一个独特而又有争议的方面是库存健康。众所周知，很难定义*库存健康*，它估计了现有库存对满足用户需求的好坏程度。

让我们快速看一下定义库存健康的一种方法，通过亲和分数和预测。我们可以通过利用需求预测来做到这一点，这是一种非常强大和流行的优化业务的方式：在接下来的*N*个时间段内，每个类别的预期销售量是多少？建立这些预测模型超出了本书的范围，但这些核心思想在 Rob Hyndman 和 George Athanasopoulos 的著名书籍《Forecasting: Principles and Practice》（Otexts）中得到了很好的捕捉。就我们讨论的目的而言，假设您能大致估计下个月按尺寸和使用类型出售的袜子数量，这可以成为您应该备有各种类型袜子数量的非常有启发性的估计。

然而，事情并不止于此；库存可能是有限的，在实践中，库存通常是销售实物商品企业的主要限制因素。在这种情况下，我们不得不转向市场需求的另一面。如果我们的需求超过了我们的供应能力，最终会让没有得到他们想要物品的用户感到失望。

让我们以销售百吉饼为例；您已经计算了罂粟籽、洋葱、阿斯亚戈芝士和鸡蛋的平均需求。在任何一天，许多顾客会来买心仪的百吉饼，但您是否有足够的百吉饼？您不销售的每一个百吉饼都是浪费；人们喜欢新鲜的百吉饼。这意味着您为每个人推荐的百吉饼都取决于良好的库存。有些用户不那么挑剔；他们可以选择两种或三种选项中的任意一种，同样可以感到满足。在这种情况下，最好为他们提供另一种百吉饼选项，并为挑剔的人节省最低库存。这是一种被称为*优化*的模型细化，涵盖了大量技术。我们不会深入讨论优化技术，但数学优化或运营研究的书籍会提供方向。Mykel J. Kochenderfer 和 Tim A. Wheeler 的《Algorithms for Optimization》（MIT Press）是一个很好的起点。

库存健康与硬性排名密切相关，因为将库存积极管理作为推荐的一部分是一种非常重要且强大的工具。最终，库存优化将降低您推荐的整体性能，但通过将其纳入业务规则的一部分，可以提高业务和推荐系统的整体健康。这就是为什么有时称为*全局优化*。

这些方法引发激烈讨论的原因在于，并非每个人都认同为了改善“整体利益”，就应该降低某些用户的推荐质量。市场健康和平均满意度是需要考虑的有用指标，但确保它们与整体推荐系统的北极星指标一致。

# 实施避免

处理规避的最简单方法是通过下游过滤。为此，你需要在推荐从排名器传递给用户之前应用用户的规避规则。实施这种方法看起来像这样：

```py
import pandas as pd

def filter_dataframe(df: pd.DataFrame, filter_dict: dict):
    """
 Filter a dataframe to exclude rows where columns have certain values.

 Args:
 df (pd.DataFrame): Input dataframe.
 filter_dict (dict): Dictionary where keys are column names
 and values are the values to exclude.

 Returns:
 pd.DataFrame: Filtered dataframe.
 """
    for col, val in filter_dict.items():
        df = df.loc[df[col] != val]
    return df

filter_dict = {'column1': 'value1', 'column2': 'value2', 'column3': 'value3'}

df = df.pipe(filter_dataframe, filter_dict)
```

诚然，这是一个微不足道但也相对天真的规避尝试。首先，纯粹在 pandas 中工作会限制你的推荐系统的可扩展性，所以让我们将其转换为 JAX：

```py
import jax
import jax.numpy as jnp

def filter_jax_array(arr: jnp.array, col_indices: list, values: list):
    """
 Filter a jax array to exclude rows where certain columns have certain values.

 Args:
 arr (jnp.array): Input array.
 col_indices (list): List of column indices to filter on.
 values (list): List of corresponding values to exclude.

 Returns:
 jnp.array: Filtered array.
 """
    assert len(col_indices) == len(values),

    masks = [arr[:, col] != val for col, val in zip(col_indices, values)]
    total_mask = jnp.logical_and(*masks)

    return arr[total_mask]
```

但还有更深层次的问题。你可能会面临的下一个问题是这些避免的集合存储在哪里。一个显而易见的地方就是像 NoSQL 数据库这样的地方，键入用户，然后你可以将所有的避免作为一个简单的查找获取。这是一个自然的特征存储的用法，就像你在“特征存储”中看到的。有些避免可能在实时应用，而其他一些则在用户入职时学习。特征存储是一个很好的容纳避免的地方。

我们天真的过滤器的下一个潜在问题是它不自然地延伸到协变避免，或者更复杂的避免情景。有些避免实际上取决于上下文——一个在劳动节后不穿白色的用户、周五不吃肉的用户，或者咖啡加工方法与某些冲泡器不搭配的情况。所有这些都需要有条件的逻辑。你可能认为你强大而有效的推荐系统模型肯定可以学会这些细节，但这只有时候是真的。事实是，这些考虑的许多种类都比你的推荐系统应该学习的大规模概念信号要低，因此很难始终学会。此外，这些规则通常是你应该要求的，而不是保持乐观的。因此，你通常应该明确指定这些限制。

这个规范通常可以通过明确的确定性算法来实现这些要求。对于咖啡问题，其中一位作者手工建立了一个决策树桩来处理几种咖啡烘焙特征和冲泡器的不良组合——*厌氧浓缩咖啡？呸！*

我们的另外两个例子（不在劳动节后穿白色和周五不吃肉）稍微有些微妙。采用显式的算法方法可能会有些棘手。我们怎么知道用户在一年中的某个时期不吃周五的肉呢？

对于这些用例，基于模型的规避可以强制执行这些要求。

# 基于模型的规避

在我们努力包含更复杂的规则并可能学习它们的过程中，我们可能会听起来像是回到了检索领域。不幸的是，即使是像宽深模型这样有很多参数同时进行用户建模和物品建模的模型，学习这种高级关系也可能会很棘手。

尽管本书的大部分内容都集中在处理相当大和深入的问题上，但推荐系统的这一部分非常适合简单模型。对于基于特征的二元预测（应该推荐这个），我们当然有很多不错的选择。最佳方法显然会严重依赖于在捕捉您希望捕捉的避免时所涉及的特征数量。记住，在本节中考虑的许多避免起初都是假设或假说：我们认为一些用户可能在劳动节后不穿白色，然后试图找到能很好地模拟这一结果的特征。通过这种方式，使用极其简单的回归模型更容易找到与所讨论的结果相关的协变特征。

这个谜题的另一个相关部分是潜在表示。对于我们的周五素食主义者，我们可能正试图推断出一个我们知道有这一规则的特定角色。这个角色是一个我们希望从其他属性中映射出来的潜在特征。在这种建模中要小心（总的来说，角色可能有些微妙，并且值得深思熟虑的决策），但它确实非常有帮助。也许看起来你大型推荐模型的用户建模部分应该学会这些——它们可以！一个有用的技巧是从该模型中提取出已学到的角色，并将它们回归到假设的避免中，以获得更多信号。然而，另一个模型并不总是学习这些角色，因为我们的检索相关性损失函数（以及下游的排名）试图从潜在的角色特征中分析出个别用户的相关性——这些特征可能仅在上下文特征中预测这些避免。

总而言之，实施避免的方法既非常简单又非常困难。在构建生产推荐系统时，当您开始提供服务时，旅程并没有结束；许多模型都会影响到过程的最后一步。

# 摘要

有时，您需要依赖更经典的方法来确保您向下游发送的推荐满足您企业的基本规则。从用户那里学到的明确或微妙的教训可以转化为简单的策略，继续让他们感到愉悦。

然而，这并非我们服务挑战的终点。另一种下游考虑是与我们在这里所做的过滤类型相关，但源自用户偏好和人类行为。确保推荐不重复、机械和冗余是下一章推荐多样性的主题。我们还将讨论在确定确切的服务内容时如何同时平衡多个优先事项。
