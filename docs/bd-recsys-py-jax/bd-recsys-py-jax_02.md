# 第一章\. 介绍

推荐系统是我们今天互联网发展的核心，并且是新兴科技公司的重要功能。除了打开网络广度给每个人的搜索排名外，每年还有更多应用推荐系统的新颖和令人兴奋的电影、所有朋友都在看的新视频，或者是公司支付高价展示给你的最相关广告。TikTok 的令人上瘾的 For You 页面，Spotify 的 Discover Weekly 播放列表，Pinterest 的板块建议以及 Apple 的 App Store 都是推荐系统技术的热门应用。如今，序列变压器模型、多模态表示和图神经网络是机器学习研发中最光明的领域之一，都被应用在推荐系统中。

任何技术的普遍性往往引发如何运作、为什么变得如此普遍以及我们是否能参与其中等问题。对于推荐系统来说，*如何*是相当复杂的。我们需要理解口味的几何形状，以及用户的少量互动如何在那个抽象空间中为我们提供一个*GPS 信号*。你将看到如何快速收集一组优秀的候选者，并将它们精细化为一组协调的推荐。最后，您将学习如何评估您的推荐器，构建服务推理的端点，并记录其行为。

我们将提出核心问题的各种变体，供推荐系统解决，但最终，激励问题的框架如下：

> 给定可能推荐的事物集合，根据特定目标选择适合当前上下文和用户的有序少数。

# 推荐系统的关键组成部分

随着复杂性和精密度的增加，让我们牢记系统的组成部分。我们将使用*字符串图表*来跟踪我们的组件，但在文献中，这些图表以多种方式呈现。

我们将确定并建立推荐系统的三个核心组件：收集者、排名器和服务器。

## 收集者

收集者的角色是了解可能推荐的事物集合及其必要的特征或属性。请注意，这个集合通常是基于上下文或状态的子集。

## 排名器

排名器的角色是接受收集者提供的集合，并根据上下文和用户的模型对其元素进行排序。

## 服务器

服务员的角色是接收排名器提供的有序子集，确保满足必要的数据模式，包括基本的业务逻辑，并返回请求的推荐数量。

例如，以餐馆服务员为例的款待场景：

> 当您坐下来看菜单时，不确定应该点什么。您问服务员：“你认为我应该点什么作为甜点？”
> 
> 侍者检查他们的笔记，并说：“柠檬派已经卖完了，但人们真的很喜欢我们的香蕉奶油派。如果你喜欢石榴，我们会从头开始制作石榴冰淇淋；而且甜甜圈冰淇淋是不会错的——这是我们最受欢迎的甜点。”

在这个简短的交流中，侍者首先充当收集者：识别菜单上的甜点，适应当前的库存情况，并通过检查它们的笔记准备讨论甜点的特性。

接下来，侍者充当排名者；他们提到在受欢迎程度方面得分较高的项目（香蕉奶油派和甜甜圈冰淇淋），以及基于顾客特征的情境高匹配项目（如果他们喜欢石榴）。

最后，侍者口头提供建议，包括他们算法的解释特性和多个选择。

虽然这似乎有点卡通 ish，但请记住，将推荐系统的讨论落实到现实世界的应用中。在 RecSys 中工作的一个优点是灵感总是在附近。

# 最简单的可能的推荐者

我们已经建立了推荐者的组件，但要真正使其实用，我们需要看到它在实践中的运行情况。虽然这本书的大部分内容都专注于实际的推荐系统，但首先我们将从一个玩具开始，并从那里构建。

## 平凡推荐者

最简单的推荐者实际上并不是很有趣，但仍然可以在框架中演示。它被称为 *平凡推荐者*（*TR*），因为它几乎没有逻辑：

```py
def get_trivial_recs() -> Optional[List[str]]:
   item_id = random.randint(0, MAX_ITEM_INDEX)

   if get_availability(item_id):
       return [item_id]
   return None
```

请注意，这个推荐者可能返回一个特定的 `item_id` 或 `None`。还请注意，这个推荐者不接受任何参数，并且 `MAX_ITEM_INDEX` 是引用了一个超出范围的变量。忽略软件原则，让我们思考这三个组件：

收集者

生成了一个随机的 `item_id`。TR 通过检查 `item_id` 的可用性进行收集。我们可以争论说，获得 `item_id` 也是收集者的责任的一部分。有条件地，可推荐的事物的收集要么是 `[item_id]`，要么是 `None`（*请回想 `None` 是集合论意义上的一个集合*）。

排名者

TR（Trivial Recommender）在与无操作相比较；即，在集合中对 1 或 0 个对象进行排名时，对该集合的恒等函数是排名，所以我们只是不做任何事情，继续进行下一步。

服务器

TR 通过其 `return` 语句提供建议。在这个例子中指定的唯一模式是 ⁠`Optional​[List[str]]` 类型的返回类型。

这个推荐者，虽然不太有趣或有用，但提供了一个我们将在进一步开发中添加的框架。

## 最受欢迎的项目推荐者

*最受欢迎的项目推荐者*（MPIR）是包含任何效用的最简单的推荐者。你可能不想围绕它构建应用程序，但它在与其他组件一起使用时很有用，除了提供进一步开发的基础之外。

MPIR 正如它所说的那样工作；它返回最受欢迎的项目：

```py
def get_item_popularities() -> Optional[Dict[str, int]]:
    ...
        # Dict of pairs: (item-identifier, count times item chosen)
        return item_choice_counts
    return None

def get_most_popular_recs(max_num_recs: int) -> Optional[List[str]]:
    items_popularity_dict = get_item_popularities()
    if items_popularity_dict:
        sorted_items = sorted(
            items_popularity_dict.items(),
            key=lambda item: item[1]),
            reverse=True,
        )
        return [i[0] for i in sorted_items][:max_num_recs]
    return None
```

在这里，我们假设`get_item_popularities`知道所有可用项目及其被选择的次数。

这个推荐系统试图返回可用的*k*个最受欢迎的项目。虽然简单，但这是一个有用的推荐系统，是构建推荐系统时的一个很好的起点。此外，我们将看到这个例子一次又一次地返回，因为其他推荐器使用这个核心并逐步改进内部组件。

让我们再次看看我们系统的三个组成部分：

收集器

MPIR 首先调用`get_item_popularities`——通过数据库或内存访问——知道哪些项目可用以及它们被选择的次数。为方便起见，我们假设项目以字典形式返回，键由标识项目的字符串给出，值表示该项目被选择的次数。我们在这里暗示假设不出现在此列表中的项目不可用。

排名器

在这里，我们看到我们的第一个简单的排名器：通过对值进行排序来排名。因为收集器组织了我们的数据，使得字典的值是计数，所以我们使用 Python 内置的排序函数`sorted`。请注意，我们使用`key`指示我们希望按元组的第二个元素排序——在这种情况下，相当于按值排序——并发送`reverse`标志来使我们的排序降序。

服务器

最后，我们需要满足我们的 API 模式，这再次通过返回类型提示提供：`Optional[List[str]]`。这表示返回类型应为可空列表，其中包含我们推荐的项目标识字符串，因此我们使用列表推导来获取元组的第一个元素。但等等！我们的函数有一个`max_num_recs`字段——它可能在做什么？当然，这暗示我们的 API 模式希望响应中不超过`max_num_recs`个结果。我们通过切片操作来处理这个问题，但请注意，我们的返回结果在 0 和`max_num_recs`之间。

考虑到你手头的 MPIR 所提供的可能性；在每个一级类别中推荐客户最喜欢的项目可能会成为电子商务推荐的一个简单但有用的第一步。当天最受欢迎的视频可能会成为你视频网站主页的良好体验。

# 对 JAX 的简要介绍

由于这本书标题中含有*JAX*，我们将在这里提供对 JAX 的简要介绍。其官方文档可以在[JAX 网站上](https://jax.readthedocs.io/en/latest/)找到。

JAX 是一个用 Python 编写数学代码的框架，它是即时编译的。即时编译允许相同的代码在 CPU、GPU 和 TPU 上运行。这使得编写利用向量处理器并行处理能力的高性能代码变得容易。 

此外，JAX 的设计哲学之一是支持张量和梯度作为核心概念，使其成为利用梯度为基础的学习在张量形状数据上的理想工具。玩转 JAX 的最简单方式可能是通过[Google Colab](https://colab.research.google.com/)，这是一个托管在网络上的 Python 笔记本。

## 基本类型、初始化和不可变性

让我们从学习 JAX 类型开始。我们将在 JAX 中构建一个小的三维向量，并指出 JAX 和 NumPy 之间的一些区别：

```py
import jax.numpy as jnp
import numpy as np

x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)

print(x)
[1. 2. 3.]

print(x.shape)
(3,)

print(x[0])
1.0

x[0] = 4.0
TypeError: '<class 'jaxlib.xla_extension.ArrayImpl'>'
object does not support item assignment. JAX arrays are immutable.
```

JAX 的接口与 NumPy 的接口大部分相似。我们按惯例导入 JAX 的 NumPy 版本作为`jnp`，以区分它和 NumPy（`np`），这样我们就知道要使用哪个数学函数的版本。这是因为有时我们可能希望在像 GPU 或 TPU 这样的向量处理器上运行代码，这时我们可以使用 JAX，或者我们可能更喜欢在 CPU 上使用 NumPy 运行一些代码。

首先要注意的是 JAX 数组具有类型。典型的浮点类型是`float32`，它使用 32 位来表示浮点数。还有其他类型，如`float64`，具有更高的精度，以及`float16`，这是一种半精度类型，通常仅在某些 GPU 上运行。

另一个要注意的地方是 JAX 张量具有形状。通常这是一个元组，因此`(3,)`表示沿第一个轴的三维向量。矩阵有两个轴，而张量有三个或更多个轴。

现在我们来看看 JAX 与 NumPy 不同的地方。非常重要的是要注意[“JAX—The Sharp Bits”](https://oreil.ly/qqcFM)来理解这些差异。JAX 的哲学是关于速度和纯度。通过使函数纯粹（没有副作用）并使数据不可变，JAX 能够向其所使用的加速线性代数（XLA）库提供一些保证。JAX 保证这些应用于数据的函数可以并行运行，并且具有确定性结果而没有副作用，因此 XLA 能够编译这些函数并使它们比仅在 NumPy 上运行时更快地运行。

您可以看到修改`x`中的一个元素会导致错误。JAX 更喜欢替换数组`x`而不是修改它。修改数组元素的一种方法是在 NumPy 中进行，而不是在 JAX 中进行，并在随后的代码需要在不可变数据上快速运行时将 NumPy 数组转换为 JAX——例如，使用`jnp.array(np_array)`。

## 索引和切片

另一个重要的学习技能是索引和切片数组：

```py
x = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.int32)

# Print the whole matrix.
print(x)
[[1 2 3]
 [4 5 6]
 [7 8 9]]

# Print the first row.
print(x[0])
[1 2 3]

# Print the last row.
print(x[-1])
[7 8 9]

# Print the second column.
print(x[:, 1])
[2 5 8]

# Print every other element
print(x[::2, ::2])
[[1 3]
 [7 9]]
```

NumPy 引入了索引和切片操作，允许我们访问数组的不同部分。一般来说，符号遵循`start:end:stride`约定。第一个元素指示从哪里开始，第二个指示结束的位置（但不包括该位置），而步长表示跳过的元素数量。该语法类似于 Python `range` 函数的语法。

切片允许我们优雅地访问张量的视图。切片和索引是重要的技能，特别是当我们开始批处理操作张量时，这通常是为了充分利用加速硬件。

## 广播

广播是 NumPy 和 JAX 的另一个要注意的特性。当应用于两个不同大小的张量的二元操作（如加法或乘法）时，具有大小为 1 的轴的张量会被提升到与较大张量相匹配的秩。例如，如果形状为 `(3,3)` 的张量乘以形状为 `(3,1)` 的张量，则在操作之前会复制第二个张量的行，使其看起来像形状为 `(3,3)` 的张量：

```py
x = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=jnp.int32)

# Scalar broadcasting.
y = 2 * x
print(y)
[[ 2  4  6]
 [ 8 10 12]
 [14 16 18]]

# Vector broadcasting. Axes with shape 1 are duplicated.
vec = jnp.reshape(jnp.array([0.5, 1.0, 2.0]), [3, 1])
y = vec * x
print(y)
[[ 0.5  1.   1.5]
 [ 4.   5.   6. ]
 [14.  16.  18. ]]

vec = jnp.reshape(vec, [1, 3])
y = vec * x
print(y)
[[ 0.5  2.   6. ]
 [ 2.   5.  12. ]
 [ 3.5  8.  18. ]]
```

第一种情况是最简单的，即标量乘法。标量在整个矩阵中进行乘法。在第二种情况中，我们有一个形状为 `(3,1)` 的向量乘以矩阵。第一行乘以 0.5，第二行乘以 1.0，第三行乘以 2.0。然而，如果向量已经重塑为 `(1,3)`，则列将分别乘以向量的连续条目。

## 随机数

伴随 JAX 的纯函数哲学而来的是其特殊的随机数处理方式。因为纯函数不会造成副作用，一个随机数生成器不能修改随机数种子，不像其他随机数生成器。相反，JAX 处理的是随机数密钥，其状态被显式地更新：

```py
import jax.random as random

key = random.PRNGKey(0)
x = random.uniform(key, shape=[3, 3])
print(x)
[[0.35490513 0.60419905 0.4275843 ]
 [0.23061597 0.6735498  0.43953657]
 [0.25099766 0.27730572 0.7678207 ]]

key, subkey = random.split(key)
x = random.uniform(key, shape=[3, 3])
print(x)
[[0.0045197  0.5135027  0.8613342 ]
 [0.06939673 0.93825936 0.85599923]
 [0.706004   0.50679076 0.6072922 ]]

y = random.uniform(subkey, shape=[3, 3])
print(y)
[[0.34896135 0.48210478 0.02053976]
 [0.53161216 0.48158717 0.78698325]
 [0.07476437 0.04522789 0.3543167 ]]
```

首先，JAX 要求你从种子创建一个随机数 `key`。然后将这个密钥传递给类似 `uniform` 的随机数生成函数，以创建范围在 0 到 1 之间的随机数。

要创建更多的随机数，然而，JAX 要求你将密钥分为两部分：一个新密钥用于生成其他密钥，一个子密钥用于生成新的随机数。这使得 JAX 即使在许多并行操作调用随机数生成器时，也能确定性地和可靠地复现随机数。我们只需将一个密钥分成需要的许多并行操作，所得的随机数现在既是随机分布的，又是可重现的。这在你希望可靠地复现实验时是一种良好的特性。

## 即时编译

当我们开始使用 JIT 编译时，JAX 在执行速度上开始与 NumPy 有所不同。JIT 编译——即时将代码转换为即时编译——允许相同的代码在 CPU、GPU 或 TPU 上运行：

```py
import jax

x = random.uniform(key, shape=[2048, 2048]) - 0.5

def my_function(x):
  x = x @ x
  return jnp.maximum(0.0, x)

%timeit my_function(x).block_until_ready()
302 ms ± 9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

my_function_jitted = jax.jit(my_function)

%timeit my_function_jitted(x).block_until_ready()
294 ms ± 5.45 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

JIT 编译的代码在 CPU 上速度提升不多，但在 GPU 或 TPU 后端上速度会显著提升。当函数第一次调用时，编译也会带来一些开销，这可能会使第一次调用的时间偏离。能够 JIT 编译的函数有一些限制，比如主要在内部调用 JAX 操作，并对循环操作有限制。长度可变的循环会触发频繁的重新编译。[“Just-in-Time Compilation with JAX” 文档](https://oreil.ly/c8ywT)详细介绍了许多 JIT 编译函数的细微差别。

# 摘要

虽然我们还没有进行太多的数学工作，但我们已经到了可以开始提供推荐和实现这些组件更深层逻辑的阶段。我们很快将开始做一些看起来像是机器学习的事情。

到目前为止，我们已经定义了推荐问题的概念，并设置了我们推荐系统的核心架构——收集器、排名器和服务器，并展示了几个简单的推荐器来说明这些部件如何组合在一起。

接下来，我们将解释推荐系统试图利用的核心关系：用户-物品矩阵。这个矩阵使我们能够构建个性化模型，从而进行排名。