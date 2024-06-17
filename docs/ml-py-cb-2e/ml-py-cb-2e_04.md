# 第四章：处理数值数据

# 4.0 引言

定量数据是某物的测量——无论是班级规模、月销售额还是学生分数。表示这些数量的自然方式是数值化（例如，29 名学生、销售额为 529,392 美元）。在本章中，我们将介绍多种策略，将原始数值数据转换为专门用于机器学习算法的特征。

# 4.1 重新调整特征

## 问题

您需要将数值特征的值重新缩放到两个值之间。

## 解决方案

使用 scikit-learn 的`MinMaxScaler`来重新调整特征数组：

```py
# Load libraries
import numpy as np
from sklearn import preprocessing

# Create feature
feature = np.array([[-500.5],
                    [-100.1],
                    [0],
                    [100.1],
                    [900.9]])

# Create scaler
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))

# Scale feature
scaled_feature = minmax_scale.fit_transform(feature)

# Show feature
scaled_feature
```

```py
array([[ 0\.        ],
       [ 0.28571429],
       [ 0.35714286],
       [ 0.42857143],
       [ 1\.        ]])
```

## 讨论

*重新缩放* 是机器学习中常见的预处理任务。本书后面描述的许多算法将假定所有特征在同一尺度上，通常是 0 到 1 或-1 到 1。有许多重新缩放技术，但最简单的之一称为*最小-最大缩放*。最小-最大缩放使用特征的最小值和最大值将值重新缩放到一个范围内。具体来说，最小-最大缩放计算：

<math display="block"><mrow><msubsup><mi>x</mi> <mi>i</mi> <mo>'</mo></msubsup> <mo>=</mo> <mfrac><mrow><msub><mi>x</mi> <mi>i</mi></msub> <mo>-</mo><mtext>min</mtext><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow> <mrow><mtext>max</mtext><mo>(</mo><mi>x</mi><mo>)</mo><mo>-</mo><mtext>min</mtext><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mfrac></mrow></math>

其中<math display="inline"><mi>x</mi></math>是特征向量，<math display="inline"><msub><mi>x</mi><mi>i</mi></msub></math>是特征<math display="inline"><mi>x</mi></math>的单个元素，<math display="inline"><msubsup><mi>x</mi> <mi>i</mi> <mo>'</mo></msubsup></math>是重新调整的元素。在我们的例子中，我们可以从输出的数组中看到，特征已成功重新调整为 0 到 1 之间：

```py
array([[ 0\.        ],
      [ 0.28571429],
      [ 0.35714286],
      [ 0.42857143],
      [ 1\.        ]])
```

scikit-learn 的`MinMaxScaler`提供了两种重新调整特征的选项。一种选项是使用`fit`来计算特征的最小值和最大值，然后使用`transform`来重新调整特征。第二个选项是使用`fit_transform`来同时执行这两个操作。这两个选项在数学上没有区别，但有时将操作分开会有实际的好处，因为这样可以将相同的转换应用于不同的*数据集*。

## 参见

+   [特征缩放，维基百科](https://oreil.ly/f2WiM)

+   [关于特征缩放和归一化，Sebastian Raschka](https://oreil.ly/Da0AH)

# 4.2 标准化特征

## 问题

您希望将一个特征转换为具有均值为 0 和标准差为 1。

## 解决方案

scikit-learn 的`StandardScaler`执行这两个转换：

```py
# Load libraries
import numpy as np
from sklearn import preprocessing

# Create feature
x = np.array([[-1000.1],
              [-200.2],
              [500.5],
              [600.6],
              [9000.9]])

# Create scaler
scaler = preprocessing.StandardScaler()

# Transform the feature
standardized = scaler.fit_transform(x)

# Show feature
standardized
```

```py
array([[-0.76058269],
       [-0.54177196],
       [-0.35009716],
       [-0.32271504],
       [ 1.97516685]])
```

## 讨论

对于问题 4.1 中讨论的最小-最大缩放的常见替代方案是将特征重新缩放为近似标准正态分布。为了实现这一目标，我们使用标准化来转换数据，使其均值<math display="inline"><mover accent="true"><mi>x</mi> <mo>¯</mo></mover></math>为 0，标准差<math display="inline"><mi>σ</mi></math>为 1。具体来说，特征中的每个元素都被转换，以便：

<math display="block"><mrow><msubsup><mi>x</mi> <mi>i</mi> <mo>'</mo></msubsup> <mo>=</mo> <mfrac><mrow><msub><mi>x</mi> <mi>i</mi></msub> <mo>-</mo><mover accent="true"><mi>x</mi> <mo>¯</mo></mover></mrow> <mi>σ</mi></mfrac></mrow></math>

其中 <math display="inline"><msubsup><mi>x</mi> <mi>i</mi> <mo>'</mo></msubsup></math> 是 <math display="inline"><msub><mi>x</mi> <mi>i</mi></msub></math> 的标准化形式。转换后的特征表示原始值与特征均值之间的标准偏差数（在统计学中也称为 *z-score*）。

标准化是机器学习预处理中常见的缩放方法，在我的经验中，它比最小-最大缩放更常用。但这取决于学习算法。例如，主成分分析通常在使用标准化时效果更好，而对于神经网络，则通常建议使用最小-最大缩放（这两种算法稍后在本书中讨论）。作为一个一般规则，我建议除非有特定原因使用其他方法，否则默认使用标准化。

我们可以通过查看解决方案输出的平均值和标准偏差来看到标准化的效果：

```py
# Print mean and standard deviation
print("Mean:", round(standardized.mean()))
print("Standard deviation:", standardized.std())
```

```py
Mean: 0.0
Standard deviation: 1.0
```

如果我们的数据存在显著的异常值，它可能通过影响特征的均值和方差而对我们的标准化产生负面影响。在这种情况下，通常可以通过使用中位数和四分位距来重新调整特征，从而提供帮助。在 scikit-learn 中，我们使用 `RobustScaler` 方法来实现这一点：

```py
# Create scaler
robust_scaler = preprocessing.RobustScaler()

# Transform feature
robust_scaler.fit_transform(x)
```

```py
array([[ -1.87387612],
       [ -0.875     ],
       [  0\.        ],
       [  0.125     ],
       [ 10.61488511]])
```

# 4.3 规范化观测值

## 问题

您希望将观测值的特征值重新调整为单位范数（总长度为 1）。

## 解决方案

使用带有 `norm` 参数的 `Normalizer`：

```py
# Load libraries
import numpy as np
from sklearn.preprocessing import Normalizer

# Create feature matrix
features = np.array([[0.5, 0.5],
                     [1.1, 3.4],
                     [1.5, 20.2],
                     [1.63, 34.4],
                     [10.9, 3.3]])

# Create normalizer
normalizer = Normalizer(norm="l2")

# Transform feature matrix
normalizer.transform(features)
```

```py
array([[ 0.70710678,  0.70710678],
       [ 0.30782029,  0.95144452],
       [ 0.07405353,  0.99725427],
       [ 0.04733062,  0.99887928],
       [ 0.95709822,  0.28976368]])
```

## 讨论

许多重新调整方法（例如，最小-最大缩放和标准化）作用于特征，但我们也可以跨个体观测值进行重新调整。`Normalizer` 将单个观测值上的值重新调整为单位范数（它们长度的总和为 1）。当存在许多等效特征时（例如，在文本分类中，每个单词或 *n*-word 组合都是一个特征时），通常会使用这种重新调整。

`Normalizer` 提供三种范数选项，其中欧几里德范数（通常称为 L2）是默认参数：

<math display="block"><mrow><msub><mfenced close="∥" open="∥"><mi>x</mi></mfenced> <mn>2</mn></msub> <mo>=</mo> <msqrt><mrow><msup><msub><mi>x</mi> <mn>1</mn></msub> <mn>2</mn></msup> <mo>+</mo> <msup><msub><mi>x</mi> <mn>2</mn></msub> <mn>2</mn></msup> <mo>+</mo> <mo>⋯</mo> <mo>+</mo> <msup><msub><mi>x</mi> <mi>n</mi></msub> <mn>2</mn></msup></mrow></msqrt></mrow></math>

其中 <math display="inline"><mi>x</mi></math> 是一个单独的观测值，<math display="inline"><msub><mi>x</mi><mi>n</mi></msub></math> 是该观测值在第 <math display="inline"><mi>n</mi></math> 个特征上的值。

```py
# Transform feature matrix
features_l2_norm = Normalizer(norm="l2").transform(features)

# Show feature matrix
features_l2_norm
```

```py
array([[ 0.70710678,  0.70710678],
       [ 0.30782029,  0.95144452],
       [ 0.07405353,  0.99725427],
       [ 0.04733062,  0.99887928],
       [ 0.95709822,  0.28976368]])
```

或者，我们可以指定曼哈顿范数（L1）：

<math display="block"><mrow><msub><mfenced close="∥" open="∥"><mi>x</mi></mfenced> <mn>1</mn></msub> <mo>=</mo> <munderover><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>n</mi></munderover> <mfenced close="|" open="|" separators=""><msub><mi>x</mi> <mi>i</mi></msub></mfenced> <mo>.</mo></mrow></math>

```py
# Transform feature matrix
features_l1_norm = Normalizer(norm="l1").transform(features)

# Show feature matrix
features_l1_norm
```

```py
array([[ 0.5       ,  0.5       ],
       [ 0.24444444,  0.75555556],
       [ 0.06912442,  0.93087558],
       [ 0.04524008,  0.95475992],
       [ 0.76760563,  0.23239437]])
```

直观上，L2 范数可以被视为鸟在纽约两点之间的距离（即直线距离），而 L1 范数可以被视为在街道上行走的人的距离（向北走一块，向东走一块，向北走一块，向东走一块，等等），这就是为什么它被称为“曼哈顿范数”或“出租车范数”的原因。

在实际应用中，注意到 `norm="l1"` 将重新调整观测值的值，使其总和为 1，这在某些情况下是一种可取的质量：

```py
# Print sum
print("Sum of the first observation\'s values:",
   features_l1_norm[0, 0] + features_l1_norm[0, 1])
```

```py
Sum of the first observation's values: 1.0
```

# 4.4 生成多项式和交互特征

## 问题

您希望创建多项式和交互特征。

## 解决方案

即使有些人选择手动创建多项式和交互特征，scikit-learn 提供了一个内置方法：

```py
# Load libraries
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Create feature matrix
features = np.array([[2, 3],
                     [2, 3],
                     [2, 3]])

# Create PolynomialFeatures object
polynomial_interaction = PolynomialFeatures(degree=2, include_bias=False)

# Create polynomial features
polynomial_interaction.fit_transform(features)
```

```py
array([[ 2.,  3.,  4.,  6.,  9.],
       [ 2.,  3.,  4.,  6.,  9.],
       [ 2.,  3.,  4.,  6.,  9.]])
```

参数`degree`确定多项式的最大次数。例如，`degree=2`将创建被提升到二次幂的新特征：

<math display="block"><mrow><msub><mi>x</mi> <mn>1</mn></msub> <mo>,</mo> <msub><mi>x</mi> <mn>2</mn></msub> <mo>,</mo> <msup><msub><mi>x</mi> <mn>1</mn></msub> <mn>2</mn></msup> <mo>,</mo> <msup><msub><mi>x</mi> <mn>1</mn></msub> <mn>2</mn></msup> <mo>,</mo> <msup><msub><mi>x</mi> <mn>2</mn></msub> <mn>2</mn></msup></mrow></math>

而`degree=3`将创建被提升到二次和三次幂的新特征：

<math display="block"><mrow><msub><mi>x</mi> <mn>1</mn></msub> <mo>,</mo> <msub><mi>x</mi> <mn>2</mn></msub> <mo>,</mo> <msup><msub><mi>x</mi> <mn>1</mn></msub> <mn>2</mn></msup> <mo>,</mo> <msup><msub><mi>x</mi> <mn>2</mn></msub> <mn>2</mn></msup> <mo>,</mo> <msup><msub><mi>x</mi> <mn>1</mn></msub> <mn>3</mn></msup> <mo>,</mo> <msup><msub><mi>x</mi> <mn>2</mn></msub> <mn>3</mn></msup> <mo>,</mo> <msup><msub><mi>x</mi> <mn>1</mn></msub> <mn>2</mn></msup> <mo>,</mo> <msup><msub><mi>x</mi> <mn>1</mn></msub> <mn>3</mn></msup> <mo>,</mo> <msup><msub><mi>x</mi> <mn>2</mn></msub> <mn>3</mn></msup></mrow></math>

此外，默认情况下，`PolynomialFeatures`包括交互特征：

<math display="block"><mrow><msub><mi>x</mi> <mn>1</mn></msub> <msub><mi>x</mi> <mn>2</mn></msub></mrow></math>

我们可以通过将`interaction_only`设置为`True`来限制仅创建交互特征：

```py
interaction = PolynomialFeatures(degree=2,
              interaction_only=True, include_bias=False)

interaction.fit_transform(features)
```

```py
array([[ 2.,  3.,  6.],
       [ 2.,  3.,  6.],
       [ 2.,  3.,  6.]])
```

## 讨论

当我们希望包括特征与目标之间存在非线性关系时，通常会创建多项式特征。例如，我们可能怀疑年龄对患重大医疗状况的概率的影响并非随时间恒定，而是随年龄增加而增加。我们可以通过生成该特征的高阶形式（<math display="inline"><msup><mi>x</mi><mn>2</mn></msup></math>，<math display="inline"><msup><mi>x</mi><mn>3</mn></msup></math>等）来编码这种非恒定效果。

此外，我们经常遇到一种情况，即一个特征的效果取决于另一个特征。一个简单的例子是，如果我们试图预测我们的咖啡是否甜，我们有两个特征：(1)咖啡是否被搅拌，以及(2)是否添加了糖。单独来看，每个特征都不能预测咖啡的甜度，但它们的效果组合起来却可以。也就是说，只有当咖啡既加了糖又被搅拌时，咖啡才会变甜。每个特征对目标（甜度）的影响取决于彼此之间的关系。我们可以通过包含一个交互特征，即两个个体特征的乘积来编码这种关系。

# 4.5 特征转换

## 问题

你希望对一个或多个特征进行自定义转换。

## 解决方案

在 scikit-learn 中，使用`FunctionTransformer`将一个函数应用到一组特征上：

```py
# Load libraries
import numpy as np
from sklearn.preprocessing import FunctionTransformer

# Create feature matrix
features = np.array([[2, 3],
                     [2, 3],
                     [2, 3]])

# Define a simple function
def add_ten(x: int) -> int:
    return x + 10

# Create transformer
ten_transformer = FunctionTransformer(add_ten)

# Transform feature matrix
ten_transformer.transform(features)
```

```py
array([[12, 13],
       [12, 13],
       [12, 13]])
```

我们可以使用`apply`在 pandas 中创建相同的转换：

```py
# Load library
import pandas as pd

# Create DataFrame
df = pd.DataFrame(features, columns=["feature_1", "feature_2"])

# Apply function
df.apply(add_ten)
```

|  | feature_1 | feature_2 |
| --- | --- | --- |
| 0 | 12 | 13 |
| 1 | 12 | 13 |
| 2 | 12 | 13 |

## 讨论

通常希望对一个或多个特征进行一些自定义转换。例如，我们可能想创建一个特征，其值是另一个特征的自然对数。我们可以通过创建一个函数，然后使用 scikit-learn 的`FunctionTransformer`或 pandas 的`apply`将其映射到特征来实现这一点。在解决方案中，我们创建了一个非常简单的函数`add_ten`，它为每个输入加了 10，但我们完全可以定义一个复杂得多的函数。

# 4.6 检测异常值

## 问题

你希望识别极端观察结果。

## 解决方案

检测异常值很遗憾更像是一种艺术而不是一种科学。然而，一种常见的方法是假设数据呈正态分布，并基于该假设在数据周围“画”一个椭圆，将椭圆内的任何观察结果归类为内围值（标记为`1`），将椭圆外的任何观察结果归类为异常值（标记为`-1`）：

```py
# Load libraries
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs

# Create simulated data
features, _ = make_blobs(n_samples = 10,
                         n_features = 2,
                         centers = 1,
                         random_state = 1)

# Replace the first observation's values with extreme values
features[0,0] = 10000
features[0,1] = 10000

# Create detector
outlier_detector = EllipticEnvelope(contamination=.1)

# Fit detector
outlier_detector.fit(features)

# Predict outliers
outlier_detector.predict(features)
```

```py
array([-1,  1,  1,  1,  1,  1,  1,  1,  1,  1])
```

在这些数组中，值为-1 表示异常值，而值为 1 表示内围值。这种方法的一个主要局限性是需要指定一个`contamination`参数，它是异常值观察值的比例，这是我们不知道的值。将`contamination`视为我们对数据清洁程度的估计。如果我们预计数据中有很少的异常值，我们可以将`contamination`设置为较小的值。但是，如果我们认为数据可能有异常值，我们可以将其设置为较高的值。

我们可以不将观察结果作为一个整体来看待，而是可以查看单个特征，并使用四分位距（IQR）来识别这些特征中的极端值：

```py
# Create one feature
feature = features[:,0]

# Create a function to return index of outliers
def indicies_of_outliers(x: int) -> np.array(int):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return np.where((x > upper_bound) | (x < lower_bound))

# Run function
indicies_of_outliers(feature)
```

```py
(array([0]),)
```

IQR 是一组数据的第一和第三四分位数之间的差异。您可以将 IQR 视为数据的主要集中区域的扩展，而异常值是远离数据主要集中区域的观察结果。异常值通常定义为第一四分位数的 1.5 倍 IQR 小于或第三四分位数的 1.5 倍 IQR 大于的任何值。

## 讨论

没有单一的最佳技术来检测异常值。相反，我们有一系列技术，各有优缺点。我们最好的策略通常是尝试多种技术（例如，`EllipticEnvelope`和基于 IQR 的检测）并综合查看结果。

如果可能的话，我们应该查看我们检测到的异常值，并尝试理解它们。例如，如果我们有一个房屋数据集，其中一个特征是房间数量，那么房间数量为 100 的异常值是否真的是一座房子，还是实际上是一个被错误分类的酒店？

## 参见

+   [检测异常值的三种方法（以及此配方中使用的 IQR 函数的来源）](https://oreil.ly/wlwmH)

# 4.7 处理异常值

## 问题

您的数据中存在异常值，您希望识别并减少其对数据分布的影响。

## 解决方案

通常我们可以采用三种策略来处理异常值。首先，我们可以放弃它们：

```py
# Load library
import pandas as pd

# Create DataFrame
houses = pd.DataFrame()
houses['Price'] = [534433, 392333, 293222, 4322032]
houses['Bathrooms'] = [2, 3.5, 2, 116]
houses['Square_Feet'] = [1500, 2500, 1500, 48000]

# Filter observations
houses[houses['Bathrooms'] < 20]
```

|  | 价格 | 浴室 | 平方英尺 |
| --- | --- | --- | --- |
| 0 | 534433 | 2.0 | 1500 |
| 1 | 392333 | 3.5 | 2500 |
| 2 | 293222 | 2.0 | 1500 |

其次，我们可以将它们标记为异常值，并将“异常值”作为特征包含在内：

```py
# Load library
import numpy as np

# Create feature based on boolean condition
houses["Outlier"] = np.where(houses["Bathrooms"] < 20, 0, 1)

# Show data
houses
```

|  | 价格 | 浴室 | 平方英尺 | 异常值 |
| --- | --- | --- | --- | --- |
| 0 | 534433 | 2.0 | 1500 | 0 |
| 1 | 392333 | 3.5 | 2500 | 0 |
| 2 | 293222 | 2.0 | 1500 | 0 |
| 3 | 4322032 | 116.0 | 48000 | 1 |

最后，我们可以转换特征以减轻异常值的影响：

```py
# Log feature
houses["Log_Of_Square_Feet"] = [np.log(x) for x in houses["Square_Feet"]]

# Show data
houses
```

|  | 价格 | 浴室 | 平方英尺 | 异常值 | 平方英尺的对数 |
| --- | --- | --- | --- | --- | --- |
| 0 | 534433 | 2.0 | 1500 | 0 | 7.313220 |
| 1 | 392333 | 3.5 | 2500 | 0 | 7.824046 |
| 2 | 293222 | 2.0 | 1500 | 0 | 7.313220 |
| 3 | 4322032 | 116.0 | 48000 | 1 | 10.778956 |

## 讨论

类似于检测异常值，处理它们没有硬性规则。我们处理它们应该基于两个方面。首先，我们应该考虑它们为何成为异常值。如果我们认为它们是数据中的错误，比如来自损坏传感器或错误编码的值，那么我们可能会删除该观测值或将异常值替换为`NaN`，因为我们不能信任这些值。然而，如果我们认为异常值是真实的极端值（例如，一个有 200 个浴室的豪宅），那么将它们标记为异常值或转换它们的值更为合适。

其次，我们处理异常值的方式应该基于我们在机器学习中的目标。例如，如果我们想根据房屋特征预测房价，我们可能合理地假设拥有超过 100 个浴室的豪宅的价格受到不同动态的驱动，而不是普通家庭住宅。此外，如果我们正在训练一个在线住房贷款网站应用程序的模型，我们可能会假设我们的潜在用户不包括寻求购买豪宅的亿万富翁。

那么如果我们有异常值应该怎么办？考虑它们为何成为异常值，设定数据的最终目标，最重要的是记住，不处理异常值本身也是一种带有影响的决策。

另外一点：如果存在异常值，标准化可能不合适，因为异常值可能会严重影响均值和方差。在这种情况下，应该使用对异常值更具鲁棒性的重新缩放方法，比如`RobustScaler`。

## 参见

+   [`RobustScaler` 文档](https://oreil.ly/zgm-1)

# 4.8 特征离散化

## 问题

您有一个数值特征，并希望将其分割成离散的箱子。

## 解决方案

根据数据分割方式的不同，我们可以使用两种技术。首先，我们可以根据某个阈值对特征进行二值化：

```py
# Load libraries
import numpy as np
from sklearn.preprocessing import Binarizer

# Create feature
age = np.array([[6],
                [12],
                [20],
                [36],
                [65]])

# Create binarizer
binarizer = Binarizer(threshold=18)

# Transform feature
binarizer.fit_transform(age)
```

```py
array([[0],
       [0],
       [1],
       [1],
       [1]])
```

其次，我们可以根据多个阈值分割数值特征：

```py
# Bin feature
np.digitize(age, bins=[20,30,64])
```

```py
array([[0],
       [0],
       [1],
       [2],
       [3]])
```

注意，`bins` 参数的参数表示每个箱的左边缘。例如，`20` 参数不包括值为 20 的元素，只包括比 20 小的两个值。我们可以通过将参数 `right` 设置为 `True` 来切换这种行为：

```py
# Bin feature
np.digitize(age, bins=[20,30,64], right=True)
```

```py
array([[0],
       [0],
       [0],
       [2],
       [3]])
```

## 讨论

当我们有理由认为数值特征应该表现得更像分类特征时，离散化可以是一种有效的策略。例如，我们可能认为 19 岁和 20 岁的人的消费习惯几乎没有什么差异，但 20 岁和 21 岁之间存在显著差异（美国的法定饮酒年龄）。在这种情况下，将数据中的个体分为可以饮酒和不能饮酒的人可能是有用的。同样，在其他情况下，将数据离散化为三个或更多的箱子可能是有用的。

在解决方案中，我们看到了两种离散化的方法——scikit-learn 的`Binarizer`用于两个区间和 NumPy 的`digitize`用于三个或更多的区间——然而，我们也可以像使用`Binarizer`那样使用`digitize`来对功能进行二值化，只需指定一个阈值：

```py
# Bin feature
np.digitize(age, bins=[18])
```

```py
array([[0],
       [0],
       [1],
       [1],
       [1]])
```

## 另请参阅

+   [`digitize` documentation](https://oreil.ly/KipXX)

# 4.9 使用聚类对观测进行分组

## 问题

您希望将观测聚类，以便将相似的观测分组在一起。

## 解决方案

如果您知道您有*k*个组，您可以使用 k 均值聚类来将相似的观测分组，并输出一个新的特征，其中包含每个观测的组成员资格：

```py
# Load libraries
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Make simulated feature matrix
features, _ = make_blobs(n_samples = 50,
                         n_features = 2,
                         centers = 3,
                         random_state = 1)

# Create DataFrame
dataframe = pd.DataFrame(features, columns=["feature_1", "feature_2"])

# Make k-means clusterer
clusterer = KMeans(3, random_state=0)

# Fit clusterer
clusterer.fit(features)

# Predict values
dataframe["group"] = clusterer.predict(features)

# View first few observations
dataframe.head(5)
```

|  | 功能 _1 | 功能 _2 | 组 |
| --- | --- | --- | --- |
| 0 | –9.877554 | –3.336145 | 0 |
| 1 | –7.287210 | –8.353986 | 2 |
| 2 | –6.943061 | –7.023744 | 2 |
| 3 | –7.440167 | –8.791959 | 2 |
| 4 | –6.641388 | –8.075888 | 2 |

## 讨论

我们稍微超前一点，并且将在本书的后面更深入地讨论聚类算法。但是，我想指出，我们可以将聚类用作预处理步骤。具体来说，我们使用无监督学习算法（如 k 均值）将观测分成组。结果是一个分类特征，具有相似观测的成员属于同一组。

如果您没有理解所有这些，不要担心：只需将聚类可用于预处理的想法存档。如果您真的等不及，现在就可以翻到第十九章。

# 4.10 删除具有缺失值的观测

## 问题

您需要删除包含缺失值的观测。

## 解决方案

使用 NumPy 的巧妙一行代码轻松删除具有缺失值的观测：

```py
# Load library
import numpy as np

# Create feature matrix
features = np.array([[1.1, 11.1],
                     [2.2, 22.2],
                     [3.3, 33.3],
                     [4.4, 44.4],
                     [np.nan, 55]])

# Keep only observations that are not (denoted by ~) missing
features[~np.isnan(features).any(axis=1)]
```

```py
array([[  1.1,  11.1],
       [  2.2,  22.2],
       [  3.3,  33.3],
       [  4.4,  44.4]])
```

或者，我们可以使用 pandas 删除缺失的观测：

```py
# Load library
import pandas as pd

# Load data
dataframe = pd.DataFrame(features, columns=["feature_1", "feature_2"])

# Remove observations with missing values
dataframe.dropna()
```

|  | 功能 _1 | 功能 _2 |
| --- | --- | --- |
| 0 | 1.1 | 11.1 |
| 1 | 2.2 | 22.2 |
| 2 | 3.3 | 33.3 |
| 3 | 4.4 | 44.4 |

## 讨论

大多数机器学习算法无法处理目标和特征数组中的任何缺失值。因此，我们不能忽略数据中的缺失值，必须在预处理过程中解决这个问题。

最简单的解决方案是删除包含一个或多个缺失值的每个观测，可以使用 NumPy 或 pandas 快速轻松地完成此任务。

也就是说，我们应该非常不情愿地删除具有缺失值的观测。删除它们是核心选项，因为我们的算法失去了观测的非缺失值中包含的信息。

同样重要的是，根据缺失值的原因，删除观测可能会向我们的数据引入偏差。有三种类型的缺失数据：

完全随机缺失（MCAR）

缺失值出现的概率与一切无关。例如，调查对象在回答问题之前掷骰子：如果她掷出六点，她会跳过那个问题。

随机缺失（MAR）

值缺失的概率并非完全随机，而是依赖于其他特征捕获的信息。例如，一项调查询问性别身份和年薪，女性更有可能跳过薪水问题；然而，她们的未响应仅依赖于我们在性别身份特征中捕获的信息。

缺失非随机（MNAR）

值缺失的概率并非随机，而是依赖于我们特征未捕获的信息。例如，一项调查询问年薪，女性更有可能跳过薪水问题，而我们的数据中没有性别身份特征。

如果数据是 MCAR 或 MAR，有时可以接受删除观测值。但是，如果值是 MNAR，缺失本身就是信息。删除 MNAR 观测值可能会在数据中引入偏差，因为我们正在删除由某些未观察到的系统效应产生的观测值。

## 另请参阅

+   [识别三种缺失数据类型](https://oreil.ly/sz9Fx)

+   [缺失数据填补](https://oreil.ly/swU2j)

# 4.11 填补缺失值

## 问题

您的数据中存在缺失值，并希望通过通用方法或预测来填补它们。

## 解决方案

您可以使用 k 最近邻（KNN）或 scikit-learn 的`SimpleImputer`类来填补缺失值。如果数据量较小，请使用 KNN 进行预测和填补缺失值：

```py
# Load libraries
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# Make a simulated feature matrix
features, _ = make_blobs(n_samples = 1000,
                         n_features = 2,
                         random_state = 1)

# Standardize the features
scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)

# Replace the first feature's first value with a missing value
true_value = standardized_features[0,0]
standardized_features[0,0] = np.nan

# Predict the missing values in the feature matrix
knn_imputer = KNNImputer(n_neighbors=5)
features_knn_imputed = knn_imputer.fit_transform(standardized_features)

# Compare true and imputed values
print("True Value:", true_value)
print("Imputed Value:", features_knn_imputed[0,0])
```

```py
True Value: 0.8730186114
Imputed Value: 1.09553327131
```

或者，我们可以使用 scikit-learn 的`imputer`模块中的`SimpleImputer`类，将缺失值用特征的均值、中位数或最频繁的值填充。然而，通常情况下，与 KNN 相比，我们通常会获得更差的结果：

```py
# Load libraries
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# Make a simulated feature matrix
features, _ = make_blobs(n_samples = 1000,
                         n_features = 2,
                         random_state = 1)

# Standardize the features
scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)

# Replace the first feature's first value with a missing value
true_value = standardized_features[0,0]
standardized_features[0,0] = np.nan

# Create imputer using the "mean" strategy
mean_imputer = SimpleImputer(strategy="mean")

# Impute values
features_mean_imputed = mean_imputer.fit_transform(features)

# Compare true and imputed values
print("True Value:", true_value)
print("Imputed Value:", features_mean_imputed[0,0])
```

```py
True Value: 0.8730186114
Imputed Value: -3.05837272461
```

## 讨论

替换缺失数据的两种主要策略都有各自的优势和劣势。首先，我们可以使用机器学习来预测缺失数据的值。为此，我们将带有缺失值的特征视为目标向量，并使用其余子集特征来预测缺失值。虽然我们可以使用各种机器学习算法来填补值，但一个流行的选择是 KNN。在第十五章深入讨论了 KNN，简而言之，该算法使用*k*个最近的观测值（根据某个距离度量）来预测缺失值。在我们的解决方案中，我们使用了五个最接近的观测值来预测缺失值。

KNN 的缺点在于为了知道哪些观测值最接近缺失值，需要计算缺失值与每个观测值之间的距离。在较小的数据集中这是合理的，但是如果数据集有数百万个观测值，则很快会变得问题重重。在这种情况下，近似最近邻（ANN）是一个更可行的方法。我们将在第 15.5 节讨论 ANN。

与 KNN 相比，一种可替代且更可扩展的策略是用平均值、中位数或众数填补数值数据的缺失值。例如，在我们的解决方案中，我们使用 scikit-learn 将缺失值填充为特征的均值。填充的值通常不如我们使用 KNN 时接近真实值，但我们可以更轻松地将均值填充应用到包含数百万观察值的数据中。

如果我们使用填充，创建一个二进制特征指示观察是否包含填充值是一个好主意。

## 另请参阅

+   [scikit-learn 文档：缺失值的填充](https://oreil.ly/1M4bn)

+   [K-最近邻居作为填充方法的研究](https://oreil.ly/012--)
