# 第十章：使用特征选择进行降维

# 10.0 Introduction

在第九章中，我们讨论了如何通过创建具有（理想情况下）类似能力的新特征来降低特征矩阵的维度。这称为*特征提取*。在本章中，我们将介绍一种替代方法：选择高质量、信息丰富的特征并丢弃不太有用的特征。这称为*特征选择*。

有三种特征选择方法：过滤、包装和嵌入。*过滤方法*通过检查特征的统计属性选择最佳特征。我们明确设置统计量的阈值或手动选择要保留的特征数的方法是通过过滤进行特征选择的示例。包装方法使用试错法找到产生质量预测模型的特征子集。*包装方法*通常是最有效的，因为它们通过实际试验而非简单的假设来找到最佳结果。最后，*嵌入方法*在学习算法的培训过程中选择最佳特征子集作为其延伸部分。

理想情况下，我们会在本章节中描述所有三种方法。然而，由于嵌入方法与特定的学习算法紧密相连，要在深入探讨算法本身之前解释它们是困难的。因此，在本章中，我们仅涵盖过滤和包装特征选择方法，将嵌入方法的讨论留到那些深入讨论这些学习算法的章节中。

# 10.1 数值特征方差阈值法

## Problem

您有一组数值特征，并希望过滤掉那些方差低（即可能包含较少信息）的特征。

## Solution

选择方差高于给定阈值的特征子集：

```py
# Load libraries
from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold

# Import some data to play with
iris = datasets.load_iris()

# Create features and target
features = iris.data
target = iris.target

# Create thresholder
thresholder = VarianceThreshold(threshold=.5)

# Create high variance feature matrix
features_high_variance = thresholder.fit_transform(features)

# View high variance feature matrix
features_high_variance[0:3]
```

```py
array([[ 5.1,  1.4,  0.2],
       [ 4.9,  1.4,  0.2],
       [ 4.7,  1.3,  0.2]])
```

## Discussion

*方差阈值法*（VT）是一种通过过滤进行特征选择的示例，也是特征选择的最基本方法之一。其动机是低方差特征可能不太有趣（并且不太有用），而高方差特征可能更有趣。VT 首先计算每个特征的方差：

<math display="block"><mrow><mrow><mi>V</mi> <mi>a</mi> <mi>r</mi></mrow> <mrow><mo>(</mo> <mi>x</mi> <mo>)</mo></mrow> <mo>=</mo> <mfrac><mn>1</mn> <mi>n</mi></mfrac> <munderover><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>n</mi></munderover> <msup><mrow><mo>(</mo><msub><mi>x</mi> <mi>i</mi></msub> <mo>-</mo><mi>μ</mi><mo>)</mo></mrow> <mn>2</mn></msup></mrow></math>

其中 <math display="inline"><mi>x</mi></math> 是特征向量，<math display="inline"><msub><mi>x</mi><mi>i</mi></msub></math> 是单个特征值，<math display="inline"><mi>μ</mi></math> 是该特征的平均值。接下来，它删除所有方差未达到该阈值的特征。

在使用 VT 时要牢记两点。首先，方差未居中；即，它位于特征本身的平方单位中。因此，当特征集包含不同单位时（例如，一个特征以年为单位，而另一个特征以美元为单位），VT 将无法正常工作。其次，方差阈值是手动选择的，因此我们必须凭借自己的判断来选择一个合适的值（或者使用第十二章中描述的模型选择技术）。我们可以使用 `variances_` 查看每个特征的方差：

```py
# View variances
thresholder.fit(features).variances_
```

```py
array([0.68112222, 0.18871289, 3.09550267, 0.57713289])
```

最后，如果特征已经标准化（均值为零，方差为单位），那么很显然 VT 将无法正确工作：

```py
# Load library
from sklearn.preprocessing import StandardScaler

# Standardize feature matrix
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# Caculate variance of each feature
selector = VarianceThreshold()
selector.fit(features_std).variances_
```

```py
array([1., 1., 1., 1.])
```

# 10.2 二进制特征方差的阈值处理

## 问题

您拥有一组二进制分类特征，并希望过滤掉方差低的特征（即可能包含少量信息）。

## 解决方案

选择一个伯努利随机变量方差高于给定阈值的特征子集：

```py
# Load library
from sklearn.feature_selection import VarianceThreshold

# Create feature matrix with:
# Feature 0: 80% class 0
# Feature 1: 80% class 1
# Feature 2: 60% class 0, 40% class 1
features = [[0, 1, 0],
            [0, 1, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0]]

# Run threshold by variance
thresholder = VarianceThreshold(threshold=(.75 * (1 - .75)))
thresholder.fit_transform(features)
```

```py
array([[0],
       [1],
       [0],
       [1],
       [0]])
```

## 讨论

与数值特征类似，选择高信息二分类特征并过滤掉信息较少的策略之一是检查它们的方差。在二进制特征（即伯努利随机变量）中，方差计算如下：

<math display="block"><mrow><mo form="prefix">Var</mo> <mo>(</mo> <mi>x</mi> <mo>)</mo> <mo>=</mo> <mi>p</mi> <mo>(</mo> <mn>1</mn> <mo>-</mo> <mi>p</mi> <mo>)</mo></mrow></math>

其中 <math display="inline"><mi>p</mi></math> 是类 `1` 观察值的比例。因此，通过设置 <math display="inline"><mi>p</mi></math>，我们可以移除大多数观察值为一类的特征。

# 10.3 处理高度相关的特征

## 问题

您有一个特征矩阵，并怀疑某些特征之间高度相关。

## 解决方案

使用相关性矩阵检查高度相关特征。如果存在高度相关的特征，请考虑删除其中一个：

```py
# Load libraries
import pandas as pd
import numpy as np

# Create feature matrix with two highly correlated features
features = np.array([[1, 1, 1],
                     [2, 2, 0],
                     [3, 3, 1],
                     [4, 4, 0],
                     [5, 5, 1],
                     [6, 6, 0],
                     [7, 7, 1],
                     [8, 7, 0],
                     [9, 7, 1]])

# Convert feature matrix into DataFrame
dataframe = pd.DataFrame(features)

# Create correlation matrix
corr_matrix = dataframe.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
                          k=1).astype(bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Drop features
dataframe.drop(dataframe.columns[to_drop], axis=1).head(3)
```

|  | 0 | 2 |
| --- | --- | --- |
| 0 | 1 | 1 |
| 1 | 2 | 0 |
| 2 | 3 | 1 |

## 讨论

在机器学习中，我们经常遇到的一个问题是高度相关的特征。如果两个特征高度相关，那么它们所包含的信息非常相似，同时包含这两个特征很可能是多余的。对于像线性回归这样简单的模型，如果不移除这些特征，则违反了线性回归的假设，并可能导致人为膨胀的 R-squared 值。解决高度相关特征的方法很简单：从特征集中删除其中一个特征。通过设置相关性阈值来移除高度相关特征是另一种筛选的例子。

在我们的解决方案中，首先我们创建了所有特征的相关性矩阵：

```py
# Correlation matrix
dataframe.corr()
```

|  | 0 | 1 | 2 |
| --- | --- | --- | --- |
| 0 | 1.000000 | 0.976103 | 0.000000 |
| 1 | 0.976103 | 1.000000 | -0.034503 |
| 2 | 0.000000 | -0.034503 | 1.000000 |

接着，我们查看相关性矩阵的上三角来识别高度相关特征的成对：

```py
# Upper triangle of correlation matrix
upper
```

|  | 0 | 1 | 2 |
| --- | --- | --- | --- |
| 0 | NaN | 0.976103 | 0.000000 |
| 1 | NaN | NaN | 0.034503 |
| 2 | NaN | NaN | NaN |

其次，我们从这些成对特征中移除一个特征。

# 10.4 删除分类中无关紧要的特征

## 问题

您有一个分类目标向量，并希望删除无信息的特征。

## 解决方案

如果特征是分类的，请计算每个特征与目标向量之间的卡方统计量（<math display="inline"><msup><mi>χ</mi> <mn>2</mn></msup></math>)：

```py
# Load libraries
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif

# Load data
iris = load_iris()
features = iris.data
target = iris.target

# Convert to categorical data by converting data to integers
features = features.astype(int)

# Select two features with highest chi-squared statistics
chi2_selector = SelectKBest(chi2, k=2)
features_kbest = chi2_selector.fit_transform(features, target)

# Show results
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kbest.shape[1])
```

```py
Original number of features: 4
Reduced number of features: 2
```

如果特征是数量的，请计算每个特征与目标向量之间的 ANOVA F 值：

```py
# Select two features with highest F-values
fvalue_selector = SelectKBest(f_classif, k=2)
features_kbest = fvalue_selector.fit_transform(features, target)

# Show results
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kbest.shape[1])
```

```py
Original number of features: 4
Reduced number of features: 2
```

而不是选择特定数量的特征，我们可以使用`SelectPercentile`来选择顶部*n*百分比的特征：

```py
# Load library
from sklearn.feature_selection import SelectPercentile

# Select top 75% of features with highest F-values
fvalue_selector = SelectPercentile(f_classif, percentile=75)
features_kbest = fvalue_selector.fit_transform(features, target)

# Show results
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kbest.shape[1])
```

```py
Original number of features: 4
Reduced number of features: 3
```

## 讨论

卡方统计检验两个分类向量的独立性。也就是说，统计量是类别特征中每个类别的观察次数与如果该特征与目标向量独立（即没有关系）时预期的观察次数之间的差异：

<math display="block"><mrow><msup><mi>χ</mi> <mn>2</mn></msup> <mo>=</mo> <munderover><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>n</mi></munderover> <mfrac><msup><mrow><mo>(</mo><msub><mi>O</mi> <mi>i</mi></msub> <mo>-</mo><msub><mi>E</mi> <mi>i</mi></msub> <mo>)</mo></mrow> <mn>2</mn></msup> <msub><mi>E</mi> <mi>i</mi></msub></mfrac></mrow></math>

其中<math display="inline"><msub><mi>O</mi><mi>i</mi></msub></math>是类别<math display="inline"><mi>i</mi></math>中观察到的观测次数，<math display="inline"><msub><mi>E</mi><mi>i</mi></msub></math>是类别<math display="inline"><mi>i</mi></math>中预期的观测次数。

卡方统计量是一个单一的数字，它告诉您观察计数和在整体人群中如果没有任何关系时预期计数之间的差异有多大。通过计算特征和目标向量之间的卡方统计量，我们可以得到两者之间独立性的度量。如果目标与特征变量无关，那么对我们来说它是无关紧要的，因为它不包含我们可以用于分类的信息。另一方面，如果两个特征高度依赖，它们可能对训练我们的模型非常有信息性。

要在特征选择中使用卡方，我们计算每个特征与目标向量之间的卡方统计量，然后选择具有最佳卡方统计量的特征。在 scikit-learn 中，我们可以使用`SelectKBest`来选择它们。参数`k`确定我们想要保留的特征数，并过滤掉信息最少的特征。

需要注意的是，卡方统计只能在两个分类向量之间计算。因此，特征选择的卡方要求目标向量和特征都是分类的。然而，如果我们有一个数值特征，我们可以通过首先将定量特征转换为分类特征来使用卡方技术。最后，为了使用我们的卡方方法，所有值都需要是非负的。

或者，如果我们有一个数值特征，我们可以使用`f_classif`来计算 ANOVA F 值统计量和每个特征以及目标向量的相关性。F 值分数检查如果我们按照目标向量对数值特征进行分组，每个组的平均值是否显著不同。例如，如果我们有一个二进制目标向量，性别和一个定量特征，测试分数，F 值将告诉我们男性的平均测试分数是否与女性的平均测试分数不同。如果不是，则测试分数对我们预测性别没有帮助，因此该特征是无关的。

# 10.5 递归消除特征

## 问题

你想要自动选择保留的最佳特征。

## 解决方案

使用 scikit-learn 的`RFECV`进行*递归特征消除*（RFE），使用交叉验证（CV）。也就是说，使用包装器特征选择方法，重复训练模型，每次删除一个特征，直到模型性能（例如准确性）变差。剩下的特征就是最好的：

```py
# Load libraries
import warnings
from sklearn.datasets import make_regression
from sklearn.feature_selection import RFECV
from sklearn import datasets, linear_model

# Suppress an annoying but harmless warning
warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")

# Generate features matrix, target vector, and the true coefficients
features, target = make_regression(n_samples = 10000,
                                   n_features = 100,
                                   n_informative = 2,
                                   random_state = 1)

# Create a linear regression
ols = linear_model.LinearRegression()

# Recursively eliminate features
rfecv = RFECV(estimator=ols, step=1, scoring="neg_mean_squared_error")
rfecv.fit(features, target)
rfecv.transform(features)
```

```py
array([[ 0.00850799,  0.7031277 ,  1.52821875],
       [-1.07500204,  2.56148527, -0.44567768],
       [ 1.37940721, -1.77039484, -0.74675125],
       ...,
       [-0.80331656, -1.60648007,  0.52231601],
       [ 0.39508844, -1.34564911,  0.4228057 ],
       [-0.55383035,  0.82880112,  1.73232647]])
```

一旦我们进行了 RFE，我们就可以看到我们应该保留的特征数量：

```py
# Number of best features
rfecv.n_features_
```

```py
3
```

我们还可以看到哪些特征应该保留：

```py
# Which categories are best
rfecv.support_
```

```py
array([False, False, False, False, False,  True, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False,  True, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False,  True, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       False])
```

我们甚至可以查看特征的排名：

```py
# Rank features best (1) to worst
rfecv.ranking_
```

```py
array([11, 92, 96, 87, 46,  1, 48, 23, 16,  2, 66, 83, 33, 27, 70, 75, 29,
       84, 54, 88, 37, 42, 85, 62, 74, 50, 80, 10, 38, 59, 79, 57, 44,  8,
       82, 45, 89, 69, 94,  1, 35, 47, 39,  1, 34, 72, 19,  4, 17, 91, 90,
       24, 32, 13, 49, 26, 12, 71, 68, 40,  1, 43, 63, 28, 73, 58, 21, 67,
        1, 95, 77, 93, 22, 52, 30, 60, 81, 14, 86, 18, 15, 41,  7, 53, 65,
       51, 64,  6,  9, 20,  5, 55, 56, 25, 36, 61, 78, 31,  3, 76])
```

## 讨论

这可能是本书到目前为止最复杂的配方，结合了一些我们尚未详细讨论的主题。然而，直觉足够简单，我们可以在这里解释它，而不是推迟到以后的章节。RFE 背后的想法是重复训练模型，每次更新该模型的*权重*或*系数*。第一次训练模型时，我们包括所有特征。然后，我们找到具有最小参数的特征（请注意，这假设特征已经重新缩放或标准化），意味着它不太重要，并从特征集中删除该特征。

那么显而易见的问题是：我们应该保留多少特征？我们可以（假设性地）重复此循环，直到我们只剩下一个特征。更好的方法要求我们包括一个新概念叫*交叉验证*。我们将在下一章详细讨论 CV，但这里是一般的想法。

给定包含（1）我们想要预测的目标和（2）特征矩阵的数据，首先我们将数据分为两组：一个训练集和一个测试集。其次，我们使用训练集训练我们的模型。第三，我们假装不知道测试集的目标，并将我们的模型应用于其特征以预测测试集的值。最后，我们将我们预测的目标值与真实的目标值进行比较，以评估我们的模型。

我们可以使用 CV 找到在 RFE 期间保留的最佳特征数。具体而言，在带有 CV 的 RFE 中，每次迭代后我们使用交叉验证评估我们的模型。如果 CV 显示在我们消除一个特征后模型改善了，那么我们继续下一个循环。然而，如果 CV 显示在我们消除一个特征后模型变差了，我们将该特征重新放回特征集，并选择这些特征作为最佳特征。

在 scikit-learn 中，使用`RFECV`实现了带有多个重要参数的 RFE 与 CV。`estimator`参数确定我们想要训练的模型类型（例如线性回归），`step`参数在每个循环中设置要删除的特征数量或比例，`scoring`参数设置我们在交叉验证期间用于评估模型质量的度量标准。

## 参见

+   [scikit-learn 文档：带交叉验证的递归特征消除](https://oreil.ly/aV-Fz)
