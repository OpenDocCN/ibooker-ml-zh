# 第十六章 Logistic 回归

# 16.0 引言

尽管其名称中带有“回归”，*逻辑回归* 实际上是一种广泛使用的监督分类技术。逻辑回归（及其扩展，如多项式逻辑回归）是一种直接、被理解的方法，用于预测观察值属于某个类别的概率。在本章中，我们将涵盖在 scikit-learn 中使用逻辑回归训练各种分类器的过程。

# 16.1 训练一个二元分类器

## 问题

您需要训练一个简单的分类器模型。

## 解决方案

使用 `LogisticRegression` 在 scikit-learn 中训练一个逻辑回归模型：

```py
# Load libraries
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Load data with only two classes
iris = datasets.load_iris()
features = iris.data[:100,:]
target = iris.target[:100]

# Standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Create logistic regression object
logistic_regression = LogisticRegression(random_state=0)

# Train model
model = logistic_regression.fit(features_standardized, target)
```

## 讨论

尽管其名称中带有“回归”，逻辑回归实际上是一种广泛使用的二元分类器（即目标向量只能取两个值）。在逻辑回归中，线性模型（例如*β[0] + β[1]x*）包含在逻辑（也称为 sigmoid）函数中，<math display="inline"><mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi> <mrow><mo>-</mo><mi>z</mi></mrow></msup></mrow></mfrac></math>，使得：

<math display="block"><mrow><mi>P</mi> <mrow><mo>(</mo> <msub><mi>y</mi> <mi>i</mi></msub> <mo>=</mo> <mn>1</mn> <mo>∣</mo> <mi>X</mi> <mo>)</mo></mrow> <mo>=</mo> <mfrac><mn>1</mn> <mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi> <mrow><mo>-</mo><mo>(</mo><msub><mi>β</mi> <mn>0</mn></msub> <mo>+</mo><msub><mi>β</mi> <mn>1</mn></msub> <mi>x</mi><mo>)</mo></mrow></msup></mrow></mfrac></mrow></math>

其中<math display="inline"><mi>P</mi><mrow><mo>(</mo><msub><mi>y</mi> <mi>i</mi></msub> <mo>=</mo><mn>1</mn><mo>∣</mo><mi>X</mi><mo>)</mo></mrow></math>是第<math display="inline"><mi>i</mi></math>个观察目标值<math display="inline"><msub><mi>y</mi><mi>i</mi></msub></math>为类别 1 的概率；<math display="inline"><mi>X</mi></math>是训练数据；<math display="inline"><msub><mi>β</mi><mn>0</mn></msub></math>和<math display="inline"><msub><mi>β</mi><mn>1</mn></msub></math>是待学习的参数；<math display="inline"><mi>e</mi></math>是自然常数。逻辑函数的效果是将函数的输出值限制在 0 到 1 之间，因此可以解释为概率。如果<math display="inline"><mi>P</mi><mrow><mo>(</mo><msub><mi>y</mi> <mi>i</mi></msub> <mo>=</mo><mn>1</mn><mo>∣</mo><mi>X</mi><mo>)</mo></mrow></math>大于 0.5，则预测为类别 1；否则，预测为类别 0。

在 scikit-learn 中，我们可以使用 `LogisticRegression` 训练一个逻辑回归模型。一旦训练完成，我们可以使用该模型预测新观察的类别：

```py
# Create new observation
new_observation = [[.5, .5, .5, .5]]

# Predict class
model.predict(new_observation)
```

```py
array([1])
```

在这个例子中，我们的观察被预测为类别 1。此外，我们可以看到观察为每个类的成员的概率：

```py
# View predicted probabilities
model.predict_proba(new_observation)
```

```py
array([[0.17738424, 0.82261576]])
```

我们的观察有 17.7%的机会属于类别 0，82.2%的机会属于类别 1。

# 16.2 训练一个多类分类器

## 问题

如果超过两个类别，则需要训练一个分类器模型。

## 解决方案

使用 `LogisticRegression` 在 scikit-learn 中训练一个逻辑回归，使用一对多或多项式方法：

```py
# Load libraries
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Create one-vs-rest logistic regression object
logistic_regression = LogisticRegression(random_state=0, multi_class="ovr")

# Train model
model = logistic_regression.fit(features_standardized, target)
```

## 讨论

单独来看，逻辑回归只是二元分类器，意味着它不能处理目标向量超过两个类。然而，逻辑回归的两个巧妙扩展可以做到。首先，在 *一对多* 逻辑回归（OvR）中，为每个预测的类别训练一个单独的模型，无论观察结果是否属于该类（从而将其转化为二元分类问题）。它假设每个分类问题（例如，类别 0 或非类别 0）是独立的。

或者，*多项式逻辑回归*（MLR）中，我们在 配方 16.1 中看到的逻辑函数被 softmax 函数取代：

<math display="block"><mrow><mi>P</mi> <mrow><mo>(</mo> <msub><mi>y</mi> <mi>i</mi></msub> <mo>=</mo> <mi>k</mi> <mo>∣</mo> <mi>X</mi> <mo>)</mo></mrow> <mo>=</mo> <mfrac><msup><mi>e</mi> <mrow><msub><mi>β</mi> <mi>k</mi></msub> <msub><mi>x</mi> <mi>i</mi></msub></mrow></msup> <mrow><msubsup><mo>∑</mo> <mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow> <mi>K</mi></msubsup> <msup><mi>e</mi> <mrow><msub><mi>β</mi> <mi>j</mi></msub> <msub><mi>x</mi> <mi>i</mi></msub></mrow></msup></mrow></mfrac></mrow></math>

其中 <math display="inline"><mi>P</mi><mrow><mo>(</mo><msub><mi>y</mi> <mi>i</mi></msub> <mo>=</mo><mi>k</mi><mo>∣</mo><mi>X</mi><mo>)</mo></mrow></math> 是第 <math display="inline"><mi>i</mi></math> 个观察目标值 <math display="inline"><msub><mi>y</mi> <mi>i</mi></msub></math> 属于类别 <math display="inline"><mi>k</mi></math> 的概率，<math display="inline"><mi>K</mi></math> 是总类别数。MLR 的一个实际优势是，使用 `predict_proba` 方法预测的概率更可靠（即更好地校准）。

当使用 `LogisticRegression` 时，我们可以选择我们想要的两种技术之一，OvR (`ovr`) 是默认参数。我们可以通过设置参数为 `multinomial` 切换到 MLR。

# 16.3 通过正则化减少方差

## 问题

你需要减少逻辑回归模型的方差。

## 解决方案

调整正则化强度超参数 `C`：

```py
# Load libraries
from sklearn.linear_model import LogisticRegressionCV
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Create decision tree regression object
logistic_regression = LogisticRegressionCV(
    penalty='l2', Cs=10, random_state=0, n_jobs=-1)

# Train model
model = logistic_regression.fit(features_standardized, target)
```

## 讨论

*正则化* 是一种惩罚复杂模型以减少其方差的方法。具体来说，是向我们试图最小化的损失函数中添加一个惩罚项，通常是 L1 和 L2 惩罚。在 L1 惩罚中：

<math display="block"><mrow><mi>α</mi> <munderover><mo>∑</mo> <mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow> <mi>p</mi></munderover> <mfenced close="|" open="|" separators=""><msub><mover accent="true"><mi>β</mi> <mo>^</mo></mover> <mi>j</mi></msub></mfenced></mrow></math>

其中 <math display="inline"><msub><mover accent="true"><mi>β</mi><mo>^</mo></mover> <mi>j</mi></msub></math> 是正在学习的第 <math display="inline"><mi>j</mi></math> 个特征的参数，<math display="inline"><mi>α</mi></math> 是表示正则化强度的超参数。使用 L2 惩罚时：

<math display="block"><mrow><mi>α</mi> <munderover><mo>∑</mo> <mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow> <mi>p</mi></munderover> <msup><msub><mover accent="true"><mi>β</mi> <mo>^</mo></mover> <mi>j</mi></msub> <mn>2</mn></msup></mrow></math>

较高的 <math display="inline"><mi>α</mi></math> 值增加了较大参数值的惩罚（即更复杂的模型）。scikit-learn 遵循使用 <math display="inline"><mi>C</mi></math> 而不是 <math display="inline"><mi>α</mi></math> 的常见方法，其中 <math display="inline"><mi>C</mi></math> 是正则化强度的倒数：<math display="inline"><mrow><mi>C</mi><mo>=</mo><mfrac><mn>1</mn> <mi>α</mi></mfrac></mrow></math>。为了在使用逻辑回归时减少方差，我们可以将 <math display="inline"><mi>C</mi></math> 视为一个超参数，用于调整以找到创建最佳模型的 <math display="inline"><mi>C</mi></math> 的值。在 scikit-learn 中，我们可以使用 `LogisticRegressionCV` 类来高效地调整 <math display="inline"><mi>C</mi></math>。`LogisticRegressionCV` 的参数 `Cs` 可以接受一个值范围供 <math display="inline"><mi>C</mi></math> 搜索（如果提供一个浮点数列表作为参数），或者如果提供一个整数，则会在对数尺度的 -10,000 到 10,000 之间生成相应数量的候选值列表。

不幸的是，`LogisticRegressionCV` 不允许我们在不同的惩罚项上搜索。为了做到这一点，我们必须使用在 第十二章 讨论的效率较低的模型选择技术。

# 16.4 在非常大的数据上训练分类器

## 问题

您需要在非常大的数据集上训练一个简单的分类器模型。

## 解决方案

使用 *stochastic average gradient*（SAG）求解器在 scikit-learn 中训练逻辑回归：

```py
# Load libraries
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Create logistic regression object
logistic_regression = LogisticRegression(random_state=0, solver="sag")

# Train model
model = logistic_regression.fit(features_standardized, target)
```

## 讨论

scikit-learn 的 `LogisticRegression` 提供了一些训练逻辑回归的技术，称为 *solvers*。大多数情况下，scikit-learn 会自动为我们选择最佳的求解器，或者警告我们无法使用某个求解器来做某事。然而，有一个特定的情况我们应该注意。

尽管详细解释超出了本书的范围（更多信息请参见 Mark Schmidt 在本章 “参见” 部分的幻灯片），随机平均梯度下降使我们能够在数据非常大时比其他求解器更快地训练模型。然而，它对特征缩放非常敏感，因此标准化我们的特征特别重要。我们可以通过设置 `solver="sag"` 来让我们的学习算法使用这个求解器。

## 参见

+   [使用随机平均梯度算法最小化有限和，Mark Schmidt](https://oreil.ly/K5rEG)

# 16.5 处理不平衡的类

## 问题

您需要训练一个简单的分类器模型。

## 解决方案

使用 scikit-learn 中的 `LogisticRegression` 训练逻辑回归模型：

```py
# Load libraries
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Make class highly imbalanced by removing first 40 observations
features = features[40:,:]
target = target[40:]

# Create target vector indicating if class 0, otherwise 1
target = np.where((target == 0), 0, 1)

# Standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Create decision tree regression object
logistic_regression = LogisticRegression(random_state=0, class_weight="balanced")

# Train model
model = logistic_regression.fit(features_standardized, target)
```

## 讨论

就像 `scikit-learn` 中许多其他学习算法一样，`LogisticRegression` 自带处理不平衡类别的方法。如果我们的类别高度不平衡，在预处理过程中没有处理它，我们可以使用 `class_weight` 参数来加权这些类别，以确保每个类别的混合平衡。具体地，`balanced` 参数将自动根据其频率的倒数加权类别：

<math display="block"><mrow><msub><mi>w</mi> <mi>j</mi></msub> <mo>=</mo> <mfrac><mi>n</mi> <mrow><mi>k</mi><msub><mi>n</mi> <mi>j</mi></msub></mrow></mfrac></mrow></math>

其中 <math display="inline"><msub><mi>w</mi><mi>j</mi></msub></math> 是类别 <math display="inline"><mi>j</mi></math> 的权重，<math display="inline"><mi>n</mi></math> 是观测数量，<math display="inline"><msub><mi>n</mi><mi>j</mi></msub></math> 是类别 <math display="inline"><mi>j</mi></math> 中的观测数量，<math display="inline"><mi>k</mi></math> 是总类别数量。
