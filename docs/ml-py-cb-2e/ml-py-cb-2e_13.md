# 第十三章：线性回归

# 13.0 引言

*线性回归*是我们工具箱中最简单的监督学习算法之一。如果您曾经在大学里修过入门统计课程，很可能您最后学到的主题就是线性回归。线性回归及其扩展在当目标向量是定量值（例如房价、年龄）时继续是一种常见且有用的预测方法。在本章中，我们将涵盖多种线性回归方法（及其扩展）来创建性能良好的预测模型。

# 13.1 拟合一条线

## 问题

您希望训练一个能够表示特征和目标向量之间线性关系的模型。

## 解决方案

使用线性回归（在 scikit-learn 中，`LinearRegression`）：

```py
# Load libraries
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Generate features matrix, target vector
features, target = make_regression(n_samples = 100,
                                   n_features = 3,
                                   n_informative = 2,
                                   n_targets = 1,
                                   noise = 0.2,
                                   coef = False,
                                   random_state = 1)

# Create linear regression
regression = LinearRegression()

# Fit the linear regression
model = regression.fit(features, target)
```

## 讨论

线性回归假设特征与目标向量之间的关系大致是线性的。也就是说，特征对目标向量的*效果*（也称为*系数*、*权重*或*参数*）是恒定的。为了解释起见，在我们的解决方案中，我们只使用了三个特征来训练我们的模型。这意味着我们的线性模型将是：

<math display="block"><mrow><mover accent="true"><mi>y</mi> <mo>^</mo></mover> <mo>=</mo> <msub><mover accent="true"><mi>β</mi> <mo>^</mo></mover> <mn>0</mn></msub> <mo>+</mo> <msub><mover accent="true"><mi>β</mi> <mo>^</mo></mover> <mn>1</mn></msub> <msub><mi>x</mi> <mn>1</mn></msub> <mo>+</mo> <msub><mover accent="true"><mi>β</mi> <mo>^</mo></mover> <mn>2</mn></msub> <msub><mi>x</mi> <mn>2</mn></msub> <mo>+</mo> <msub><mover accent="true"><mi>β</mi> <mo>^</mo></mover> <mn>3</mn></msub> <msub><mi>x</mi> <mn>3</mn></msub> <mo>+</mo> <mi>ϵ</mi></mrow></math>

这里，<math display="inline"><mover accent="true"><mi>y</mi> <mo>^</mo></mover></math> 是我们的目标，<math display="inline"><msub><mi>x</mi><mi>i</mi></msub></math> 是单个特征的数据，<math display="inline"><msub><mover accent="true"><mi>β</mi> <mo>^</mo></mover> <mn>1</mn></msub></math>，<math display="inline"><msub><mover accent="true"><mi>β</mi><mo>^</mo></mover> <mn>2</mn></msub></math>和<math display="inline"><msub><mover accent="true"><mi>β</mi><mo>^</mo></mover> <mn>3</mn></msub></math>是通过拟合模型确定的系数，<math display="inline"><mi>ϵ</mi></math>是误差。在拟合模型后，我们可以查看每个参数的值。例如，<math display="inline"><msub><mover accent="true"><mi>β</mi> <mo>^</mo></mover> <mn>0</mn></msub></math>，也称为*偏差*或*截距*，可以使用`intercept_`查看：

```py
# View the intercept
model.intercept_
```

```py
-0.009650118178816669
```

而`coef_`显示了<math display="inline"><msub><mover accent="true"><mi>β</mi><mo>^</mo></mover> <mn>1</mn></msub></math>和<math display="inline"><msub><mover accent="true"><mi>β</mi><mo>^</mo></mover> <mn>2</mn></msub></math>：

```py
# View the feature coefficients
model.coef_
```

```py
array([1.95531234e-02, 4.42087450e+01, 5.81494563e+01])
```

在我们的数据集中，目标值是一个随机生成的连续变量：

```py
# First value in the target vector
target[0]
```

```py
-20.870747595269407
```

使用`predict`方法，我们可以根据输入特征预测输出：

```py
# Predict the target value of the first observation
model.predict(features)[0]
```

```py
-20.861927709296808
```

不错！我们的模型只偏离了约 0.01！

线性回归的主要优势在于其可解释性，这在很大程度上是因为模型的系数是目标向量一单位变化的影响。我们模型的第一个特征的系数约为~–0.02，这意味着我们在第一个特征每增加一个单位时目标的变化。

使用`score`函数，我们还可以看到我们的模型在数据上的表现：

```py
# Print the score of the model on the training data
print(model.score(features, target))
```

```py
0.9999901732607787
```

scikit learn 中线性回归的默认得分是 R²，范围从 0.0（最差）到 1.0（最好）。正如我们在这个例子中所看到的，我们非常接近完美值 1.0。然而值得注意的是，我们是在模型已经见过的数据（训练数据）上评估该模型，而通常我们会在一个独立的测试数据集上进行评估。尽管如此，在实际情况下，这样高的分数对我们的模型是个好兆头。

# 13.2 处理交互效应

## 问题

你有一个特征，其对目标变量的影响取决于另一个特征。

## 解决方案

创建一个交互项来捕获这种依赖关系，使用 scikit-learn 的 `PolynomialFeatures`：

```py
# Load libraries
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import make_regression

# Generate features matrix, target vector
features, target = make_regression(n_samples = 100,
                                   n_features = 2,
                                   n_informative = 2,
                                   n_targets = 1,
                                   noise = 0.2,
                                   coef = False,
                                   random_state = 1)

# Create interaction term
interaction = PolynomialFeatures(
    degree=3, include_bias=False, interaction_only=True)
features_interaction = interaction.fit_transform(features)

# Create linear regression
regression = LinearRegression()

# Fit the linear regression
model = regression.fit(features_interaction, target)
```

## 讨论

有时，一个特征对目标变量的影响至少部分依赖于另一个特征。例如，想象一个简单的基于咖啡的例子，我们有两个二进制特征——是否加糖（`sugar`）和是否搅拌（`stirred`）——我们想预测咖啡是否甜。仅仅加糖（`sugar=1, stirred=0`）不会使咖啡变甜（所有的糖都在底部！），仅仅搅拌咖啡而不加糖（`sugar=0, stirred=1`）也不会使其变甜。实际上，是将糖放入咖啡并搅拌（`sugar=1, stirred=1`）才能使咖啡变甜。`sugar` 和 `stirred` 对甜味的影响是相互依赖的。在这种情况下，我们称` sugar` 和 `stirred` 之间存在*交互效应*。

我们可以通过包含一个新特征来考虑交互效应，该特征由交互特征的相应值的乘积组成：

<math display="block"><mrow><mover accent="true"><mi>y</mi> <mo>^</mo></mover> <mo>=</mo> <msub><mover accent="true"><mi>β</mi> <mo>^</mo></mover> <mn>0</mn></msub> <mo>+</mo> <msub><mover accent="true"><mi>β</mi> <mo>^</mo></mover> <mn>1</mn></msub> <msub><mi>x</mi> <mn>1</mn></msub> <mo>+</mo> <msub><mover accent="true"><mi>β</mi> <mo>^</mo></mover> <mn>2</mn></msub> <msub><mi>x</mi> <mn>2</mn></msub> <mo>+</mo> <msub><mover accent="true"><mi>β</mi> <mo>^</mo></mover> <mn>3</mn></msub> <msub><mi>x</mi> <mn>1</mn></msub> <msub><mi>x</mi> <mn>2</mn></msub> <mo>+</mo> <mi>ϵ</mi></mrow></math>

其中 <math display="inline"><msub><mi>x</mi><mn>1</mn></msub></math> 和 <math display="inline"><msub><mi>x</mi><mn>2</mn></msub></math> 分别是 `sugar` 和 `stirred` 的值，<math display="inline"><msub><mi>x</mi><mn>1</mn></msub><msub><mi>x</mi><mn>2</mn></msub></math> 表示两者之间的交互作用。

在我们的解决方案中，我们使用了一个只包含两个特征的数据集。以下是每个特征的第一个观察值：

```py
# View the feature values for first observation
features[0]
```

```py
array([0.0465673 , 0.80186103])
```

要创建一个交互项，我们只需为每个观察值将这两个值相乘：

```py
# Import library
import numpy as np

# For each observation, multiply the values of the first and second feature
interaction_term = np.multiply(features[:, 0], features[:, 1])
```

我们可以看到第一次观察的交互项：

```py
# View interaction term for first observation
interaction_term[0]
```

```py
0.037340501965846186
```

然而，虽然我们经常有充分的理由相信两个特征之间存在交互作用，但有时我们也没有。在这些情况下，使用 scikit-learn 的 `PolynomialFeatures` 为所有特征组合创建交互项会很有用。然后，我们可以使用模型选择策略来识别产生最佳模型的特征组合和交互项。

要使用`PolynomialFeatures`创建交互项，我们需要设置三个重要的参数。最重要的是，`interaction_only=True`告诉`PolynomialFeatures`仅返回交互项（而不是多项式特征，我们将在 Recipe 13.3 中讨论）。默认情况下，`PolynomialFeatures`会添加一个名为*bias*的包含 1 的特征。我们可以通过`include_bias=False`来防止这种情况发生。最后，`degree`参数确定从中创建交互项的特征的最大数量（以防我们想要创建的交互项是三个特征的组合）。我们可以通过检查我们的解决方案中`PolynomialFeatures`的输出，看看第一个观察值的特征值和交互项值是否与我们手动计算的版本匹配：

```py
# View the values of the first observation
features_interaction[0]
```

```py
array([0.0465673 , 0.80186103, 0.0373405 ])
```

# 13.3 拟合非线性关系

## 问题

您希望对非线性关系进行建模。

## 解决方案

通过在线性回归模型中包含多项式特征来创建多项式回归：

```py
# Load library
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import make_regression

# Generate features matrix, target vector
features, target = make_regression(n_samples = 100,
                                   n_features = 3,
                                   n_informative = 2,
                                   n_targets = 1,
                                   noise = 0.2,
                                   coef = False,
                                   random_state = 1)

# Create polynomial features x² and x³
polynomial = PolynomialFeatures(degree=3, include_bias=False)
features_polynomial = polynomial.fit_transform(features)

# Create linear regression
regression = LinearRegression()

# Fit the linear regression
model = regression.fit(features_polynomial, target)
```

## 讨论

到目前为止，我们只讨论了建模线性关系。线性关系的一个例子是建筑物的层数与建筑物的高度之间的关系。在线性回归中，我们假设层数和建筑物高度的影响大致是恒定的，这意味着一个 20 层的建筑物大致会比一个 10 层的建筑物高出两倍，而一个 5 层的建筑物大致会比一个 10 层的建筑物高出两倍。然而，许多感兴趣的关系并不严格是线性的。

我们经常希望建模非线性关系，例如学生学习时间与她在考试中得分之间的关系。直觉上，我们可以想象，对于一个小时的学习和没有学习的学生之间的考试成绩差异很大。然而，在学习时间增加到 99 小时和 100 小时之间时，学生的考试成绩差异就会变得很小。随着学习小时数的增加，一个小时的学习对学生考试成绩的影响逐渐减小。

多项式回归是线性回归的扩展，允许我们建模非线性关系。要创建多项式回归，将我们在 Recipe 13.1 中使用的线性函数转换为多项式函数：

<math display="block"><mrow><mover accent="true"><mi>y</mi> <mo>^</mo></mover> <mo>=</mo> <msub><mover accent="true"><mi>β</mi> <mo>^</mo></mover> <mn>0</mn></msub> <mo>+</mo> <msub><mover accent="true"><mi>β</mi> <mo>^</mo></mover> <mn>1</mn></msub> <msub><mi>x</mi> <mn>1</mn></msub> <mo>+</mo> <mi>ϵ</mi></mrow></math>

通过添加多项式特征将线性回归模型扩展为多项式函数：

<math display="block"><mrow><mover accent="true"><mi>y</mi> <mo>^</mo></mover> <mo>=</mo> <msub><mover accent="true"><mi>β</mi> <mo>^</mo></mover> <mn>0</mn></msub> <mo>+</mo> <msub><mover accent="true"><mi>β</mi> <mo>^</mo></mover> <mn>1</mn></msub> <msub><mi>x</mi> <mn>1</mn></msub> <mo>+</mo> <msub><mover accent="true"><mi>β</mi> <mo>^</mo></mover> <mn>2</mn></msub> <msup><msub><mi>x</mi> <mn>1</mn></msub> <mn>2</mn></msup> <mo>+</mo> <mo>.</mo> <mo>.</mo> <mo>.</mo> <mo>+</mo> <msub><mover accent="true"><mi>β</mi> <mo>^</mo></mover> <mi>d</mi></msub> <msup><msub><mi>x</mi> <mn>1</mn></msub> <mi>d</mi></msup> <mo>+</mo> <mi>ϵ</mi></mrow></math>

其中<math display="inline"><mi>d</mi></math>是多项式的次数。我们如何能够对非线性函数使用线性回归？答案是我们不改变线性回归拟合模型的方式，而只是添加多项式特征。也就是说，线性回归并不“知道”<math display="inline"><msup><mi>x</mi><mn>2</mn></msup></math>是<math display="inline"><mi>x</mi></math>的二次转换，它只是将其视为另一个变量。

可能需要更实际的描述。为了建模非线性关系，我们可以创建将现有特征 <math display="inline"><mi>x</mi></math> 提升到某个幂次的新特征： <math display="inline"><msup><mi>x</mi><mn>2</mn></msup></math>、<math display="inline"><msup><mi>x</mi><mn>3</mn></msup></math> 等。我们添加的这些新特征越多，模型创建的“线”就越灵活。为了更加明确，想象我们想要创建一个三次多项式。为了简单起见，我们将专注于数据集中的第一个观察值：

```py
# View first observation
features[0]
```

```py
array([-0.61175641])
```

要创建一个多项式特征，我们将第一个观察值的值提升到二次方，<math display="inline"><msup><msub><mi>x</mi> <mn>1</mn></msub> <mn>2</mn></msup></math>：

```py
# View first observation raised to the second power, x²
features[0]**2
```

```py
array([0.37424591])
```

这将是我们的新功能。然后，我们还将第一个观察值的值提升到三次方，<math display="inline"><msup><msub><mi>x</mi> <mn>1</mn></msub> <mn>3</mn></msup></math>：

```py
# View first observation raised to the third power, x³
features[0]**3
```

```py
array([-0.22894734])
```

通过在我们的特征矩阵中包含所有三个特征（<math display="inline"><mi>x</mi></math>、<math display="inline"><msup><mi>x</mi><mn>2</mn></msup></math> 和 <math display="inline"><msup><mi>x</mi><mn>3</mn></msup></math>）并运行线性回归，我们进行了多项式回归：

```py
# View the first observation's values for x, x², and x³
features_polynomial[0]
```

```py
array([-0.61175641,  0.37424591, -0.22894734])
```

`PolynomialFeatures` 有两个重要参数。首先，`degree` 确定多项式特征的最大次数。例如，`degree=3` 会生成 <math display="inline"><msup><mi>x</mi><mn>2</mn></msup></math> 和 <math display="inline"><msup><mi>x</mi><mn>3</mn></msup></math>。其次，默认情况下 `PolynomialFeatures` 包括一个只包含 1 的特征（称为偏差）。我们可以通过设置 `include_bias=False` 来删除它。

# 13.4 通过正则化减少方差

## 问题

您希望减少线性回归模型的方差。

## 解决方案

使用包含*收缩惩罚*（也称为*正则化*）的学习算法，例如岭回归和拉索回归：

```py
# Load libraries
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression

# Generate features matrix, target vector
features, target = make_regression(n_samples = 100,
                                   n_features = 3,
                                   n_informative = 2,
                                   n_targets = 1,
                                   noise = 0.2,
                                   coef = False,
                                   random_state = 1)

# Standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Create ridge regression with an alpha value
regression = Ridge(alpha=0.5)

# Fit the linear regression
model = regression.fit(features_standardized, target)
```

## 讨论

在标准线性回归中，模型训练以最小化真实值（<math display="inline"><msub><mi>y</mi><mi>i</mi></msub></math>）与预测值（<math display="inline"><msub><mover accent="true"><mi>y</mi><mo>^</mo></mover> <mi>i</mi></msub></math>）目标值或残差平方和（RSS）之间的平方误差：

<math display="block"><mstyle displaystyle="true" scriptlevel="0"><mrow><mi>R</mi> <mi>S</mi> <mi>S</mi> <mo>=</mo> <munderover><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>n</mi></munderover> <msup><mrow><mo>(</mo><msub><mi>y</mi> <mi>i</mi></msub> <mo>-</mo><msub><mover accent="true"><mi>y</mi> <mo>^</mo></mover> <mi>i</mi></msub> <mo>)</mo></mrow> <mn>2</mn></msup></mrow></mstyle></math>

正则化回归学习者类似，除了它们试图最小化 RSS *和* 系数值总大小的某种惩罚，称为*收缩惩罚*，因为它试图“收缩”模型。线性回归的两种常见类型的正则化学习者是岭回归和拉索。唯一的形式上的区别是使用的收缩惩罚类型。在*岭回归*中，收缩惩罚是一个调整超参数，乘以所有系数的平方和：

<math display="block"><mrow><mtext>RSS</mtext> <mo>+</mo> <mi>α</mi> <munderover><mo>∑</mo> <mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow> <mi>p</mi></munderover> <msup><msub><mover accent="true"><mi>β</mi> <mo>^</mo></mover> <mi>j</mi></msub> <mn>2</mn></msup></mrow></math>

其中<math display="inline"><msub><mover accent="true"><mi>β</mi><mo>^</mo></mover> <mi>j</mi></msub></math>是第<math display="inline"><mi>j</mi></math>个<math display="inline"><mi>p</mi></math>特征的系数，<math display="inline"><mi>α</mi></math>是一个超参数（接下来会讨论）。*Lasso*则类似，只是收缩惩罚是一个调整的超参数，乘以所有系数的绝对值的和：

<math display="block"><mrow><mfrac><mn>1</mn> <mrow><mn>2</mn><mi>n</mi></mrow></mfrac> <mtext>RSS</mtext> <mo>+</mo> <mi>α</mi> <munderover><mo>∑</mo> <mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow> <mi>p</mi></munderover> <mfenced close="|" open="|" separators=""><msub><mover accent="true"><mi>β</mi> <mo>^</mo></mover> <mi>j</mi></msub></mfenced></mrow></math>

其中<math display="inline"><mi>n</mi></math>是观察数。那么我们应该使用哪一个？作为一个非常一般的经验法则，岭回归通常比 lasso 产生稍微更好的预测，但 lasso（我们将在 Recipe 13.5 中讨论原因）产生更可解释的模型。如果我们希望在岭回归和 lasso 的惩罚函数之间取得平衡，我们可以使用*弹性网*，它只是一个包含两种惩罚的回归模型。无论我们使用哪一个，岭回归和 lasso 回归都可以通过将系数值包括在我们试图最小化的损失函数中来对大或复杂的模型进行惩罚。

超参数<math display="inline"><mi>α</mi></math>让我们控制对系数的惩罚程度，较高的<math display="inline"><mi>α</mi></math>值会创建更简单的模型。理想的<math display="inline"><mi>α</mi></math>值应像其他超参数一样进行调整。在 scikit-learn 中，可以使用`alpha`参数设置<math display="inline"><mi>α</mi></math>。

scikit-learn 包含一个`RidgeCV`方法，允许我们选择理想的<math display="inline"><mi>α</mi></math>值：

```py
# Load library
from sklearn.linear_model import RidgeCV

# Create ridge regression with three alpha values
regr_cv = RidgeCV(alphas=[0.1, 1.0, 10.0])

# Fit the linear regression
model_cv = regr_cv.fit(features_standardized, target)

# View coefficients
model_cv.coef_
```

```py
array([1.29223201e-02, 4.40972291e+01, 5.38979372e+01])
```

我们可以轻松查看最佳模型的<math display="inline"><mi>α</mi></math>值：

```py
# View alpha
model_cv.alpha_
```

```py
0.1
```

最后一点：因为在线性回归中系数的值部分由特征的尺度确定，在正则化模型中所有系数都被合并在一起，因此在训练之前必须确保对特征进行标准化。

# 13.5 使用 Lasso 回归减少特征

## 问题

您希望通过减少特征来简化您的线性回归模型。

## 解决方案

使用 lasso 回归：

```py
# Load library
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression

# Generate features matrix, target vector
features, target = make_regression(n_samples = 100,
                                   n_features = 3,
                                   n_informative = 2,
                                   n_targets = 1,
                                   noise = 0.2,
                                   coef = False,
                                   random_state = 1)

# Standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Create lasso regression with alpha value
regression = Lasso(alpha=0.5)

# Fit the linear regression
model = regression.fit(features_standardized, target)
```

## 讨论

lasso 回归惩罚的一个有趣特征是它可以将模型的系数收缩到零，有效减少模型中的特征数。例如，在我们的解决方案中，我们将`alpha`设置为`0.5`，我们可以看到许多系数为 0，意味着它们对应的特征未在模型中使用：

```py
# View coefficients
model.coef_
```

```py
array([-0\.        , 43.58618393, 53.39523724])
```

然而，如果我们将<math display="inline"><mi>α</mi></math>增加到一个更高的值，我们会看到几乎没有特征被使用：

```py
# Create lasso regression with a high alpha
regression_a10 = Lasso(alpha=10)
model_a10 = regression_a10.fit(features_standardized, target)
model_a10.coef_
```

```py
array([-0\.        , 32.92181899, 42.73086731])
```

这种效果的实际好处在于，我们可以在特征矩阵中包含 100 个特征，然后通过调整 lasso 的 α 超参数，生成仅使用最重要的 10 个特征之一的模型（例如）。这使得我们能够在提升模型的可解释性的同时减少方差（因为更少的特征更容易解释）。
