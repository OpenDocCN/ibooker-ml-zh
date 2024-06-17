# 第十八章 朴素贝叶斯

# 18.0 引言

*贝叶斯定理*是理解某些事件概率的首选方法，如在给定一些新信息 <math display="inline"><mi>P</mi> <mrow><mo>(</mo> <mi>A</mi> <mo>∣</mo> <mi>B</mi> <mo>)</mo></mrow></math> 和对事件概率的先验信念 <math display="inline"><mi>P</mi> <mrow><mo>(</mo> <mi>A</mi> <mo>)</mo></mrow></math> 的情况下，事件 <math display="inline"><mi>P</mi> <mrow><mo>(</mo> <mi>B</mi> <mo>∣</mo> <mi>A</mi> <mo>)</mo></mrow></math> 的概率。

<math display="block"><mrow><mi>P</mi> <mrow><mo>(</mo> <mi>A</mi> <mo>∣</mo> <mi>B</mi> <mo>)</mo></mrow> <mo>=</mo> <mfrac><mrow><mi>P</mi><mo>(</mo><mi>B</mi><mo>∣</mo><mi>A</mi><mo>)</mo><mi>P</mi><mo>(</mo><mi>A</mi><mo>)</mo></mrow> <mrow><mi>P</mi><mo>(</mo><mi>B</mi><mo>)</mo></mrow></mfrac></mrow></math>

贝叶斯方法在过去十年中的流行度急剧上升，越来越多地在学术界、政府和企业中与传统的频率学应用竞争。在机器学习中，贝叶斯定理在分类问题上的一种应用是*朴素贝叶斯分类器*。朴素贝叶斯分类器将多种实用的机器学习优点结合到一个单一的分类器中。这些优点包括：

+   一种直观的方法

+   能够处理少量数据

+   训练和预测的低计算成本

+   在各种设置中通常能够产生可靠的结果

具体来说，朴素贝叶斯分类器基于：

<math display="block"><mrow><mi>P</mi> <mrow><mo>(</mo> <mi>y</mi> <mo>∣</mo> <msub><mi>x</mi> <mn>1</mn></msub> <mo>,</mo> <mo>…</mo> <mo>,</mo> <msub><mi>x</mi> <mi>j</mi></msub> <mo>)</mo></mrow> <mo>=</mo> <mfrac><mrow><mi>P</mi><mrow><mo>(</mo><msub><mi>x</mi> <mn>1</mn></msub> <mo>,</mo><mo>…</mo><mo>,</mo><msub><mi>x</mi> <mi>j</mi></msub> <mo>∣</mo><mi>y</mi><mo>)</mo></mrow><mi>P</mi><mrow><mo>(</mo><mi>y</mi><mo>)</mo></mrow></mrow> <mrow><mi>P</mi><mo>(</mo><msub><mi>x</mi> <mn>1</mn></msub> <mo>,</mo><mo>…</mo><mo>,</mo><msub><mi>x</mi> <mi>j</mi></msub> <mo>)</mo></mrow></mfrac></mrow></math>

其中：

+   <math display="inline"><mi>P</mi> <mrow><mo>(</mo> <mi>y</mi> <mo>∣</mo> <msub><mi>x</mi> <mn>1</mn></msub> <mo>,</mo> <mo>…</mo> <mo>,</mo> <msub><mi>x</mi> <mi>j</mi></msub> <mo>)</mo></mrow></math> 被称为*后验概率*，表示观察值为 <math display="inline"><msub><mi>x</mi> <mn>1</mn></msub> <mo>,</mo> <mo>…</mo> <mo>,</mo> <msub><mi>x</mi> <mi>j</mi></msub></math> 特征时类别 <math display="inline"><mi>y</mi></math> 的概率。

+   <math display="inline"><mi>P</mi><mrow><mo>(</mo><msub><mi>x</mi> <mn>1</mn></msub> <mo>,</mo><mo>…</mo><mo>,</mo><msub><mi>x</mi> <mi>j</mi></msub> <mo>∣</mo><mi>y</mi><mo>)</mo></mrow></math> 被称为*似然*，表示在给定类别 <math display="inline"><mi>y</mi></math> 时，特征 <math display="inline"><msub><mi>x</mi> <mn>1</mn></msub> <mo>,</mo> <mo>…</mo> <mo>,</mo> <msub><mi>x</mi> <mi>j</mi></msub></math> 的观察值的可能性。

+   <math display="inline"><mi>P</mi><mrow><mo>(</mo><mi>y</mi><mo>)</mo></mrow></math> 被称为*先验概率*，表示在观察数据之前，类别 <math display="inline"><mi>y</mi></math> 的概率信念。

+   <math display="inline"><mi>P</mi><mo>(</mo><msub><mi>x</mi> <mn>1</mn></msub> <mo>,</mo><mo>…</mo><mo>,</mo><msub><mi>x</mi> <mi>j</mi></msub> <mo>)</mo></math> 被称为*边缘概率*。

在朴素贝叶斯中，我们比较每个可能类别的观测后验概率值。具体来说，因为边际概率在这些比较中是恒定的，我们比较每个类别后验的分子部分。对于每个观测，具有最大后验分子的类别成为预测类别，<math display="inline"><mover accent="true"><mi>y</mi><mo>^</mo></mover></math>。

有两个关于朴素贝叶斯分类器需要注意的重要事项。首先，对于数据中的每个特征，我们必须假设似然的统计分布，<math display="inline"><mrow><mi>P</mi><mo>(</mo><msub><mi>x</mi> <mi>j</mi></msub> <mo>∣</mo><mi>y</mi><mo>)</mo></mrow></math>。常见的分布包括正态（高斯）、多项式和伯努利分布。选择的分布通常由特征的性质（连续、二进制等）决定。其次，朴素贝叶斯之所以得名，是因为我们假设每个特征及其结果的似然是独立的。这种“朴素”的假设在实践中往往是错误的，但并不会阻止构建高质量的分类器。

在本章中，我们将介绍使用 scikit-learn 训练三种类型的朴素贝叶斯分类器，使用三种不同的似然分布。此后，我们将学习如何校准朴素贝叶斯模型的预测，使其可解释。

# 18.1 训练连续特征的分类器

## 问题

您只有连续特征，并且希望训练朴素贝叶斯分类器。

## 解决方案

在 scikit-learn 中使用高斯朴素贝叶斯分类器：

```py
# Load libraries
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Create Gaussian naive Bayes object
classifer = GaussianNB()

# Train model
model = classifer.fit(features, target)
```

## 讨论

最常见的朴素贝叶斯分类器类型是*高斯朴素贝叶斯*。在高斯朴素贝叶斯中，我们假设给定观测的特征值的似然，<math display="inline"><mi>x</mi></math>，属于类别<math display="inline"><mi>y</mi></math>，遵循正态分布：

<math display="block"><mrow><mi>p</mi> <mrow><mo>(</mo> <msub><mi>x</mi> <mi>j</mi></msub> <mo>∣</mo> <mi>y</mi> <mo>)</mo></mrow> <mo>=</mo> <mfrac><mn>1</mn> <msqrt><mrow><mn>2</mn><mi>π</mi><msup><msub><mi>σ</mi> <mi>y</mi></msub> <mn>2</mn></msup></mrow></msqrt></mfrac> <msup><mi>e</mi> <mrow><mo>-</mo><mfrac><msup><mrow><mo>(</mo><msub><mi>x</mi> <mi>j</mi></msub> <mo>-</mo><msub><mi>μ</mi> <mi>y</mi></msub> <mo>)</mo></mrow> <mn>2</mn></msup> <mrow><mn>2</mn><msup><msub><mi>σ</mi> <mi>y</mi></msub> <mn>2</mn></msup></mrow></mfrac></mrow></msup></mrow></math>

其中<math display="inline"><msup><msub><mi>σ</mi> <mi>y</mi></msub> <mn>2</mn></msup></math>和<math display="inline"><msub><mi>μ</mi> <mi>y</mi></msub></math>分别是特征<math display="inline"><msub><mi>x</mi> <mi>j</mi></msub></math>对类别<math display="inline"><mi>y</mi></math>的方差和均值。由于正态分布的假设，高斯朴素贝叶斯最适合于所有特征均为连续的情况。

在 scikit-learn 中，我们像训练其他模型一样训练高斯朴素贝叶斯，使用`fit`，然后可以对观测的类别进行预测：

```py
# Create new observation
new_observation = [[ 4,  4,  4,  0.4]]

# Predict class
model.predict(new_observation)
```

```py
array([1])
```

朴素贝叶斯分类器的一个有趣方面之一是，它们允许我们对目标类别分配先验信念。我们可以使用`GaussianNB priors`参数来实现这一点，该参数接受目标向量每个类别的概率列表：

```py
# Create Gaussian naive Bayes object with prior probabilities of each class
clf = GaussianNB(priors=[0.25, 0.25, 0.5])

# Train model
model = classifer.fit(features, target)
```

如果我们不向`priors`参数添加任何参数，则根据数据调整先验。

最后，请注意，从高斯朴素贝叶斯获得的原始预测概率（使用`predict_proba`输出）未经校准。也就是说，它们不应被信任。如果我们想要创建有用的预测概率，我们需要使用等渗回归或相关方法进行校准。

## 另请参阅

+   [机器学习中朴素贝叶斯分类器的工作原理](https://oreil.ly/9yqSw)

# 18.2 训练离散和计数特征的分类器

## 问题

给定离散或计数数据，您需要训练一个朴素贝叶斯分类器。

## 解决方案

使用多项式朴素贝叶斯分类器：

```py
# Load libraries
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Create text
text_data = np.array(['I love Brazil. Brazil!',
                      'Brazil is best',
                      'Germany beats both'])

# Create bag of words
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)

# Create feature matrix
features = bag_of_words.toarray()

# Create target vector
target = np.array([0,0,1])

# Create multinomial naive Bayes object with prior probabilities of each class
classifer = MultinomialNB(class_prior=[0.25, 0.5])

# Train model
model = classifer.fit(features, target)
```

## 讨论

*多项式朴素贝叶斯*的工作方式与高斯朴素贝叶斯类似，但特征被假定为多项式分布。实际上，这意味着当我们有离散数据时（例如，电影评分从 1 到 5），这种分类器通常被使用。多项式朴素贝叶斯最常见的用途之一是使用词袋或<math display="inline"><mtext fontstyle="italic">tf-idf</mtext></math>方法进行文本分类（参见 Recipes 6.9 和 6.10)。

在我们的解决方案中，我们创建了一个包含三个观察结果的玩具文本数据集，并将文本字符串转换为词袋特征矩阵和相应的目标向量。然后，我们使用`MultinomialNB`来训练一个模型，同时为两个类别（支持巴西和支持德国）定义了先验概率。

`MultinomialNB`的工作方式类似于`GaussianNB`；模型使用`fit`进行训练，并且可以使用`predict`进行预测：

```py
# Create new observation
new_observation = [[0, 0, 0, 1, 0, 1, 0]]

# Predict new observation's class
model.predict(new_observation)
```

```py
array([0])
```

如果未指定`class_prior`，则使用数据学习先验概率。但是，如果我们想要使用均匀分布作为先验，可以设置`fit_prior=False`。

最后，`MultinomialNB`包含一个添加平滑的超参数`alpha`，应该进行调节。默认值为`1.0`，`0.0`表示不进行平滑。

# 18.3 训练二元特征的朴素贝叶斯分类器

## 问题

您有二元特征数据，并需要训练一个朴素贝叶斯分类器。

## 解决方案

使用伯努利朴素贝叶斯分类器：

```py
# Load libraries
import numpy as np
from sklearn.naive_bayes import BernoulliNB

# Create three binary features
features = np.random.randint(2, size=(100, 3))

# Create a binary target vector
target = np.random.randint(2, size=(100, 1)).ravel()

# Create Bernoulli naive Bayes object with prior probabilities of each class
classifer = BernoulliNB(class_prior=[0.25, 0.5])

# Train model
model = classifer.fit(features, target)
```

## 讨论

*伯努利朴素贝叶斯*分类器假设所有特征都是二元的，即它们只能取两个值（例如，已经进行了独热编码的名义分类特征）。与其多项式兄弟一样，伯努利朴素贝叶斯在文本分类中经常被使用，当我们的特征矩阵仅是文档中单词的存在或不存在时。此外，像`MultinomialNB`一样，`BernoulliNB`也有一个添加平滑的超参数`alpha`，我们可以使用模型选择技术来调节。最后，如果我们想使用先验概率，可以使用`class_prior`参数并将其设置为包含每个类的先验概率的列表。如果我们想指定均匀先验，可以设置`fit_prior=False`：

```py
model_uniform_prior = BernoulliNB(class_prior=None, fit_prior=False)
```

# 18.4 校准预测概率

## 问题

您希望校准朴素贝叶斯分类器的预测概率，以便能够解释它们。

## 解决方案

使用 `CalibratedClassifierCV`：

```py
# Load libraries
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Create Gaussian naive Bayes object
classifer = GaussianNB()

# Create calibrated cross-validation with sigmoid calibration
classifer_sigmoid = CalibratedClassifierCV(classifer, cv=2, method='sigmoid')

# Calibrate probabilities
classifer_sigmoid.fit(features, target)

# Create new observation
new_observation = [[ 2.6,  2.6,  2.6,  0.4]]

# View calibrated probabilities
classifer_sigmoid.predict_proba(new_observation)
```

```py
array([[0.31859969, 0.63663466, 0.04476565]])
```

## 讨论

类概率是机器学习模型中常见且有用的一部分。在 scikit-learn 中，大多数学习算法允许我们使用 `predict_proba` 来查看类成员的预测概率。例如，如果我们只想在模型预测某个类的概率超过 90%时预测该类，这将非常有用。然而，一些模型，包括朴素贝叶斯分类器，输出的概率不是基于现实世界的。也就是说，`predict_proba` 可能会预测一个观测属于某一类的概率是 0.70，而实际上可能是 0.10 或 0.99。具体来说，在朴素贝叶斯中，虽然对不同目标类的预测概率排序是有效的，但原始预测概率往往会取极端值，接近 0 或 1。

要获得有意义的预测概率，我们需要进行所谓的*校准*。在 scikit-learn 中，我们可以使用 `CalibratedClassifierCV` 类通过 k 折交叉验证创建良好校准的预测概率。在 `CalibratedClassifierCV` 中，训练集用于训练模型，测试集用于校准预测概率。返回的预测概率是 k 折交叉验证的平均值。

使用我们的解决方案，我们可以看到原始和良好校准的预测概率之间的差异。在我们的解决方案中，我们创建了一个高斯朴素贝叶斯分类器。如果我们训练该分类器，然后预测新观测的类概率，我们可以看到非常极端的概率估计：

```py
# Train a Gaussian naive Bayes then predict class probabilities
classifer.fit(features, target).predict_proba(new_observation)
```

```py
array([[2.31548432e-04, 9.99768128e-01, 3.23532277e-07]])
```

然而，如果在我们校准预测的概率之后（我们在我们的解决方案中完成了这一步），我们得到非常不同的结果：

```py
# View calibrated probabilities
array([[0.31859969, 0.63663466, 0.04476565]])
```

```py
array([[ 0.31859969,  0.63663466,  0.04476565]])
```

`CalibratedClassifierCV` 提供两种校准方法——Platt 的 sigmoid 模型和等温回归——由 `method` 参数定义。虽然我们没有空间详细讨论，但由于等温回归是非参数的，当样本量非常小时（例如 100 个观测），它往往会过拟合。在我们的解决方案中，我们使用了包含 150 个观测的鸢尾花数据集，因此使用了 Platt 的 sigmoid 模型。
