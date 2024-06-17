# 第五章：处理分类数据

# 5.0 介绍

通常有用的是，我们不仅仅用数量来衡量物体，而是用某种质量来衡量。我们经常用类别如性别、颜色或汽车品牌来表示定性信息。然而，并非所有分类数据都相同。没有内在排序的类别集称为*名义*。名义类别的例子包括：

+   蓝色，红色，绿色

+   男，女

+   香蕉，草莓，苹果

相比之下，当一组类别具有一些自然顺序时，我们称之为*序数*。例如：

+   低，中，高

+   年轻，年老

+   同意，中立，不同意

此外，分类信息通常以向量或字符串列（例如`"Maine"`、`"Texas"`、`"Delaware"`）的形式表示在数据中。问题在于，大多数机器学习算法要求输入为数值。

k 最近邻算法是需要数值数据的一个例子。算法中的一步是计算观测之间的距离，通常使用欧氏距离：

<math display="block"><msqrt><mrow><msubsup><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>n</mi></msubsup> <msup><mrow><mo>(</mo><msub><mi>x</mi> <mi>i</mi></msub> <mo>-</mo><msub><mi>y</mi> <mi>i</mi></msub> <mo>)</mo></mrow> <mn>2</mn></msup></mrow></msqrt></math>

其中<math display="inline"><mi>x</mi></math>和<math display="inline"><mi>y</mi></math>是两个观测值，下标<math display="inline"><mi>i</mi></math>表示观测的第<math display="inline"><mi>i</mi></math>个特征的值。然而，如果<math display="inline"><msub><mi>x</mi><mi>i</mi></msub></math>的值是一个字符串（例如`"Texas"`），显然是无法进行距离计算的。我们需要将字符串转换为某种数值格式，以便可以将其输入到欧氏距离方程中。我们的目标是以一种能够正确捕捉类别信息（序数性，类别之间的相对间隔等）的方式转换数据。在本章中，我们将涵盖使这种转换以及克服处理分类数据时经常遇到的其他挑战的技术。

# 5.1 编码名义分类特征

## 问题

您有一个没有内在排序的名义类别特征（例如苹果，梨，香蕉），并且希望将该特征编码为数值。

## 解决方案

使用 scikit-learn 的`LabelBinarizer`对特征进行独热编码：

```py
# Import libraries
import numpy as np
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

# Create feature
feature = np.array([["Texas"],
                    ["California"],
                    ["Texas"],
                    ["Delaware"],
                    ["Texas"]])

# Create one-hot encoder
one_hot = LabelBinarizer()

# One-hot encode feature
one_hot.fit_transform(feature)
```

```py
array([[0, 0, 1],
       [1, 0, 0],
       [0, 0, 1],
       [0, 1, 0],
       [0, 0, 1]])
```

我们可以使用`classes_`属性来输出类别：

```py
# View feature classes
one_hot.classes_
```

```py
array(['California', 'Delaware', 'Texas'],
      dtype='<U10')
```

如果我们想要反向进行独热编码，我们可以使用`inverse_transform`：

```py
# Reverse one-hot encoding
one_hot.inverse_transform(one_hot.transform(feature))
```

```py
array(['Texas', 'California', 'Texas', 'Delaware', 'Texas'],
      dtype='<U10')
```

我们甚至可以使用 pandas 来进行独热编码：

```py
# Import library
import pandas as pd

# Create dummy variables from feature
pd.get_dummies(feature[:,0])
```

|  | 加利福尼亚 | 特拉华州 | 德克萨斯州 |
| --- | --- | --- | --- |
| 0 | 0 | 0 | 1 |
| 1 | 1 | 0 | 0 |
| 2 | 0 | 0 | 1 |
| 3 | 0 | 1 | 0 |
| 4 | 0 | 0 | 1 |

scikit-learn 的一个有用特性是能够处理每个观测列表包含多个类别的情况：

```py
# Create multiclass feature
multiclass_feature = [("Texas", "Florida"),
                      ("California", "Alabama"),
                      ("Texas", "Florida"),
                      ("Delaware", "Florida"),
                      ("Texas", "Alabama")]

# Create multiclass one-hot encoder
one_hot_multiclass = MultiLabelBinarizer()

# One-hot encode multiclass feature
one_hot_multiclass.fit_transform(multiclass_feature)
```

```py
array([[0, 0, 0, 1, 1],
       [1, 1, 0, 0, 0],
       [0, 0, 0, 1, 1],
       [0, 0, 1, 1, 0],
       [1, 0, 0, 0, 1]])
```

再次，我们可以使用`classes_`方法查看类别：

```py
# View classes
one_hot_multiclass.classes_
```

```py
array(['Alabama', 'California', 'Delaware', 'Florida', 'Texas'], dtype=object)
```

## 讨论

我们可能认为正确的策略是为每个类分配一个数值（例如，Texas = 1，California = 2）。然而，当我们的类没有内在的顺序（例如，Texas 不是比 California “更少”），我们的数值值误创建了一个不存在的排序。

适当的策略是为原始特征的每个类创建一个二进制特征。在机器学习文献中通常称为 *独热编码*，而在统计和研究文献中称为 *虚拟化*。我们解决方案的特征是一个包含三个类（即 Texas、California 和 Delaware）的向量。在独热编码中，每个类都成为其自己的特征，当类出现时为 1，否则为 0。因为我们的特征有三个类，独热编码返回了三个二进制特征（每个类一个）。通过使用独热编码，我们可以捕捉观察值在类中的成员身份，同时保持类缺乏任何层次结构的概念。

最后，经常建议在对一个特征进行独热编码后，删除结果矩阵中的一个独热编码特征，以避免线性相关性。

## 参见

+   [回归模型中的虚拟变量陷阱，Algosome](https://oreil.ly/xjBhG)

+   [使用独热编码时删除其中一列，Cross Validated](https://oreil.ly/CTdpG)

# 5.2 编码序数分类特征

## 问题

您有一个序数分类特征（例如高、中、低），并且希望将其转换为数值。

## 解决方案

使用 pandas DataFrame 的 `replace` 方法将字符串标签转换为数值等价物：

```py
# Load library
import pandas as pd

# Create features
dataframe = pd.DataFrame({"Score": ["Low", "Low", "Medium", "Medium", "High"]})

# Create mapper
scale_mapper = {"Low":1,
                "Medium":2,
                "High":3}

# Replace feature values with scale
dataframe["Score"].replace(scale_mapper)
```

```py
0    1
1    1
2    2
3    2
4    3
Name: Score, dtype: int64
```

## 讨论

经常情况下，我们有一个具有某种自然顺序的类的特征。一个著名的例子是 Likert 量表：

+   强烈同意

+   同意

+   中立

+   不同意

+   强烈不同意

在将特征编码用于机器学习时，我们需要将序数类转换为保持排序概念的数值。最常见的方法是创建一个将类的字符串标签映射到数字的字典，然后将该映射应用于特征。

根据我们对序数类的先前信息，选择数值值是很重要的。在我们的解决方案中，`high` 比 `low` 大三倍。在许多情况下这是可以接受的，但如果假设的类之间间隔不均等，这种方法可能失效：

```py
dataframe = pd.DataFrame({"Score": ["Low",
                                    "Low",
                                    "Medium",
                                    "Medium",
                                    "High",
                                    "Barely More Than Medium"]})

scale_mapper = {"Low":1,
                "Medium":2,
                "Barely More Than Medium":3,
                "High":4}

dataframe["Score"].replace(scale_mapper)
```

```py
0    1
1    1
2    2
3    2
4    4
5    3
Name: Score, dtype: int64
```

在此示例中，`Low` 和 `Medium` 之间的距离与 `Medium` 和 `Barely More Than Medium` 之间的距离相同，这几乎肯定不准确。最佳方法是在映射到类的数值值时要注意：

```py
scale_mapper = {"Low":1,
                "Medium":2,
                "Barely More Than Medium":2.1,
                "High":3}

dataframe["Score"].replace(scale_mapper)
```

```py
0    1.0
1    1.0
2    2.0
3    2.0
4    3.0
5    2.1
Name: Score, dtype: float64
```

# 5.3 编码特征字典

## 问题

您有一个字典，并希望将其转换为特征矩阵。

## 解决方案

使用 `DictVectorizer`：

```py
# Import library
from sklearn.feature_extraction import DictVectorizer

# Create dictionary
data_dict = [{"Red": 2, "Blue": 4},
             {"Red": 4, "Blue": 3},
             {"Red": 1, "Yellow": 2},
             {"Red": 2, "Yellow": 2}]

# Create dictionary vectorizer
dictvectorizer = DictVectorizer(sparse=False)

# Convert dictionary to feature matrix
features = dictvectorizer.fit_transform(data_dict)

# View feature matrix
features
```

```py
array([[ 4.,  2.,  0.],
       [ 3.,  4.,  0.],
       [ 0.,  1.,  2.],
       [ 0.,  2.,  2.]])
```

默认情况下，`DictVectorizer`输出一个仅存储值非 0 的稀疏矩阵。当我们遇到大规模矩阵（通常在自然语言处理中）并希望最小化内存需求时，这非常有帮助。我们可以使用`sparse=False`来强制`DictVectorizer`输出一个密集矩阵。

我们可以使用`get_feature_names`方法获取每个生成特征的名称：

```py
# Get feature names
feature_names = dictvectorizer.get_feature_names()

# View feature names
feature_names
```

```py
['Blue', 'Red', 'Yellow']
```

虽然不必要，为了说明我们可以创建一个 pandas DataFrame 来更好地查看输出：

```py
# Import library
import pandas as pd

# Create dataframe from features
pd.DataFrame(features, columns=feature_names)
```

|  | 蓝色 | 红色 | 黄色 |
| --- | --- | --- | --- |
| 0 | 4.0 | 2.0 | 0.0 |
| 1 | 3.0 | 4.0 | 0.0 |
| 2 | 0.0 | 1.0 | 2.0 |
| 3 | 0.0 | 2.0 | 2.0 |

## 讨论

字典是许多编程语言中常用的数据结构；然而，机器学习算法期望数据以矩阵的形式存在。我们可以使用 scikit-learn 的`DictVectorizer`来实现这一点。

这是自然语言处理时常见的情况。例如，我们可能有一系列文档，每个文档都有一个字典，其中包含每个单词在文档中出现的次数。使用`DictVectorizer`，我们可以轻松创建一个特征矩阵，其中每个特征是每个文档中单词出现的次数：

```py
# Create word count dictionaries for four documents
doc_1_word_count = {"Red": 2, "Blue": 4}
doc_2_word_count = {"Red": 4, "Blue": 3}
doc_3_word_count = {"Red": 1, "Yellow": 2}
doc_4_word_count = {"Red": 2, "Yellow": 2}

# Create list
doc_word_counts = [doc_1_word_count,
                   doc_2_word_count,
                   doc_3_word_count,
                   doc_4_word_count]

# Convert list of word count dictionaries into feature matrix
dictvectorizer.fit_transform(doc_word_counts)
```

```py
array([[ 4.,  2.,  0.],
       [ 3.,  4.,  0.],
       [ 0.,  1.,  2.],
       [ 0.,  2.,  2.]])
```

在我们的示例中，只有三个唯一的单词（`红色`，`黄色`，`蓝色`），所以我们的矩阵中只有三个特征；然而，如果每个文档实际上是大学图书馆中的一本书，我们的特征矩阵将非常庞大（然后我们将希望将`sparse`设置为`True`）。

## 参见

+   [如何在 Python 中创建字典](https://oreil.ly/zu5hU)

+   [SciPy 稀疏矩阵](https://oreil.ly/5nAsU)

# 5.4 填充缺失的类值

## 问题

您有一个包含缺失值的分类特征，您希望用预测值替换它。

## 解决方案

理想的解决方案是训练一个机器学习分类器算法来预测缺失值，通常是 k 近邻（KNN）分类器：

```py
# Load libraries
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Create feature matrix with categorical feature
X = np.array([[0, 2.10, 1.45],
              [1, 1.18, 1.33],
              [0, 1.22, 1.27],
              [1, -0.21, -1.19]])

# Create feature matrix with missing values in the categorical feature
X_with_nan = np.array([[np.nan, 0.87, 1.31],
                       [np.nan, -0.67, -0.22]])

# Train KNN learner
clf = KNeighborsClassifier(3, weights='distance')
trained_model = clf.fit(X[:,1:], X[:,0])

# Predict class of missing values
imputed_values = trained_model.predict(X_with_nan[:,1:])

# Join column of predicted class with their other features
X_with_imputed = np.hstack((imputed_values.reshape(-1,1), X_with_nan[:,1:]))

# Join two feature matrices
np.vstack((X_with_imputed, X))
```

```py
array([[ 0\.  ,  0.87,  1.31],
       [ 1\.  , -0.67, -0.22],
       [ 0\.  ,  2.1 ,  1.45],
       [ 1\.  ,  1.18,  1.33],
       [ 0\.  ,  1.22,  1.27],
       [ 1\.  , -0.21, -1.19]])
```

另一种解决方案是使用特征的最频繁值填充缺失值：

```py
from sklearn.impute import SimpleImputer

# Join the two feature matrices
X_complete = np.vstack((X_with_nan, X))

imputer = SimpleImputer(strategy='most_frequent')

imputer.fit_transform(X_complete)
```

```py
array([[ 0\.  ,  0.87,  1.31],
       [ 0\.  , -0.67, -0.22],
       [ 0\.  ,  2.1 ,  1.45],
       [ 1\.  ,  1.18,  1.33],
       [ 0\.  ,  1.22,  1.27],
       [ 1\.  , -0.21, -1.19]])
```

## 讨论

当分类特征中存在缺失值时，我们最好的解决方案是打开我们的机器学习算法工具箱，预测缺失观测值的值。我们可以通过将具有缺失值的特征视为目标向量，其他特征视为特征矩阵来实现此目标。常用的算法之一是 KNN（在第十五章中详细讨论），它将缺失值分配给*k*个最近观测中最频繁出现的类别。

或者，我们可以使用特征的最频繁类别填充缺失值，甚至丢弃具有缺失值的观测。虽然不如 KNN 复杂，但这些选项在处理大数据时更具可扩展性。无论哪种情况，都建议包含一个二元特征，指示哪些观测包含了填充值。

## 参见

+   [scikit-learn 文档：缺失值的插补](https://oreil.ly/joZ6J)

+   [在随机森林分类器中克服缺失值](https://oreil.ly/TcvOf)

+   [K 最近邻方法作为插补方法的研究](https://oreil.ly/kDFEC)

# 5.5 处理不平衡类别

## 问题

如果您有一个具有高度不平衡类别的目标向量，并且希望进行调整以处理类别不平衡。

## 解决方案

收集更多数据。如果不可能，请更改用于评估模型的指标。如果这样做不起作用，请考虑使用模型的内置类权重参数（如果可用），下采样或上采样。我们将在后面的章节中介绍评估指标，因此现在让我们专注于类权重参数、下采样和上采样。

为了演示我们的解决方案，我们需要创建一些具有不平衡类别的数据。Fisher 的鸢尾花数据集包含三个平衡类别的 50 个观察，每个类别表示花的物种（*Iris setosa*、*Iris virginica* 和 *Iris versicolor*）。为了使数据集不平衡，我们移除了 50 个 *Iris setosa* 观察中的 40 个，并合并了 *Iris virginica* 和 *Iris versicolor* 类别。最终结果是一个二元目标向量，指示观察是否为 *Iris setosa* 花。结果是 10 个 *Iris setosa*（类别 0）的观察和 100 个非 *Iris setosa*（类别 1）的观察：

```py
# Load libraries
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load iris data
iris = load_iris()

# Create feature matrix
features = iris.data

# Create target vector
target = iris.target

# Remove first 40 observations
features = features[40:,:]
target = target[40:]

# Create binary target vector indicating if class 0
target = np.where((target == 0), 0, 1)

# Look at the imbalanced target vector
target
```

```py
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
```

scikit-learn 中的许多算法在训练期间提供一个参数来加权类别，以抵消其不平衡的效果。虽然我们尚未涵盖它，`RandomForestClassifier` 是一种流行的分类算法，并包含一个 `class_weight` 参数；在 14.4 节 中了解更多关于 `RandomForestClassifier` 的信息。您可以传递一个参数显式指定所需的类权重：

```py
# Create weights
weights = {0: 0.9, 1: 0.1}

# Create random forest classifier with weights
RandomForestClassifier(class_weight=weights)
```

```py
RandomForestClassifier(class_weight={0: 0.9, 1: 0.1})
```

或者您可以传递 `balanced`，它会自动创建与类别频率成反比的权重：

```py
# Train a random forest with balanced class weights
RandomForestClassifier(class_weight="balanced")
```

```py
RandomForestClassifier(class_weight='balanced')
```

或者，我们可以对多数类进行下采样或者对少数类进行上采样。在 *下采样* 中，我们从多数类中无放回随机抽样（即观察次数较多的类别）以创建一个新的观察子集，其大小等于少数类。例如，如果少数类有 10 个观察，我们将从多数类中随机选择 10 个观察，然后使用这 20 个观察作为我们的数据。在这里，我们正是利用我们不平衡的鸢尾花数据做到这一点：

```py
# Indicies of each class's observations
i_class0 = np.where(target == 0)[0]
i_class1 = np.where(target == 1)[0]

# Number of observations in each class
n_class0 = len(i_class0)
n_class1 = len(i_class1)

# For every observation of class 0, randomly sample
# from class 1 without replacement
i_class1_downsampled = np.random.choice(i_class1, size=n_class0, replace=False)

# Join together class 0's target vector with the
# downsampled class 1's target vector
np.hstack((target[i_class0], target[i_class1_downsampled]))
```

```py
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
```

```py
# Join together class 0's feature matrix with the
# downsampled class 1's feature matrix
np.vstack((features[i_class0,:], features[i_class1_downsampled,:]))[0:5]
```

```py
array([[ 5\. ,  3.5,  1.3,  0.3],
       [ 4.5,  2.3,  1.3,  0.3],
       [ 4.4,  3.2,  1.3,  0.2],
       [ 5\. ,  3.5,  1.6,  0.6],
       [ 5.1,  3.8,  1.9,  0.4]])
```

我们的另一种选择是对少数类进行上采样。在 *上采样* 中，对于多数类中的每个观察，我们从少数类中随机选择一个观察，可以重复选择。结果是来自少数和多数类的相同数量的观察。上采样的实现非常类似于下采样，只是反向操作：

```py
# For every observation in class 1, randomly sample from class 0 with
# replacement
i_class0_upsampled = np.random.choice(i_class0, size=n_class1, replace=True)

# Join together class 0's upsampled target vector with class 1's target vector
np.concatenate((target[i_class0_upsampled], target[i_class1]))
```

```py
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
```

```py
# Join together class 0's upsampled feature matrix with class 1's feature matrix
np.vstack((features[i_class0_upsampled,:], features[i_class1,:]))[0:5]
```

```py
array([[ 5\. ,  3.5,  1.6,  0.6],
       [ 5\. ,  3.5,  1.6,  0.6],
       [ 5\. ,  3.3,  1.4,  0.2],
       [ 4.5,  2.3,  1.3,  0.3],
       [ 4.8,  3\. ,  1.4,  0.3]])
```

## 讨论

在现实世界中，不平衡的类别随处可见—大多数访问者不会点击购买按钮，而许多类型的癌症又是相当罕见的。因此，在机器学习中处理不平衡的类别是一项常见的活动。

我们最好的策略就是简单地收集更多的观察数据—尤其是来自少数类的观察数据。然而，通常情况下这并不可能，所以我们必须求助于其他选择。

第二种策略是使用更适合于不平衡类别的模型评估指标。准确率通常被用作评估模型性能的指标，但在存在不平衡类别的情况下，准确率可能并不合适。例如，如果只有 0.5%的观察数据属于某种罕见的癌症，那么即使是一个简单的模型预测没有人有癌症，准确率也会达到 99.5%。显然，这并不理想。我们将在后面的章节中讨论一些更好的指标，如混淆矩阵、精确度、召回率、*F[1]*分数和 ROC 曲线。

第三种策略是使用一些模型实现中包含的类别加权参数。这使得算法能够调整不平衡的类别。幸运的是，许多 scikit-learn 分类器都有一个`class_weight`参数，这使得它成为一个不错的选择。

第四和第五种策略是相关的：下采样和上采样。在下采样中，我们创建一个与少数类相同大小的多数类的随机子集。在上采样中，我们从少数类中重复有放回地抽样，使其大小与多数类相等。选择使用下采样还是上采样是与上下文相关的决定，通常我们应该尝试两种方法，看看哪一种效果更好。
