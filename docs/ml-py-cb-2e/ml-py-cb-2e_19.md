# 第十九章 聚类

# 19.0 介绍

在本书的大部分内容中，我们已经研究了监督机器学习——我们既可以访问特征又可以访问目标的情况。不幸的是，这并不总是事实。经常情况下，我们遇到的情况是我们只知道特征。例如，想象一下我们有一家杂货店的销售记录，并且我们想要按照购物者是否是折扣俱乐部会员来分割销售记录。使用监督学习是不可能的，因为我们没有一个目标来训练和评估我们的模型。然而，还有另一种选择：无监督学习。如果杂货店的折扣俱乐部会员和非会员的行为实际上是不同的，那么两个会员之间的平均行为差异将小于会员和非会员购物者之间的平均行为差异。换句话说，会有两个观察结果的簇。

聚类算法的目标是识别那些潜在的观察结果分组，如果做得好，即使没有目标向量，我们也能够预测观察结果的类别。有许多聚类算法，它们有各种各样的方法来识别数据中的簇。在本章中，我们将介绍一些使用 scikit-learn 的聚类算法以及如何在实践中使用它们。

# 19.1 使用 K 均值进行聚类

## 问题

你想要将观察结果分成*k*组。

## 解决方案

使用*k 均值聚类*：

```py
# Load libraries
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load data
iris = datasets.load_iris()
features = iris.data

# Standardize features
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# Create k-means object
cluster = KMeans(n_clusters=3, random_state=0, n_init="auto")

# Train model
model = cluster.fit(features_std)
```

## 讨论

k 均值聚类是最常见的聚类技术之一。在 k 均值聚类中，算法试图将观察结果分成*k*组，每组的方差大致相等。组数*k*由用户作为超参数指定。具体来说，在 k 均值聚类中：

1.  在随机位置创建*k*聚类的“中心”点。

1.  对于每个观察结果：

    1.  计算每个观察结果与*k*中心点的距离。

    1.  观察结果被分配到最近中心点的簇中。

1.  中心点被移动到各自簇的平均值（即，中心）。

1.  步骤 2 和 3 重复，直到没有观察结果在簇成员资格上发生变化。

在这一点上，算法被认为已经收敛并停止。

关于 k 均值聚类有三点需要注意。首先，k 均值聚类假设簇是凸形的（例如，圆形，球形）。其次，所有特征都是等比例缩放的。在我们的解决方案中，我们标准化了特征以满足这个假设。第三，各组是平衡的（即，观察结果的数量大致相同）。如果我们怀疑无法满足这些假设，我们可以尝试其他聚类方法。

在 scikit-learn 中，k-means 聚类是在 `KMeans` 类中实现的。最重要的参数是 `n_clusters`，它设置聚类数 *k*。在某些情况下，数据的性质将决定 *k* 的值（例如，学校学生的数据将有一个班级对应一个聚类），但通常我们不知道聚类数。在这些情况下，我们希望基于某些准则选择 *k*。例如，轮廓系数（参见第 11.9 节）可以衡量聚类内部的相似性与聚类间的相似性。此外，由于 k-means 聚类计算开销较大，我们可能希望利用计算机的所有核心。我们可以通过设置 `n_jobs=-1` 来实现这一点。

在我们的解决方案中，我们有点作弊，使用了已知包含三个类别的鸢尾花数据。因此，我们设置 *k = 3*。我们可以使用 `labels_` 查看每个观测数据的预测类别：

```py
# View predicted class
model.labels_
```

```py
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2,
       1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 1, 2, 1, 2,
       2, 1, 2, 1, 1, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2,
       2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 1], dtype=int32)
```

如果我们将其与观测数据的真实类别进行比较，可以看到，尽管类标签有所不同（即 `0`、`1` 和 `2`），k-means 的表现还是相当不错的：

```py
# View true class
iris.target
```

```py
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
```

然而，正如你所想象的那样，如果我们选择了错误的聚类数，k-means 的性能将显著甚至可能严重下降。

最后，与其他 scikit-learn 模型一样，我们可以使用训练好的聚类模型来预测新观测数据的值：

```py
# Create new observation
new_observation = [[0.8, 0.8, 0.8, 0.8]]

# Predict observation's cluster
model.predict(new_observation)
```

```py
array([2], dtype=int32)
```

预测观测数据属于距离其最近的聚类中心点。我们甚至可以使用 `cluster_centers_` 查看这些中心点：

```py
# View cluster centers
model.cluster_centers_
```

```py
array([[-1.01457897,  0.85326268, -1.30498732, -1.25489349],
       [-0.01139555, -0.87600831,  0.37707573,  0.31115341],
       [ 1.16743407,  0.14530299,  1.00302557,  1.0300019 ]])
```

## 参见

+   [介绍 K-means 聚类](https://oreil.ly/HDfUz)

# 19.2 加速 K-Means 聚类

## 问题

您希望将观测数据分组成 *k* 组，但 k-means 太耗时。

## 解决方案

使用 mini-batch k-means：

```py
# Load libraries
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

# Load data
iris = datasets.load_iris()
features = iris.data

# Standardize features
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# Create k-mean object
cluster = MiniBatchKMeans(n_clusters=3, random_state=0, batch_size=100,
       n_init="auto")

# Train model
model = cluster.fit(features_std)
```

## 讨论

*Mini-batch k-means* 类似于讨论中的 k-means 算法（见第 19.1 节）。不详细讨论的话，两者的区别在于，mini-batch k-means 中计算成本最高的步骤仅在随机抽样的观测数据上进行，而不是全部观测数据。这种方法可以显著减少算法找到收敛（即拟合数据）所需的时间，只有少量的质量损失。

`MiniBatchKMeans` 类似于 `KMeans`，但有一个显著的区别：`batch_size` 参数。`batch_size` 控制每个批次中随机选择的观测数据数量。批次大小越大，训练过程的计算成本越高。

# 19.3 使用均值漂移进行聚类

## 问题

您希望在不假设聚类数或形状的情况下对观测数据进行分组。

## 解决方案

使用均值漂移聚类：

```py
# Load libraries
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift

# Load data
iris = datasets.load_iris()
features = iris.data

# Standardize features
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# Create mean shift object
cluster = MeanShift(n_jobs=-1)

# Train model
model = cluster.fit(features_std)
```

## 讨论

我们之前讨论过 k-means 聚类的一个缺点是在训练之前需要设置聚类数 *k*，并且该方法对聚类形状作出了假设。一种无此限制的聚类算法是均值漂移。

*均值漂移*是一个简单的概念，但有些难以解释。因此，通过类比可能是最好的方法。想象一个非常雾蒙蒙的足球场（即，二维特征空间），上面站着 100 个人（即，我们的观察结果）。因为有雾，一个人只能看到很短的距离。每分钟，每个人都会四处张望，并朝着能看到最多人的方向迈出一步。随着时间的推移，人们开始团结在一起，重复向更大的人群迈步。最终的结果是围绕场地的人群聚类。人们被分配到他们最终停留的聚类中。

scikit-learn 的实际均值漂移实现，`MeanShift`，更为复杂，但遵循相同的基本逻辑。`MeanShift`有两个重要的参数我们应该注意。首先，`bandwidth`设置了观察使用的区域（即核心）的半径，以确定向何处移动。在我们的类比中，bandwidth 代表一个人透过雾能看到的距离。我们可以手动设置此参数，但默认情况下会自动估算一个合理的带宽（计算成本显著增加）。其次，在均值漂移中有时没有其他观察在观察的核心中。也就是说，在我们的足球场上，一个人看不到其他人。默认情况下，`MeanShift`将所有这些“孤儿”观察分配给最近观察的核心。但是，如果我们希望排除这些孤儿，我们可以设置`cluster_all=False`，其中孤儿观察被赋予标签`-1`。

## 参见

+   [均值漂移聚类算法，EFAVDB](https://oreil.ly/Gb3VG)

# 19.4 使用 DBSCAN 进行聚类

## 问题

您希望将观察结果分组为高密度的聚类。

## 解决方案

使用 DBSCAN 聚类：

```py
# Load libraries
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Load data
iris = datasets.load_iris()
features = iris.data

# Standardize features
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# Create DBSCAN object
cluster = DBSCAN(n_jobs=-1)

# Train model
model = cluster.fit(features_std)
```

## 讨论

*DBSCAN*的动机在于，聚类将是许多观察结果密集堆积的区域，并且不对聚类形状做出假设。具体来说，在 DBSCAN 中：

1.  选择一个随机观察结果，*x[i]*。

1.  如果*x[i]*有足够数量的近邻观察，我们认为它是聚类的一部分。

1.  步骤 2 递归地重复对*x[i]*的所有邻居，邻居的邻居等的处理。这些是聚类的核心观察结果。

1.  一旦步骤 3 耗尽附近的观察，就会选择一个新的随机点（即，在步骤 1 重新开始）。

一旦完成这一步骤，我们就得到了多个聚类的核心观察结果集。最终，任何靠近聚类但不是核心样本的观察被认为是聚类的一部分，而不靠近聚类的观察则被标记为离群值。

`DBSCAN`有三个主要的参数需要设置：

`eps`

一个观察到另一个观察的最大距离，以便将其视为邻居。

`min_samples`

小于`eps`距离的观察数目最少的观察，被认为是核心观察结果。

`metric`

`eps`使用的距离度量——例如，`minkowski`或`euclidean`（注意，如果使用 Minkowski 距离，参数`p`可以用来设置 Minkowski 度量的幂）。

如果我们查看我们的训练数据中的集群，我们可以看到已经识别出两个集群，`0`和`1`，而异常值观测被标记为`-1`：

```py
# Show cluster membership
model.labels_
```

```py
array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1,
        0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  1,
        1,  1,  1,  1,  1, -1, -1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1,
       -1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
       -1,  1, -1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1, -1,  1, -1,  1,
        1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1, -1,  1,  1, -1, -1,
       -1,  1,  1, -1,  1,  1, -1,  1,  1,  1, -1, -1, -1,  1,  1,  1, -1,
       -1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1,  1])
```

## 另请参阅

+   [DBSCAN，维基百科](https://oreil.ly/QBx3a)

# 19.5 使用分层合并进行聚类

## 问题

您想使用集群的层次结构对观测进行分组。

## 解决方案

使用聚类：

```py
# Load libraries
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Load data
iris = datasets.load_iris()
features = iris.data

# Standardize features
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# Create agglomerative clustering object
cluster = AgglomerativeClustering(n_clusters=3)

# Train model
model = cluster.fit(features_std)
```

## 讨论

*凝聚式聚类*是一种强大、灵活的分层聚类算法。在凝聚式聚类中，所有观测都开始作为自己的集群。接下来，满足一些条件的集群被合并。这个过程重复进行，直到达到某个结束点为止。在 scikit-learn 中，`AgglomerativeClustering`使用`linkage`参数来确定最小化合并策略：

+   合并集群的方差（`ward`）

+   来自成对集群的观察之间的平均距离（`average`）

+   来自成对集群的观察之间的最大距离（`complete`）

还有两个有用的参数需要知道。首先，`affinity`参数确定用于`linkage`的距离度量（`minkowski`、`euclidean`等）。其次，`n_clusters`设置聚类算法将尝试找到的聚类数。也就是说，集群被连续合并，直到只剩下`n_clusters`。

与我们讨论过的其他聚类算法一样，我们可以使用`labels_`来查看每个观察被分配到的集群：

```py
# Show cluster membership
model.labels_
```

```py
array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1,
       1, 1, 1, 1, 1, 1, 0, 0, 0, 2, 0, 2, 0, 2, 0, 2, 2, 0, 2, 0, 2, 0,
       2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 2, 0, 0, 2,
       2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
```
