# 第十二章 模型选择

# 12.0 引言

在机器学习中，我们使用训练算法通过最小化某个损失函数来学习模型的参数。然而，许多学习算法（例如支持向量分类器和随机森林）有额外的 *超参数*，由用户定义，并影响模型学习其参数的方式。正如我们在本书的前面提到的，*参数*（有时也称为模型权重）是模型在训练过程中学习的内容，而超参数是我们手动提供的（用户提供的）内容。

例如，随机森林是决策树的集合（因此有 *森林* 一词）；然而，森林中决策树的数量并非由算法学习，而必须在拟合之前设置好。这通常被称为 *超参数调优*、*超参数优化* 或 *模型选择*。此外，我们可能希望尝试多个学习算法（例如尝试支持向量分类器和随机森林，看哪种学习方法产生最佳模型）。

尽管在这个领域术语广泛变化，但在本书中，我们将选择最佳学习算法及其最佳超参数称为模型选择。原因很简单：想象我们有数据，并且想要训练一个支持向量分类器，有 10 个候选超参数值，以及一个随机森林分类器，有 10 个候选超参数值。结果是我们尝试从一组 20 个候选模型中选择最佳模型。在本章中，我们将介绍有效地从候选集中选择最佳模型的技术。

在本章中，我们将提到特定的超参数，比如 C（正则化强度的倒数）。如果你不知道超参数是什么，不要担心。我们将在后面的章节中介绍它们。相反，只需将超参数视为在开始训练之前必须选择的学习算法的设置。通常，找到能够产生最佳性能的模型和相关超参数是实验的结果——尝试各种可能性并找出最佳的那个。

# 12.1 使用穷举搜索选择最佳模型

## 问题

你想通过搜索一系列超参数来选择最佳模型。

## 解决方案

使用 scikit-learn 的 `GridSearchCV`：

```py
# Load libraries
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Create logistic regression
logistic = linear_model.LogisticRegression(max_iter=500, solver='liblinear')

# Create range of candidate penalty hyperparameter values
penalty = ['l1','l2']

# Create range of candidate regularization hyperparameter values
C = np.logspace(0, 4, 10)

# Create dictionary of hyperparameter candidates
hyperparameters = dict(C=C, penalty=penalty)

# Create grid search
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)

# Fit grid search
best_model = gridsearch.fit(features, target)

# Show the best model
print(best_model.best_estimator_)
```

```py
LogisticRegression(C=7.742636826811269, max_iter=500, penalty='l1',
                   solver='liblinear')
```

## 讨论

`GridSearchCV` 是一种使用交叉验证进行模型选择的蛮力方法。具体来说，用户定义一个或多个超参数可能的值集合，然后 `GridSearchCV` 使用每个值和/或值组合来训练模型。选择具有最佳性能得分的模型作为最佳模型。

例如，在我们的解决方案中，我们使用逻辑回归作为我们的学习算法，并调整了两个超参数：C 和正则化惩罚。我们还指定了另外两个参数，解算器和最大迭代次数。如果您不知道这些术语的含义也没关系；我们将在接下来的几章中详细讨论它们。只需意识到 C 和正则化惩罚可以取一系列值，这些值在训练之前必须指定。对于 C，我们定义了 10 个可能的值：

```py
np.logspace(0, 4, 10)
```

```py
array([1.00000000e+00, 2.78255940e+00, 7.74263683e+00, 2.15443469e+01,
       5.99484250e+01, 1.66810054e+02, 4.64158883e+02, 1.29154967e+03,
       3.59381366e+03, 1.00000000e+04])
```

类似地，我们定义了两个正则化惩罚的可能值：`['l1', 'l2']`。对于每个 C 和正则化惩罚值的组合，我们训练模型并使用 k 折交叉验证进行评估。在我们的解决方案中，C 有 10 个可能的值，正则化惩罚有 2 个可能的值，并且使用 5 折交叉验证。它们创建了 10 × 2 × 5 = 100 个候选模型，其中选择最佳模型。

一旦完成`GridSearchCV`，我们可以看到最佳模型的超参数：

```py
# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
```

```py
Best Penalty: l1
Best C: 7.742636826811269
```

默认情况下，确定了最佳超参数后，`GridSearchCV`会在整个数据集上重新训练一个模型（而不是留出一个折用于交叉验证）。我们可以像对待其他 scikit-learn 模型一样使用该模型来预测值：

```py
# Predict target vector
best_model.predict(features)
```

```py
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
```

`GridSearchCV`的一个参数值得注意：`verbose`。虽然大多数情况下不需要，但在长时间搜索过程中，接收到搜索正在进行中的指示可能会让人放心。`verbose`参数确定了搜索过程中输出消息的数量，`0`表示没有输出，而`1`到`3`表示额外的输出消息。

## 参见

+   [scikit-learn 文档：GridSearchCV](https://oreil.ly/XlMPG)

# 12.2 使用随机搜索选择最佳模型

## 问题

您希望选择最佳模型的计算成本较低的方法。

## 解决方案

使用 scikit-learn 的`RandomizedSearchCV`：

```py
# Load libraries
from scipy.stats import uniform
from sklearn import linear_model, datasets
from sklearn.model_selection import RandomizedSearchCV

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Create logistic regression
logistic = linear_model.LogisticRegression(max_iter=500, solver='liblinear')

# Create range of candidate regularization penalty hyperparameter values
penalty = ['l1', 'l2']

# Create distribution of candidate regularization hyperparameter values
C = uniform(loc=0, scale=4)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

# Create randomized search
randomizedsearch = RandomizedSearchCV(
    logistic, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0,
    n_jobs=-1)

# Fit randomized search
best_model = randomizedsearch.fit(features, target)

# Print best model
print(best_model.best_estimator_)
```

```py
LogisticRegression(C=1.668088018810296, max_iter=500, penalty='l1',
                   solver='liblinear')
```

## 讨论

在 Recipe 12.1 中，我们使用`GridSearchCV`在用户定义的一组超参数值上搜索最佳模型，根据评分函数。比`GridSearchCV`的蛮力搜索更高效的方法是从用户提供的分布（例如正态分布、均匀分布）中随机组合一定数量的超参数值进行搜索。scikit-learn 使用`RandomizedSearchCV`实现了这种随机搜索技术。

使用`RandomizedSearchCV`，如果我们指定一个分布，scikit-learn 将从该分布中随机抽样且不重复地抽取超参数值。例如，这里我们从范围为 0 到 4 的均匀分布中随机抽取 10 个值作为一般概念的示例：

```py
# Define a uniform distribution between 0 and 4, sample 10 values
uniform(loc=0, scale=4).rvs(10)
```

```py
array([3.95211699, 0.30693116, 2.88237794, 3.00392864, 0.43964702,
       1.46670526, 0.27841863, 2.56541664, 2.66475584, 0.79611958])
```

或者，如果我们指定一个值列表，例如两个正则化惩罚超参数值`['l1', 'l2']`，`RandomizedSearchCV`将从列表中进行带替换的随机抽样。

就像`GridSearchCV`一样，我们可以看到最佳模型的超参数值：

```py
# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
```

```py
Best Penalty: l1
Best C: 1.668088018810296
```

就像使用`GridSearchCV`一样，在完成搜索后，`RandomizedSearchCV`会使用最佳超参数在整个数据集上拟合一个新模型。我们可以像使用 scikit-learn 中的任何其他模型一样使用这个模型；例如，进行预测：

```py
# Predict target vector
best_model.predict(features)
```

```py
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2,
       2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
```

超参数组合的采样数量（即训练的候选模型数量）由`n_iter`（迭代次数）设置指定。值得注意的是，`RandomizedSearchCV`并不比`GridSearchCV`更快，但通常在较短时间内通过测试更少的组合来实现与`GridSearchCV`可比较的性能。

## 参见

+   [scikit-learn 文档：RandomizedSearchCV](https://oreil.ly/rpiSs)

+   [用于超参数优化的随机搜索](https://oreil.ly/iBcbo)

# 12.3 从多个学习算法中选择最佳模型

## 问题

通过在一系列学习算法及其相应的超参数上进行搜索，您可以选择最佳模型。

## 解决方案

创建一个包含候选学习算法及其超参数的字典，作为`GridSearchCV`的搜索空间：

```py
# Load libraries
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Set random seed
np.random.seed(0)

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Create a pipeline
pipe = Pipeline([("classifier", RandomForestClassifier())])

# Create dictionary with candidate learning algorithms and their hyperparameters
search_space = [{"classifier": [LogisticRegression(max_iter=500,
       solver='liblinear')],
                 "classifier__penalty": ['l1', 'l2'],
                 "classifier__C": np.logspace(0, 4, 10)},
                {"classifier": [RandomForestClassifier()],
                 "classifier__n_estimators": [10, 100, 1000],
                 "classifier__max_features": [1, 2, 3]}]

# Create grid search
gridsearch = GridSearchCV(pipe, search_space, cv=5, verbose=0)

# Fit grid search
best_model = gridsearch.fit(features, target)

# Print best model
print(best_model.best_estimator_)
```

```py
Pipeline(steps=[('classifier',
                 LogisticRegression(C=7.742636826811269, max_iter=500,
                                    penalty='l1', solver='liblinear'))])
```

## 讨论

在前两个示例中，我们通过搜索学习算法的可能超参数值来找到最佳模型。但是，如果我们不确定要使用哪种学习算法怎么办？scikit-learn 允许我们将学习算法作为搜索空间的一部分。在我们的解决方案中，我们定义了一个搜索空间，其中包含两个学习算法：逻辑回归和随机森林分类器。每个学习算法都有自己的超参数，并且我们使用`classifier__[*hyperparameter name*]`的格式定义其候选值。例如，对于我们的逻辑回归，为了定义可能的正则化超参数空间`C`的可能值集合以及潜在的正则化惩罚类型`penalty`，我们创建一个字典：

```py
{'classifier': [LogisticRegression(max_iter=500, solver='liblinear')],
 'classifier__penalty': ['l1', 'l2'],
 'classifier__C': np.logspace(0, 4, 10)}
```

我们也可以为随机森林的超参数创建一个类似的字典：

```py
{'classifier': [RandomForestClassifier()],
 'classifier__n_estimators': [10, 100, 1000],
 'classifier__max_features': [1, 2, 3]}
```

完成搜索后，我们可以使用 `best_estimator_` 查看最佳模型的学习算法和超参数：

```py
# View best model
print(best_model.best_estimator_.get_params()["classifier"])
```

```py
LogisticRegression(C=7.742636826811269, max_iter=500, penalty='l1',
                   solver='liblinear')
```

就像前面两个示例一样，一旦我们完成了模型选择搜索，我们就可以像使用任何其他 scikit-learn 模型一样使用这个最佳模型：

```py
# Predict target vector
best_model.predict(features)
```

```py
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
```

# 12.4 在预处理时选择最佳模型

## 问题

您希望在模型选择过程中包含一个预处理步骤。

## 解决方案

创建一个包含预处理步骤及其任何参数的管道： 

```py
# Load libraries
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set random seed
np.random.seed(0)

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Create a preprocessing object that includes StandardScaler features and PCA
preprocess = FeatureUnion([("std", StandardScaler()), ("pca", PCA())])

# Create a pipeline
pipe = Pipeline([("preprocess", preprocess),
                 ("classifier", LogisticRegression(max_iter=1000,
                     solver='liblinear'))])

# Create space of candidate values
search_space = [{"preprocess__pca__n_components": [1, 2, 3],
                 "classifier__penalty": ["l1", "l2"],
                 "classifier__C": np.logspace(0, 4, 10)}]

# Create grid search
clf = GridSearchCV(pipe, search_space, cv=5, verbose=0, n_jobs=-1)

# Fit grid search
best_model = clf.fit(features, target)

# Print best model
print(best_model.best_estimator_)
```

```py
Pipeline(steps=[('preprocess',
                 FeatureUnion(transformer_list=[('std', StandardScaler()),
                                                ('pca', PCA(n_components=1))])),
                ('classifier',
                 LogisticRegression(C=7.742636826811269, max_iter=1000,
                                    penalty='l1', solver='liblinear'))])
```

## 讨论

在使用数据训练模型之前，我们通常需要对数据进行预处理。在进行模型选择时，我们必须小心处理预处理。首先，`GridSearchCV`使用交叉验证确定哪个模型性能最高。然而，在交叉验证中，我们实际上是在假装保留为测试集的折叠是不可见的，因此不是拟合任何预处理步骤的一部分（例如，缩放或标准化）。因此，我们不能对数据进行预处理然后运行`GridSearchCV`。相反，预处理步骤必须成为`GridSearchCV`采取的操作集的一部分。

这可能看起来很复杂，但 scikit-learn 让它变得简单。`FeatureUnion`允许我们正确地组合多个预处理操作。在我们的解决方案中，我们使用`FeatureUnion`来组合两个预处理步骤：标准化特征值（`StandardScaler`）和主成分分析（`PCA`）。该对象称为`preprocess`，包含我们的两个预处理步骤。然后，我们将`preprocess`包含在一个管道中与我们的学习算法一起。结果是，这使我们能够将拟合、转换和训练模型与超参数组合的正确（而令人困惑的）处理外包给 scikit-learn。

第二，一些预处理方法有它们自己的参数，通常需要用户提供。例如，使用 PCA 进行降维需要用户定义要使用的主成分数量，以产生转换后的特征集。理想情况下，我们会选择产生在某个评估测试指标下性能最佳模型的组件数量。

幸运的是，scikit-learn 使这变得容易。当我们在搜索空间中包含候选组件值时，它们被视为要搜索的任何其他超参数。在我们的解决方案中，我们在搜索空间中定义了`features__pca__n_components': [1, 2, 3]`，以指示我们要发现一个、两个或三个主成分是否产生最佳模型。

在模型选择完成后，我们可以查看产生最佳模型的预处理值。例如，我们可以查看最佳的主成分数量：

```py
# View best n_components
best_model.best_estimator_.get_params()['preprocess__pca__n_components']
```

```py
1
```

# 12.5 使用并行化加快模型选择速度

## 问题

需要加快模型选择速度。

## 解决方案

通过设置`n_jobs=-1`来利用机器中的所有核心，从而使您能够同时训练多个模型：

```py
# Load libraries
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Create logistic regression
logistic = linear_model.LogisticRegression(max_iter=500, solver='liblinear')

# Create range of candidate regularization penalty hyperparameter values
penalty = ["l1", "l2"]

# Create range of candidate values for C
C = np.logspace(0, 4, 1000)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

# Create grid search
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, n_jobs=-1, verbose=1)

# Fit grid search
best_model = gridsearch.fit(features, target)

# Print best model
print(best_model.best_estimator_)
```

```py
Fitting 5 folds for each of 2000 candidates, totalling 10000 fits
LogisticRegression(C=5.926151812475554, max_iter=500, penalty='l1',
                   solver='liblinear')
```

## 讨论

在本章的示例中，我们将候选模型的数量保持较少，以使代码迅速完整。但是，在现实世界中，我们可能有成千上万甚至成千上万个模型要训练。因此，找到最佳模型可能需要花费很多小时。

为了加快这一过程，scikit-learn 允许我们同时训练多个模型。不深入技术细节，scikit-learn 可以同时训练多达机器上的核心数量的模型。现代大多数笔记本电脑至少有四个核心，因此（假设您当前使用的是笔记本电脑），我们可以同时训练四个模型。这将大大增加我们模型选择过程的速度。参数 `n_jobs` 定义了并行训练的模型数量。

在我们的解决方案中，我们将 `n_jobs` 设置为 `-1`，这告诉 scikit-learn 使用 *所有* 核心。然而，默认情况下 `n_jobs` 被设置为 `1`，这意味着它只使用一个核心。为了演示这一点，如果我们像在解决方案中一样运行相同的 `GridSearchCV`，但使用 `n_jobs=1`，我们可以看到找到最佳模型要花费显著更长的时间（确切时间取决于您的计算机）：

```py
# Create grid search using one core
clf = GridSearchCV(logistic, hyperparameters, cv=5, n_jobs=1, verbose=1)

# Fit grid search
best_model = clf.fit(features, target)

# Print best model
print(best_model.best_estimator_)
```

```py
Fitting 5 folds for each of 2000 candidates, totalling 10000 fits
LogisticRegression(C=5.926151812475554, max_iter=500, penalty='l1',
                   solver='liblinear')
```

# 12.6 使用算法特定方法加速模型选择

## 问题

您需要加速模型选择，但不使用额外的计算资源。

## 解决方案

如果您正在使用一些特定的学习算法，请使用 scikit-learn 的模型特定的交叉验证超参数调整，例如 `LogisticRegressionCV`：

```py
# Load libraries
from sklearn import linear_model, datasets

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Create cross-validated logistic regression
logit = linear_model.LogisticRegressionCV(Cs=100, max_iter=500,
       solver='liblinear')

# Train model
logit.fit(features, target)

# Print model
print(logit)
```

```py
LogisticRegressionCV(Cs=100, max_iter=500, solver='liblinear')
```

## 讨论

有时候学习算法的特性使我们能够比蛮力或随机模型搜索方法显著更快地搜索最佳超参数。在 scikit-learn 中，许多学习算法（例如岭回归、套索回归和弹性网络回归）都有一种特定于算法的交叉验证方法，以利用这一点。例如，`LogisticRegression` 用于进行标准的逻辑回归分类器，而 `LogisticRegressionCV` 实现了一个高效的交叉验证逻辑回归分类器，可以识别超参数 C 的最佳值。

scikit-learn 的 `LogisticRegressionCV` 方法包括参数 `Cs`。如果提供了一个列表，`Cs` 包含要从中选择的候选超参数值。如果提供了一个整数，参数 `Cs` 将生成该数量的候选值列表。候选值从 0.0001 到 10,0000 的对数范围内抽取（这是 C 的合理值范围）。

然而，`LogisticRegressionCV` 的一个主要缺点是它只能搜索 C 的一系列值。在 配方 12.1 中，我们的可能超参数空间包括 C 和另一个超参数（正则化惩罚范数）。这种限制是许多 scikit-learn 模型特定的交叉验证方法的共同特点。

## 参见

+   [scikit-learn 文档：LogisticRegressionCV](https://oreil.ly/uguJi)

+   [scikit-learn 文档：模型特定的交叉验证](https://oreil.ly/6xfn6)

# 12.7 在模型选择后评估性能

## 问题

您希望评估通过模型选择找到的模型的性能。

## 解决方案

使用嵌套交叉验证以避免偏倚评估：

```py
# Load libraries
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV, cross_val_score

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Create logistic regression
logistic = linear_model.LogisticRegression(max_iter=500, solver='liblinear')

# Create range of 20 candidate values for C
C = np.logspace(0, 4, 20)

# Create hyperparameter options
hyperparameters = dict(C=C)

# Create grid search
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, n_jobs=-1, verbose=0)

# Conduct nested cross-validation and output the average score
cross_val_score(gridsearch, features, target).mean()
```

```py
0.9733333333333334
```

## 讨论

在模型选择过程中的嵌套交叉验证对许多人来说是一个难以理解的概念。请记住，在 k 折交叉验证中，我们在数据的 *k-1* 折上训练模型，使用该模型对剩余的一折进行预测，然后评估我们的模型预测与真实值的比较。然后我们重复这个过程 *k* 次。

在本章描述的模型选择搜索中（即 `GridSearchCV` 和 `RandomizedSearchCV`），我们使用交叉验证来评估哪些超参数值产生了最佳模型。然而，一个微妙且通常被低估的问题出现了：因为我们使用数据来选择最佳的超参数值，所以我们不能再使用同样的数据来评估模型的性能。解决方案是？将用于模型搜索的交叉验证包装在另一个交叉验证中！在嵌套交叉验证中，“内部”交叉验证选择最佳模型，而“外部”交叉验证提供了模型性能的无偏评估。在我们的解决方案中，内部交叉验证是我们的 `GridSearchCV` 对象，然后我们使用 `cross_val_score` 将其包装在外部交叉验证中。

如果你感到困惑，可以尝试一个简单的实验。首先，设置 `verbose=1`，这样我们可以看到发生了什么：

```py
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, verbose=1)
```

接下来，运行 `gridsearch.fit(features, target)`，这是我们用来找到最佳模型的内部交叉验证：

```py
best_model = gridsearch.fit(features, target)
```

```py
Fitting 5 folds for each of 20 candidates, totalling 100 fits
```

从输出中可以看出，内部交叉验证训练了 20 个候选模型五次，总计 100 个模型。接下来，将 `clf` 嵌套在一个新的交叉验证中，默认为五折：

```py
scores = cross_val_score(gridsearch, features, target)
```

```py
Fitting 5 folds for each of 20 candidates, totalling 100 fits
Fitting 5 folds for each of 20 candidates, totalling 100 fits
Fitting 5 folds for each of 20 candidates, totalling 100 fits
Fitting 5 folds for each of 20 candidates, totalling 100 fits
Fitting 5 folds for each of 20 candidates, totalling 100 fits
```

输出显示，内部交叉验证训练了 20 个模型五次，以找到最佳模型，然后使用外部五折交叉验证评估了该模型，总共训练了 500 个模型。
