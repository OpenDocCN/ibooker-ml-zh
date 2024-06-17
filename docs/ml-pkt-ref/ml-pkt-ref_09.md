# 第九章：不平衡类别

如果您正在对数据进行分类，并且类的大小不相对平衡，那么对更流行的类别的偏向可能会在模型中体现出来。例如，如果您有 1 个正例和 99 个负例，您可以通过将所有内容分类为负例获得 99％的准确率。处理*不平衡类别*的各种选项。

# 使用不同的度量标准

一个提示是使用除准确率之外的度量（AUC 是一个不错的选择）来校准模型。当目标大小不同时，精确度和召回率也是更好的选择。但是，还有其他考虑的选项。

# 基于树的算法和集成方法

基于树的模型可能根据较小类的分布表现更好。如果它们倾向于聚集，它们可以更容易地被分类。

集成方法可以进一步帮助提取少数类。装袋和提升是树模型（如随机森林和极端梯度增强（XGBoost））中的选项。

# 惩罚模型

许多 scikit-learn 分类模型支持`class_weight`参数。将其设置为`'balanced'`将尝试正则化少数类，并激励模型正确分类它们。或者，您可以进行网格搜索，并通过传递将类映射到权重的字典来指定权重选项（给较小类更高的权重）。

[XGBoost](https://xgboost.readthedocs.io)库有`max_delta_step`参数，可以设置为 1 到 10，使更新步骤更加保守。它还有`scale_pos_weight`参数，用于设置负样本到正样本的比例（针对二元类）。此外，对于分类，`eval_metric`应设置为`'auc'`而不是默认值`'error'`。

KNN 模型有一个`weights`参数，可以偏向邻近的邻居。如果少数类样本靠近在一起，将此参数设置为`'distance'`可能会提高性能。

# 上采样少数类

您可以通过几种方式增加少数类。以下是一个 sklearn 的实现：

```py
>>> from sklearn.utils import resample
>>> mask = df.survived == 1
>>> surv_df = df[mask]
>>> death_df = df[~mask]
>>> df_upsample = resample(
...     surv_df,
...     replace=True,
...     n_samples=len(death_df),
...     random_state=42,
... )
>>> df2 = pd.concat([death_df, df_upsample])

>>> df2.survived.value_counts()
1    809
0    809
Name: survived, dtype: int64
```

我们还可以使用 imbalanced-learn 库进行随机替换的采样：

```py
>>> from imblearn.over_sampling import (
...     RandomOverSampler,
... )
>>> ros = RandomOverSampler(random_state=42)
>>> X_ros, y_ros = ros.fit_sample(X, y)
>>> pd.Series(y_ros).value_counts()
1    809
0    809
dtype: int64
```

# 生成少数类数据

imbalanced-learn 库还可以使用 Synthetic Minority Over-sampling Technique（SMOTE）和 Adaptive Synthetic（ADASYN）采样方法生成少数类的新样本。SMOTE 通过选择其 k 个最近邻之一，连接到其中之一，并沿着该线选择一个点来工作。ADASYN 与 SMOTE 类似，但会从更难学习的样本生成更多样本。imbanced-learn 中的类名为`over_sampling.SMOTE`和`over_sampling.ADASYN`。

# 下采样多数类

平衡类别的另一种方法是对多数类进行下采样。以下是一个 sklearn 的例子：

```py
>>> from sklearn.utils import resample
>>> mask = df.survived == 1
>>> surv_df = df[mask]
>>> death_df = df[~mask]
>>> df_downsample = resample(
...     death_df,
...     replace=False,
...     n_samples=len(surv_df),
...     random_state=42,
... )
>>> df3 = pd.concat([surv_df, df_downsample])

>>> df3.survived.value_counts()
1    500
0    500
Name: survived, dtype: int64
```

###### 提示

在进行下采样时不要使用替换。

imbalanced-learn 库还实现了各种下采样算法：

`ClusterCentroids`

此类使用 K-means 来合成具有质心的数据。

`RandomUnderSampler`

此类随机选择样本。

`NearMiss`

此类使用最近邻来进行下采样。

`TomekLink`

此类通过移除彼此接近的样本来进行下采样。

`EditedNearestNeighbours`

此类移除具有邻居不在大多数或完全相同类别中的样本。

`RepeatedNearestNeighbours`

此类重复调用 `EditedNearestNeighbours`。

`AllKNN`

此类类似，但在下采样迭代期间增加了最近邻居的数量。

`CondensedNearestNeighbour`

此类选择要下采样的类的一个样本，然后迭代该类的其他样本，如果 KNN 不会误分类，则将该样本添加进去。

`OneSidedSelection`

此类移除噪声样本。

`NeighbourhoodCleaningRule`

此类使用 `EditedNearestNeighbours` 的结果，并对其应用 KNN。

`InstanceHardnessThreshold`

此类训练模型，然后移除概率低的样本。

所有这些类都支持 `.fit_sample` 方法。

# 先上采样再下采样

imbalanced-learn 库实现了 `SMOTEENN` 和 `SMOTETomek`，它们都是先上采样然后再应用下采样来清理数据。
