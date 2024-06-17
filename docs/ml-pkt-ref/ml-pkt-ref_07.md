# 第七章. 数据预处理

本章将探讨使用此数据的常见预处理步骤：

```py
>>> X2 = pd.DataFrame(
...     {
...         "a": range(5),
...         "b": [-100, -50, 0, 200, 1000],
...     }
... )
>>> X2
 a     b
0  0  -100
1  1   -50
2  2     0
3  3   200
4  4  1000
```

# 标准化

某些算法（如 SVM）在数据*标准化*时表现更好。每列的均值应为 0，标准偏差应为 1。Sklearn 提供了一个`.fit_transform`方法，结合了`.fit`和`.transform`：

```py
>>> from sklearn import preprocessing
>>> std = preprocessing.StandardScaler()
>>> std.fit_transform(X2)
array([[-1.41421356, -0.75995002],
 [-0.70710678, -0.63737744],
 [ 0\.        , -0.51480485],
 [ 0.70710678, -0.02451452],
 [ 1.41421356,  1.93664683]])
```

拟合后，我们可以检查各种属性：

```py
>>> std.scale_
array([  1.41421356, 407.92156109])
>>> std.mean_
array([  2., 210.])
>>> std.var_
array([2.000e+00, 1.664e+05])
```

这是一个 pandas 版本。请记住，如果用于预处理，您需要跟踪原始的均值和标准偏差。稍后用于预测的任何样本都需要使用这些相同的值进行标准化：

```py
>>> X_std = (X2 - X2.mean()) / X2.std()
>>> X_std
 a         b
0 -1.264911 -0.679720
1 -0.632456 -0.570088
2  0.000000 -0.460455
3  0.632456 -0.021926
4  1.264911  1.732190

>>> X_std.mean()
a    4.440892e-17
b    0.000000e+00
dtype: float64

>>> X_std.std()
a    1.0
b    1.0
dtype: float64
```

fastai 库也实现了这一功能：

```py
>>> X3 = X2.copy()
>>> from fastai.structured import scale_vars
>>> scale_vars(X3, mapper=None)
>>> X3.std()
a    1.118034
b    1.118034
dtype: float64
>>> X3.mean()
a    0.000000e+00
b    4.440892e-17
dtype: float64
```

# 缩放到范围

缩放到范围是将数据转换为 0 到 1 之间的值。限制数据的范围可能是有用的。但是，如果您有异常值，可能需要谨慎使用此方法：

```py
>>> from sklearn import preprocessing
>>> mms = preprocessing.MinMaxScaler()
>>> mms.fit(X2)
>>> mms.transform(X2)
array([[0\.     , 0\.     ],
 [0.25   , 0.04545],
 [0.5    , 0.09091],
 [0.75   , 0.27273],
 [1\.     , 1\.     ]])
```

这是一个 pandas 版本：

```py
>>> (X2 - X2.min()) / (X2.max() - X2.min())
 a         b
0  0.00  0.000000
1  0.25  0.045455
2  0.50  0.090909
3  0.75  0.272727
4  1.00  1.000000
```

# 虚拟变量

我们可以使用 pandas 从分类数据创建虚拟变量。这也称为独热编码或指示编码。如果数据是名义（无序）的，虚拟变量特别有用。pandas 中的`get_dummies`函数为分类列创建多个列，每列中的值为 1 或 0，如果原始列有该值：

```py
>>> X_cat = pd.DataFrame(
...     {
...         "name": ["George", "Paul"],
...         "inst": ["Bass", "Guitar"],
...     }
... )
>>> X_cat
 name    inst
0  George    Bass
1    Paul  Guitar
```

这是 pandas 版本。注意`drop_first`选项可以用来消除一列（虚拟列中的一列是其他列的线性组合）。

```py
>>> pd.get_dummies(X_cat, drop_first=True)
 name_Paul  inst_Guitar
0          0            0
1          1            1
```

pyjanitor 库还具有使用`expand_column`函数拆分列的功能：

```py
>>> X_cat2 = pd.DataFrame(
...     {
...         "A": [1, None, 3],
...         "names": [
...             "Fred,George",
...             "George",
...             "John,Paul",
...         ],
...     }
... )
>>> jn.expand_column(X_cat2, "names", sep=",")
 A        names  Fred  George  John  Paul
0  1.0  Fred,George     1       1     0     0
1  NaN       George     0       1     0     0
2  3.0    John,Paul     0       0     1     1
```

如果我们有高基数名义数据，我们可以使用*标签编码*。这将在下一节介绍。

# 标签编码器

虚拟变量编码的替代方法是标签编码。这将把分类数据转换为数字。对于高基数数据非常有用。该编码器强加序数性，这可能是需要的也可能不需要的。它所占空间比独热编码少，而且一些（树）算法可以处理此编码。

标签编码器一次只能处理一列：

```py
>>> from sklearn import preprocessing
>>> lab = preprocessing.LabelEncoder()
>>> lab.fit_transform(X_cat)
array([0,1])
```

如果您已经编码了值，则可以应用`.inverse_transform`方法对其进行解码：

```py
>>> lab.inverse_transform([1, 1, 0])
array(['Guitar', 'Guitar', 'Bass'], dtype=object)
```

您也可以使用 pandas 进行标签编码。首先，将列转换为分类列类型，然后从中提取数值代码。

此代码将从 pandas 系列创建新的数值数据。我们使用`.as_ordered`方法确保分类是有序的：

```py
>>> X_cat.name.astype(
...     "category"
... ).cat.as_ordered().cat.codes + 1
0    1
1    2
dtype: int8
```

# 频率编码

处理高基数分类数据的另一种选择是*频率编码*。这意味着用训练数据中的计数替换类别的名称。我们将使用 pandas 来执行此操作。首先，我们将使用 pandas 的`.value_counts`方法创建一个映射（将字符串映射到计数的 pandas 系列）。有了映射，我们可以使用`.map`方法进行编码：

```py
>>> mapping = X_cat.name.value_counts()
>>> X_cat.name.map(mapping)
0    1
1    1
Name: name, dtype: int64
```

确保存储训练映射，以便以后使用相同的数据对未来数据进行编码。

# 从字符串中提取类别

提高 Titanic 模型准确性的一种方法是从姓名中提取称号。找到最常见的三重组的一个快速方法是使用 `Counter` 类：

```py
>>> from collections import Counter
>>> c = Counter()
>>> def triples(val):
...     for i in range(len(val)):
...         c[val[i : i + 3]] += 1
>>> df.name.apply(triples)
>>> c.most_common(10)
[(', M', 1282),
 (' Mr', 954),
 ('r. ', 830),
 ('Mr.', 757),
 ('s. ', 460),
 ('n, ', 320),
 (' Mi', 283),
 ('iss', 261),
 ('ss.', 261),
 ('Mis', 260)]
```

我们可以看到，“Mr.” 和 “Miss.” 非常常见。

另一个选项是使用正则表达式提取大写字母后跟小写字母和句点：

```py
>>> df.name.str.extract(
...     "([A-Za-z]+)\.", expand=False
... ).head()
0      Miss
1    Master
2      Miss
3        Mr
4       Mrs
Name: name, dtype: object
```

我们可以使用 `.value_counts` 查看这些的频率：

```py
>>> df.name.str.extract(
...     "([A-Za-z]+)\.", expand=False
... ).value_counts()
Mr          757
Miss        260
Mrs         197
Master       61
Dr            8
Rev           8
Col           4
Mlle          2
Ms            2
Major         2
Dona          1
Don           1
Lady          1
Countess      1
Capt          1
Sir           1
Mme           1
Jonkheer      1
Name: name, dtype: int64
```

###### 注意

对正则表达式的完整讨论超出了本书的范围。此表达式捕获一个或多个字母字符的组。该组后面将跟随一个句点。

使用这些操作和 pandas，您可以创建虚拟变量或将低计数的列组合到其他类别中（或将它们删除）。

# 其他分类编码

[categorical_encoding 库](https://oreil.ly/JbxWG) 是一组 scikit-learn 转换器，用于将分类数据转换为数值数据。该库的一个优点是它支持输出 pandas DataFrame（不像 scikit-learn 那样将它们转换为 numpy 数组）。

该库实现的一个算法是哈希编码器。如果您事先不知道有多少类别，或者正在使用词袋表示文本，这将会很有用。它将分类列哈希到 `n_components`。如果您使用在线学习（可以更新模型），这将非常有用：

```py
>>> import category_encoders as ce
>>> he = ce.HashingEncoder(verbose=1)
>>> he.fit_transform(X_cat)
 col_0  col_1  col_2  col_3  col_4  col_5  col_6  col_7
0      0      0      0      1      0      1      0      0
1      0      2      0      0      0      0      0      0
```

序数编码器可以将具有顺序的分类列转换为单个数字列。在这里，我们将大小列转换为序数数字。如果在映射字典中找不到一个值，则使用 `-1` 的默认值：

```py
>>> size_df = pd.DataFrame(
...     {
...         "name": ["Fred", "John", "Matt"],
...         "size": ["small", "med", "xxl"],
...     }
... )
>>> ore = ce.OrdinalEncoder(
...     mapping=[
...         {
...             "col": "size",
...             "mapping": {
...                 "small": 1,
...                 "med": 2,
...                 "lg": 3,
...             },
...         }
...     ]
... )
>>> ore.fit_transform(size_df)
 name  size
0  Fred   1.0
1  John   2.0
2  Matt  -1.0
```

此 [参考](https://oreil.ly/JUtYh) 解释了 categorical_encoding 库的许多算法。

如果您有高基数数据（大量唯一值），考虑使用其中一个贝叶斯编码器，它们会为每个分类列输出单独的列。这些包括 `TargetEncoder`, `LeaveOneOutEncoder`, `WOEEncoder`, `JamesSteinEncoder` 和 `MEstimateEncoder`。

例如，要将 Titanic 生存列转换为目标的后验概率和给定标题（分类）信息的先验概率的混合，可以使用以下代码：

```py
>>> def get_title(df):
...     return df.name.str.extract(
...         "([A-Za-z]+)\.", expand=False
...     )
>>> te = ce.TargetEncoder(cols="Title")
>>> te.fit_transform(
...     df.assign(Title=get_title), df.survived
... )["Title"].head()
0    0.676923
1    0.508197
2    0.676923
3    0.162483
4    0.786802
Name: Title, dtype: float64
```

# 日期特征工程

fastai 库有一个 `add_datepart` 函数，它将根据日期时间列生成日期属性列。这对大多数机器学习算法很有用，因为它们无法从日期的数值表示中推断出这种类型的信号：

```py
>>> from fastai.tabular.transform import (
...     add_datepart,
... )
>>> dates = pd.DataFrame(
...     {
...         "A": pd.to_datetime(
...             ["9/17/2001", "Jan 1, 2002"]
...         )
...     }
... )

>>> add_datepart(dates, "A")
>>> dates.T
 0           1
AYear                    2001        2002
AMonth                      9           1
AWeek                      38           1
ADay                       17           1
ADayofweek                  0           1
ADayofyear                260           1
AIs_month_end           False       False
AIs_month_start         False        True
AIs_quarter_end         False       False
AIs_quarter_start       False        True
AIs_year_end            False       False
AIs_year_start          False        True
AElapsed           1000684800  1009843200
```

###### 警告

`add_datepart` 会改变 DataFrame，这是 pandas 能做到的，但通常不这样做！

# 添加 col_na 特征

fastai 库曾经有一个函数用于创建一个列以填充缺失值（使用中位数）并指示缺失值。知道一个值是否缺失可能会有一些信号。以下是该函数的副本及其使用示例：

```py
>>> from pandas.api.types import is_numeric_dtype
>>> def fix_missing(df, col, name, na_dict):
...     if is_numeric_dtype(col):
...         if pd.isnull(col).sum() or (
...             name in na_dict
...         ):
...             df[name + "_na"] = pd.isnull(col)
...             filler = (
...                 na_dict[name]
...                 if name in na_dict
...                 else col.median()
...             )
...             df[name] = col.fillna(filler)
...             na_dict[name] = filler
...     return na_dict
>>> data = pd.DataFrame({"A": [0, None, 5, 100]})
>>> fix_missing(data, data.A, "A", {})
{'A': 5.0}
>>> data
 A   A_na
0    0.0  False
1    5.0   True
2    5.0  False
3  100.0  False
```

下面是一个 pandas 版本：

```py
>>> data = pd.DataFrame({"A": [0, None, 5, 100]})
>>> data["A_na"] = data.A.isnull()
>>> data["A"] = data.A.fillna(data.A.median())
```

# 手动特征工程

我们可以使用 pandas 生成新特征。对于 Titanic 数据集，我们可以添加聚合船舱数据（每个船舱的最大年龄、平均年龄等）。要获得每个船舱的聚合数据并将其合并回原始数据中，使用 pandas 的 `.groupby` 方法创建数据。然后使用 `.merge` 方法将其与原始数据对齐：

```py
>>> agg = (
...     df.groupby("cabin")
...     .agg("min,max,mean,sum".split(","))
...     .reset_index()
... )
>>> agg.columns = [
...     "_".join(c).strip("_")
...     for c in agg.columns.values
... ]
>>> agg_df = df.merge(agg, on="cabin")
```

如果你想总结“好”或“坏”列，你可以创建一个新列，该列是聚合列的总和（或另一个数学操作）。这在某种程度上是一种艺术，也需要理解数据。
