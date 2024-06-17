# 第三章 数据整理

# 3.0 引言

*数据整理*是一个广义术语，通常非正式地用来描述将原始数据转换为干净、有组织的格式，以便于使用的过程。对于我们来说，数据整理只是数据预处理的一个步骤，但是这是一个重要的步骤。

“整理”数据最常用的数据结构是数据框，它既直观又非常灵活。数据框是表格型的，意味着它们基于行和列，就像您在电子表格中看到的那样。这是一个根据*泰坦尼克号*乘客数据创建的数据框示例：

```py
# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data as a dataframe
dataframe = pd.read_csv(url)

# Show first five rows
dataframe.head(5)
```

|  | 名称 | PClass | 年龄 | 性别 | 幸存 | 性别代码 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 艾伦，伊丽莎白·沃尔顿小姐 | 1st | 29.00 | 女性 | 1 | 1 |
| 1 | 艾莉森，海伦·洛林小姐 | 1st | 2.00 | 女性 | 0 | 1 |
| 2 | 艾莉森，哈德森·约书亚·克莱顿先生 | 1st | 30.00 | 男性 | 0 | 0 |
| 3 | 艾莉森，哈德森 JC 夫人（贝西·沃尔多·丹尼尔斯） | 1st | 25.00 | 女性 | 0 | 1 |
| 4 | 艾莉森，哈德森·特雷弗小主人 | 1st | 0.92 | 男性 | 1 | 0 |

在这个数据框中，有三个重要的事情需要注意。

首先，在数据框中，每一行对应一个观察结果（例如一个乘客），每一列对应一个特征（性别、年龄等）。例如，通过查看第一个观察结果，我们可以看到伊丽莎白·沃尔顿·艾伦小姐住在头等舱，年龄为 29 岁，是女性，并且幸存于这场灾难。

其次，在数据框中，每一行对应一个观察结果（例如一个乘客），每一列对应一个特征（性别、年龄等）。例如，通过查看第一个观察结果，我们可以看到伊丽莎白·沃尔顿·艾伦小姐住在头等舱，年龄为 29 岁，是女性，并且幸存于这场灾难。

第三，两列`Sex`和`SexCode`以不同格式包含相同的信息。在`Sex`中，女性用字符串`female`表示，而在`SexCode`中，女性用整数`1`表示。我们希望所有的特征都是唯一的，因此我们需要删除其中一列。

在本章中，我们将涵盖使用 pandas 库操作数据框的各种技术，旨在创建一个干净、结构良好的观察结果集以便进行进一步的预处理。

# 3.1 创建数据框

## 问题

您想要创建一个新的数据框。

## 解决方案

pandas 有许多用于创建新数据框对象的方法。一个简单的方法是使用 Python 字典实例化一个`DataFrame`。在字典中，每个键是列名，每个值是一个列表，其中每个项目对应一行：

```py
# Load library
import pandas as pd

# Create a dictionary
dictionary = {
  "Name": ['Jacky Jackson', 'Steven Stevenson'],
  "Age": [38, 25],
  "Driver": [True, False]
}

# Create DataFrame
dataframe = pd.DataFrame(dictionary)

# Show DataFrame
dataframe
```

|  | 名称 | 年龄 | 驾驶员 |
| --- | --- | --- | --- |
| 0 | 杰基·杰克逊 | 38 | True |
| 1 | 史蒂文·史蒂文森 | 25 | False |

使用值列表很容易向任何数据框添加新列：

```py
# Add a column for eye color
dataframe["Eyes"] = ["Brown", "Blue"]

# Show DataFrame
dataframe
```

|  | 名称 | 年龄 | 驾驶员 | 眼睛 |
| --- | --- | --- | --- | --- |
| 0 | 杰基·杰克逊 | 38 | True | Brown |
| 1 | 史蒂文·史蒂文森 | 25 | False | 蓝色 |

## 讨论

pandas 提供了几乎无限种方法来创建 DataFrame。在实际应用中，几乎不会先创建一个空的 DataFrame，然后再填充数据。相反，我们的 DataFrame 通常是从其他来源（如 CSV 文件或数据库）加载真实数据而创建的。

# 3.2 获取关于数据的信息

## 问题

您想查看 DataFrame 的一些特征。

## 解决方案

加载数据后，最简单的事情之一是使用 `head` 查看前几行数据：

```py
# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Show two rows
dataframe.head(2)
```

|  | 姓名 | 舱位 | 年龄 | 性别 | 生还 | 性别编码 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Allen, Miss Elisabeth Walton | 1st | 29.0 | 女性 | 1 | 1 |
| 1 | Allison, Miss Helen Loraine | 1st | 2.0 | 女性 | 0 | 1 |

我们也可以查看行数和列数：

```py
# Show dimensions
dataframe.shape
```

```py
(1313, 6)
```

我们可以使用 `describe` 获取任何数值列的描述统计信息：

```py
# Show statistics
dataframe.describe()
```

|  | 年龄 | 生还 | 性别编码 |
| --- | --- | --- | --- |
| count | 756.000000 | 1313.000000 | 1313.000000 |
| mean | 30.397989 | 0.342727 | 0.351866 |
| std | 14.259049 | 0.474802 | 0.477734 |
| min | 0.170000 | 0.000000 | 0.000000 |
| 25% | 21.000000 | 0.000000 | 0.000000 |
| 50% | 28.000000 | 0.000000 | 0.000000 |
| 75% | 39.000000 | 1.000000 | 1.000000 |
| max | 71.000000 | 1.000000 | 1.000000 |

另外，`info` 方法可以显示一些有用的信息：

```py
# Show info
dataframe.info()
```

```py
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1313 entries, 0 to 1312
Data columns (total 6 columns):
 #   Column    Non-Null Count  Dtype
---  ------    --------------  -----
 0   Name      1313 non-null   object
 1   PClass    1313 non-null   object
 2   Age       756 non-null    float64
 3   Sex       1313 non-null   object
 4   Survived  1313 non-null   int64
 5   SexCode   1313 non-null   int64
dtypes: float64(1), int64(2), object(3)
memory usage: 61.7+ KB
```

## 讨论

加载数据后，了解其结构和包含的信息是个好主意。理想情况下，我们可以直接查看完整的数据。但在大多数实际情况下，数据可能有数千到数百万行和列。因此，我们必须依靠提取样本来查看小部分数据片段，并计算数据的汇总统计信息。

在我们的解决方案中，我们使用了 *Titanic* 乘客的玩具数据集。使用 `head` 可以查看数据的前几行（默认为五行）。或者，我们可以使用 `tail` 查看最后几行。使用 `shape` 可以查看 DataFrame 包含多少行和列。使用 `describe` 可以查看任何数值列的基本描述统计信息。最后，`info` 显示了关于 DataFrame 的一些有用数据点，包括索引和列的数据类型、非空值和内存使用情况。

值得注意的是，汇总统计数据并不总是能完全反映事物的全部情况。例如，pandas 将 `Survived` 和 `SexCode` 列视为数值列，因为它们包含 1 和 0。然而，在这种情况下，这些数值实际上表示的是类别。例如，如果 `Survived` 等于 1，则表示该乘客在事故中生还。因此，某些汇总统计数据可能不适用，比如 `SexCode` 列的标准差（乘客性别的指示器）。

# 3.3 切片 DataFrame

## 问题

您需要选择特定的数据子集或 DataFrame 的切片。

## 解决方案

使用 `loc` 或 `iloc` 来选择一个或多个行或值：

```py
# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Select first row
dataframe.iloc[0]
```

```py
Name        Allen, Miss Elisabeth Walton
PClass                               1st
Age                                   29
Sex                               female
Survived                               1
SexCode                                1
Name: 0, dtype: object
```

我们可以使用`:`来定义我们想要的行切片，比如选择第二、第三和第四行：

```py
# Select three rows
dataframe.iloc[1:4]
```

|  | 名称 | PClass | 年龄 | 性别 | 幸存 | 性别代码 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 艾莉森小姐海伦·洛林 | 1st | 2.0 | 女性 | 0 | 1 |
| 2 | 艾莉森先生哈德森·乔舒亚·克莱顿 | 1st | 30.0 | 男性 | 0 | 0 |
| 3 | 艾莉森夫人哈德森 JC（贝西·沃尔多·丹尼尔斯） | 1st | 25.0 | 女性 | 0 | 1 |

我们甚至可以使用它来获取某个点之前的所有行，比如包括第四行在内的所有行：

```py
# Select four rows
dataframe.iloc[:4]
```

|  | 名称 | PClass | 年龄 | 性别 | 幸存 | 性别代码 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 艾伦小姐伊丽莎白·沃尔顿 | 1st | 29.0 | 女性 | 1 | 1 |
| 1 | 艾莉森小姐海伦·洛林 | 1st | 2.0 | 女性 | 0 | 1 |
| 2 | 艾莉森先生哈德森·乔舒亚·克莱顿 | 1st | 30.0 | 男性 | 0 | 0 |
| 3 | 艾莉森夫人哈德森 JC（贝西·沃尔多·丹尼尔斯） | 1st | 25.0 | 女性 | 0 | 1 |

数据框不需要数值索引。我们可以将数据框的索引设置为任何唯一的值，比如乘客的姓名，然后通过姓名选择行：

```py
# Set index
dataframe = dataframe.set_index(dataframe['Name'])

# Show row
dataframe.loc['Allen, Miss Elisabeth Walton']
```

```py
Name        Allen, Miss Elisabeth Walton
PClass                               1st
Age                                   29
Sex                               female
Survived                               1
SexCode                                1
Name: Allen, Miss Elisabeth Walton, dtype: object
```

## 讨论

pandas 数据框中的所有行都有唯一的索引值。默认情况下，这个索引是一个整数，表示数据框中的行位置；然而，并不一定要这样。数据框索引可以设置为唯一的字母数字字符串或客户号码。为了选择单独的行和行的切片，pandas 提供了两种方法：

+   `loc`在数据框的索引是标签时非常有用（例如，字符串）。

+   `iloc`的工作原理是查找数据框中的位置。例如，`iloc[0]`将返回数据框中的第一行，无论索引是整数还是标签。

熟悉`loc`和`iloc`在数据清洗过程中非常有用。

# 3.4 根据条件选择行

## 问题

您希望根据某些条件选择数据框的行。

## 解决方案

这在 pandas 中很容易实现。例如，如果我们想要选择所有*泰坦尼克号*上的女性：

```py
# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Show top two rows where column 'sex' is 'female'
dataframe[dataframe['Sex'] == 'female'].head(2)
```

|  | 名称 | PClass | 年龄 | 性别 | 幸存 | 性别代码 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 艾伦小姐伊丽莎白·沃尔顿 | 1st | 29.0 | 女性 | 1 | 1 |
| 1 | 艾莉森小姐海伦·洛林 | 1st | 2.0 | 女性 | 0 | 1 |

请花点时间看一下这个解决方案的格式。我们的条件语句是`dataframe['Sex'] == 'female'`；通过将其包装在`dataframe[]`中，我们告诉 pandas“选择数据框中`dataframe['Sex']`值为'female'的所有行”。这些条件导致了一个布尔值的 pandas 系列。

多条件也很容易。例如，这里我们选择所有女性且年龄在 65 岁或以上的乘客：

```py
# Filter rows
dataframe[(dataframe['Sex'] == 'female') & (dataframe['Age'] >= 65)]
```

|  | 名称 | PClass | 年龄 | 性别 | 幸存 | 性别代码 |
| --- | --- | --- | --- | --- | --- | --- |
| 73 | 克罗斯比夫人爱德华·吉福德（凯瑟琳·伊丽莎白... | 1st | 69.0 | 女性 | 1 | 1 |

## 讨论

在数据整理中，有条件地选择和过滤数据是最常见的任务之一。您很少需要源数据的所有原始数据；相反，您只对某些子集感兴趣。例如，您可能只对特定州的商店或特定年龄段的患者记录感兴趣。

# 3.5 排序数值

## 问题

您需要按列中的值对数据框进行排序。

## 解决方案

使用 pandas 的 `sort_values` 函数：

```py
# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Sort the dataframe by age, show two rows
dataframe.sort_values(by=["Age"]).head(2)
```

|  | 姓名 | PClass | 年龄 | 性别 | 幸存 | 性别代码 |
| --- | --- | --- | --- | --- | --- | --- |
| 763 | 迪恩，伊丽莎白·格莱迪斯（米尔文娜）小姐 | 3rd | 0.17 | 女性 | 1 | 1 |
| 751 | 丹博姆，吉尔伯特·西格瓦德·埃马纽尔大师 | 3rd | 0.33 | 男性 | 0 | 0 |

## 讨论

在数据分析和探索过程中，按照特定列或一组列对 DataFrame 进行排序通常非常有用。`sort_values` 的 `by` 参数接受一个列名列表，按列表中列名的顺序对 DataFrame 进行排序。

默认情况下，`ascending` 参数设置为 `True`，因此它会将值从最低到最高排序。如果我们想要最年长的乘客而不是最年轻的，我们可以将其设置为 `False`。

# 3.6 替换值

## 问题

您需要在 DataFrame 中替换值。

## 解决方案

pandas 的 `replace` 方法是查找和替换值的简便方法。例如，我们可以将 `Sex` 列中的任何 `"female"` 实例替换为 `"Woman"`：

```py
# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Replace values, show two rows
dataframe['Sex'].replace("female", "Woman").head(2)
```

```py
0    Woman
1    Woman
Name: Sex, dtype: object
```

我们还可以同时替换多个值：

```py
# Replace "female" and "male" with "Woman" and "Man"
dataframe['Sex'].replace(["female", "male"], ["Woman", "Man"]).head(5)
```

```py
0    Woman
1    Woman
2      Man
3    Woman
4      Man
Name: Sex, dtype: object
```

我们还可以通过指定整个 DataFrame 而不是单个列来查找并替换 `DataFrame` 对象中的所有值：

```py
# Replace values, show two rows
dataframe.replace(1, "One").head(2)
```

|  | 姓名 | PClass | 年龄 | 性别 | 幸存 | 性别代码 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 艾伦，伊丽莎白·沃尔顿小姐 | 1st | 29 | 女性 | One | One |
| 1 | 艾莉森，洛林小姐海伦 | 1st | 2 | 女性 | 0 | One |

`replace` 也接受正则表达式：

```py
# Replace values, show two rows
dataframe.replace(r"1st", "First", regex=True).head(2)
```

|  | 姓名 | PClass | 年龄 | 性别 | 幸存 | 性别代码 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 艾伦，伊丽莎白·沃尔顿小姐 | First | 29.0 | 女性 | 1 | 1 |
| 1 | 艾莉森，洛林小姐海伦 | First | 2.0 | 女性 | 0 | 1 |

## 讨论

`replace` 是我们用来替换值的工具。它简单易用，同时能够接受正则表达式。

# 3.7 重命名列

## 问题

您想要在 pandas DataFrame 中重命名列。

## 解决方案

使用 `rename` 方法重命名列：

```py
# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Rename column, show two rows
dataframe.rename(columns={'PClass': 'Passenger Class'}).head(2)
```

|  | 姓名 | 乘客等级 | 年龄 | 性别 | 幸存 | 性别代码 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 艾伦，伊丽莎白·沃尔顿小姐 | 1st | 29.0 | 女性 | 1 | 1 |
| 1 | 艾莉森，洛林小姐海伦 | 1st | 2.0 | 女性 | 0 | 1 |

注意，`rename` 方法可以接受一个字典作为参数。我们可以使用字典一次性更改多个列名：

```py
# Rename columns, show two rows
dataframe.rename(columns={'PClass': 'Passenger Class', 'Sex': 'Gender'}).head(2)
```

|  | 姓名 | 乘客等级 | 年龄 | 性别 | 幸存 | 性别代码 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 艾伦，伊丽莎白·沃尔顿小姐 | 1st | 29.0 | 女性 | 1 | 1 |
| 1 | 艾莉森，洛林小姐海伦 | 1st | 2.0 | 女性 | 0 | 1 |

## 讨论

使用将字典作为参数传递给 `columns` 参数的 `rename` 是我首选的重命名列的方法，因为它适用于任意数量的列。 如果我们想一次重命名所有列，这段有用的代码片段会创建一个以旧列名作为键、空字符串作为值的字典：

```py
# Load library
import collections

# Create dictionary
column_names = collections.defaultdict(str)

# Create keys
for name in dataframe.columns:
    column_names[name]

# Show dictionary
column_names
```

```py
defaultdict(str,
            {'Age': '',
             'Name': '',
             'PClass': '',
             'Sex': '',
             'SexCode': '',
             'Survived': ''})
```

# 3.8 寻找最小值、最大值、总和、平均值和计数

## 问题

您想要找到数值列的最小值、最大值、总和、平均值或计数。

## 解决方案

pandas 提供了一些内置方法，用于常用的描述统计，如 `min`、`max`、`mean`、`sum` 和 `count`：

```py
# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Calculate statistics
print('Maximum:', dataframe['Age'].max())
print('Minimum:', dataframe['Age'].min())
print('Mean:', dataframe['Age'].mean())
print('Sum:', dataframe['Age'].sum())
print('Count:', dataframe['Age'].count())
```

```py
Maximum: 71.0
Minimum: 0.17
Mean: 30.397989417989415
Sum: 22980.879999999997
Count: 756
```

## 讨论

除了解决方案中使用的统计数据外，pandas 还提供了方差(`var`)、标准差(`std`)、峰度(`kurt`)、偏度(`skew`)、均值标准误(`sem`)、众数(`mode`)、中位数(`median`)、值计数以及其他几种统计数据。

此外，我们还可以将这些方法应用于整个 DataFrame：

```py
# Show counts
dataframe.count()
```

```py
Name        1313
PClass      1313
Age          756
Sex         1313
Survived    1313
SexCode     1313
dtype: int64
```

# 3.9 寻找唯一值

## 问题

您想选择某一列中的所有唯一值。

## 解决方案

使用 `unique` 查看列中所有唯一值的数组：

```py
# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Select unique values
dataframe['Sex'].unique()
```

```py
array(['female', 'male'], dtype=object)
```

或者，`value_counts` 将显示所有唯一值及其出现次数：

```py
# Show counts
dataframe['Sex'].value_counts()
```

```py
male      851
female    462
Name: Sex, dtype: int64
```

## 讨论

`unique` 和 `value_counts` 对于操作和探索分类列非常有用。 在分类列中，通常需要在数据整理阶段处理类别。 例如，在 *Titanic* 数据集中，`PClass` 是指乘客票的类别。 在 *Titanic* 上有三个等级; 但是，如果我们使用 `value_counts`，我们会发现一个问题：

```py
# Show counts
dataframe['PClass'].value_counts()
```

```py
3rd    711
1st    322
2nd    279
*        1
Name: PClass, dtype: int64
```

尽管几乎所有乘客都属于预期的三个类别之一，但是有一个乘客的类别是 `*`。 在处理这类问题时有多种策略，我们将在第五章中进行讨论，但现在只需意识到，在分类数据中，“额外”类别是常见的，不应忽略。

最后，如果我们只想计算唯一值的数量，我们可以使用 `nunique`：

```py
# Show number of unique values
dataframe['PClass'].nunique()
```

```py
4
```

# 3.10 处理缺失值

## 问题

您想要选择 DataFrame 中的缺失值。

## 解决方案

`isnull` 和 `notnull` 返回布尔值，指示值是否缺失：

```py
# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

## Select missing values, show two rows
dataframe[dataframe['Age'].isnull()].head(2)
```

|  | 名称 | 舱位 | 年龄 | 性别 | 幸存 | 性别代码 |
| --- | --- | --- | --- | --- | --- | --- |
| 12 | Aubert, Mrs Leontine Pauline | 1st | NaN | female | 1 | 1 |
| 13 | Barkworth, Mr Algernon H | 1st | NaN | male | 1 | 0 |

## 讨论

缺失值是数据整理中普遍存在的问题，然而许多人低估了处理缺失数据的难度。 pandas 使用 NumPy 的 `NaN`（非数值）值表示缺失值，但重要的是要注意，pandas 中没有完全本地实现 `NaN`。 例如，如果我们想用包含 `male` 的所有字符串替换缺失值，我们会得到一个错误：

```py
# Attempt to replace values with NaN
dataframe['Sex'] = dataframe['Sex'].replace('male', NaN)
```

```py
---------------------------------------------------------------------------

NameError                                 Traceback (most recent call last)

<ipython-input-7-5682d714f87d> in <module>()
      1 # Attempt to replace values with NaN
----> 2 dataframe['Sex'] = dataframe['Sex'].replace('male', NaN)

NameError: name 'NaN' is not defined
---------------------------------------------------------------------------

```

要完全使用 `NaN` 的功能，我们首先需要导入 NumPy 库：

```py
# Load library
import numpy as np

# Replace values with NaN
dataframe['Sex'] = dataframe['Sex'].replace('male', np.nan)
```

很多时候，数据集使用特定的值来表示缺失的观察值，例如 `NONE`、`-999` 或 `..`。pandas 的 `read_csv` 函数包括一个参数，允许我们指定用于指示缺失值的值：

```py
# Load data, set missing values
dataframe = pd.read_csv(url, na_values=[np.nan, 'NONE', -999])
```

我们还可以使用 pandas 的 `fillna` 函数来填充列的缺失值。在这里，我们使用 `isna` 函数显示 `Age` 为空的位置，然后用乘客的平均年龄填充这些值。

```py
# Get a single null row
null_entry = dataframe[dataframe["Age"].isna()].head(1)

print(null_entry)
```

|  | 姓名 | 舱位 | 年龄 | 性别 | 幸存 | 性别编码 |
| --- | --- | --- | --- | --- | --- | --- |
| 12 | 奥贝特，利昂汀·波琳娜夫人 | 1st | NaN | 女性 | 1 | 1 |

```py
# Fill all null values with the mean age of passengers
null_entry.fillna(dataframe["Age"].mean())
```

|  | 姓名 | 舱位 | 年龄 | 性别 | 幸存 | 性别编码 |
| --- | --- | --- | --- | --- | --- | --- |
| 12 | 奥贝特，利昂汀·波琳娜夫人 | 1st | 30.397989 | 女性 | 1 | 1 |

# 3.11 删除列

## 问题

您想要从您的 DataFrame 中删除一列。

## 解决方案

删除列的最佳方法是使用带有参数 `axis=1`（即列轴）的 `drop`：

```py
# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Delete column
dataframe.drop('Age', axis=1).head(2)
```

|  | 姓名 | 舱位 | 性别 | 幸存 | 性别编码 |
| --- | --- | --- | --- | --- | --- |
| 0 | 艾伦，伊丽莎白·沃尔顿小姐 | 1st | 女性 | 1 | 1 |
| 1 | 艾莉森，露萍小姐 | 1st | 女性 | 0 | 1 |

你也可以使用列名的列表作为删除多列的主要参数：

```py
# Drop columns
dataframe.drop(['Age', 'Sex'], axis=1).head(2)
```

|  | 姓名 | 舱位 | 幸存 | 性别编码 |
| --- | --- | --- | --- | --- |
| 0 | 艾伦，伊丽莎白·沃尔顿小姐 | 1st | 1 | 1 |
| 1 | 艾莉森，露萍小姐 | 1st | 0 | 1 |

如果列没有名称（有时可能会发生），您可以使用 `dataframe.columns` 按其列索引删除它：

```py
# Drop column
dataframe.drop(dataframe.columns[1], axis=1).head(2)
```

|  | 姓名 | 年龄 | 性别 | 幸存 | 性别编码 |
| --- | --- | --- | --- | --- | --- |
| 0 | 艾伦，伊丽莎白·沃尔顿小姐 | 29.0 | 女性 | 1 | 1 |
| 1 | 艾莉森，露萍小姐 | 2.0 | 女性 | 0 | 1 |

## 讨论

`drop` 是删除列的成语方法。另一种方法是 `del dataframe['Age']`，大多数情况下可以工作，但不建议使用，因为它在 pandas 中的调用方式（其细节超出本书的范围）。

我建议您避免使用 pandas 的 `inplace=True` 参数。许多 pandas 方法包括一个 `inplace` 参数，当设置为 `True` 时，直接编辑 DataFrame。这可能会导致在更复杂的数据处理管道中出现问题，因为我们将 DataFrame 视为可变对象（从技术上讲确实如此）。我建议将 DataFrame 视为不可变对象。例如：

```py
# Create a new DataFrame
dataframe_name_dropped = dataframe.drop(dataframe.columns[0], axis=1)
```

在这个例子中，我们没有改变 DataFrame `dataframe`，而是创建了一个新的 DataFrame，称为 `dataframe_name_dropped`，它是 `dataframe` 的修改版本。如果您将 DataFrame 视为不可变对象，那么您将会在将来避免很多麻烦。

# 3.12 删除行

## 问题

您想要从 DataFrame 中删除一行或多行。

## 解决方案

使用布尔条件创建一个新的 DataFrame，排除你想要删除的行：

```py
# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Delete rows, show first three rows of output
dataframe[dataframe['Sex'] != 'male'].head(3)
```

|  | 姓名 | 舱位 | 年龄 | 性别 | 幸存 | 性别编码 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Allen, Miss Elisabeth Walton | 1st | 29.0 | 女性 | 1 | 1 |
| 1 | Allison, Miss Helen Loraine | 1st | 2.0 | 女性 | 0 | 1 |
| 3 | Allison, Mrs Hudson JC (Bessie Waldo Daniels) | 1st | 25.00 | 女性 | 0 | 1 |

## 讨论

技术上你可以使用`drop`方法（例如，`dataframe.drop([0, 1], axis=0)`来删除前两行），但更实用的方法是简单地将布尔条件包装在`dataframe[]`中。这使我们能够利用条件语句的威力来删除单行或（更有可能）多行。

我们可以使用布尔条件轻松删除单行，通过匹配唯一值：

```py
# Delete row, show first two rows of output
dataframe[dataframe['Name'] != 'Allison, Miss Helen Loraine'].head(2)
```

|  | 姓名 | 票类 | 年龄 | 性别 | 生存 | 性别编码 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Allen, Miss Elisabeth Walton | 1st | 29.0 | 女性 | 1 | 1 |
| 2 | Allison, Mr Hudson Joshua Creighton | 1st | 30.0 | 男性 | 0 | 0 |

我们甚至可以通过指定行索引来使用它删除单行：

```py
# Delete row, show first two rows of output
dataframe[dataframe.index != 0].head(2)
```

|  | 姓名 | 票类 | 年龄 | 性别 | 生存 | 性别编码 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Allison, Miss Helen Loraine | 1st | 2.0 | 女性 | 0 | 1 |
| 2 | Allison, Mr Hudson Joshua Creighton | 1st | 30.0 | 男性 | 0 | 0 |

# 3.13 删除重复行

## 问题

您想从 DataFrame 中删除重复的行。

## 解决方案

使用`drop_duplicates`，但要注意参数：

```py
# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Drop duplicates, show first two rows of output
dataframe.drop_duplicates().head(2)
```

|  | 姓名 | 票类 | 年龄 | 性别 | 生存 | 性别编码 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Allen, Miss Elisabeth Walton | 1st | 29.0 | 女性 | 1 | 1 |
| 1 | Allison, Miss Helen Loraine | 1st | 2.0 | 女性 | 0 | 1 |

## 讨论

一个敏锐的读者会注意到，解决方案实际上并没有删除任何行：

```py
# Show number of rows
print("Number Of Rows In The Original DataFrame:", len(dataframe))
print("Number Of Rows After Deduping:", len(dataframe.drop_duplicates()))
```

```py
Number Of Rows In The Original DataFrame: 1313
Number Of Rows After Deduping: 1313
```

这是因为`drop_duplicates`默认仅删除所有列完全匹配的行。因为我们 DataFrame 中的每一行都是唯一的，所以不会被删除。然而，通常我们只想考虑部分列来检查重复行。我们可以使用`subset`参数来实现这一点：

```py
# Drop duplicates
dataframe.drop_duplicates(subset=['Sex'])
```

|  | 姓名 | 票类 | 年龄 | 性别 | 生存 | 性别编码 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Allen, Miss Elisabeth Walton | 1st | 29.0 | 女性 | 1 | 1 |
| 2 | Allison, Mr Hudson Joshua Creighton | 1st | 30.0 | 男性 | 0 | 0 |

仔细观察上述输出：我们告诉`drop_duplicates`仅考虑具有相同`Sex`值的任意两行为重复行，并将其删除。现在我们只剩下两行的 DataFrame：一个女性和一个男性。你可能会问为什么`drop_duplicates`决定保留这两行而不是两行不同的行。答案是`drop_duplicates`默认保留重复行的第一次出现并丢弃其余的。我们可以使用`keep`参数来控制这种行为：

```py
# Drop duplicates
dataframe.drop_duplicates(subset=['Sex'], keep='last')
```

|  | 姓名 | 票类 | 年龄 | 性别 | 生存 | 性别编码 |
| --- | --- | --- | --- | --- | --- | --- |
| 1307 | Zabour, Miss Tamini | 3rd | NaN | 女性 | 0 | 1 |
| 1312 | Zimmerman, Leo | 3rd | 29.0 | 男性 | 0 | 0 |

一个相关的方法是`duplicated`，它返回一个布尔系列，指示行是否是重复的。如果您不想简单地删除重复项，这是一个不错的选择：

```py
dataframe.duplicated()
```

```py
0       False
1       False
2       False
3       False
4       False
        ...
1308    False
1309    False
1310    False
1311    False
1312    False
Length: 1313, dtype: bool
```

# 3.14 按值分组行

## 问题

您希望根据某些共享值对单独的行进行分组。

## 解决方案

`groupby`是 pandas 中最强大的特性之一：

```py
# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Group rows by the values of the column 'Sex', calculate mean # of each group
dataframe.groupby('Sex').mean(numeric_only=True)
```

| 性别 | 年龄 | 幸存者 | 性别代码 |
| --- | --- | --- | --- |
| 女性 | 29.396424 | 0.666667 | 1.0 |
| 男性 | 31.014338 | 0.166863 | 0.0 |

## 讨论

`groupby`是数据处理真正开始成形的地方。DataFrame 中每行代表一个人或事件是非常普遍的，我们想根据某些标准对它们进行分组，然后计算统计量。例如，您可以想象一个 DataFrame，其中每行是全国餐厅连锁店的单笔销售，我们想要每个餐厅的总销售额。我们可以通过按独立餐厅分组行，然后计算每组的总和来实现这一点。

新用户对`groupby`经常写出这样的一行，然后对返回的内容感到困惑：

```py
# Group rows
dataframe.groupby('Sex')
```

```py
<pandas.core.groupby.DataFrameGroupBy object at 0x10efacf28>
```

为什么它没有返回更有用的东西？原因是`groupby`需要与我们想应用于每个组的某些操作配对，比如计算聚合统计（例如均值、中位数、总和）。在讨论分组时，我们经常使用简写说“按性别分组”，但这是不完整的。为了使分组有用，我们需要按某些标准分组，然后对每个组应用函数：

```py
# Group rows, count rows
dataframe.groupby('Survived')['Name'].count()
```

```py
Survived
0    863
1    450
Name: Name, dtype: int64
```

注意在`groupby`后添加了`Name`？这是因为特定的摘要统计只对某些类型的数据有意义。例如，按性别计算平均年龄是有意义的，但按性别计算总年龄不是。在这种情况下，我们将数据分组为幸存或未幸存，然后计算每个组中的名称数量（即乘客数）。

我们还可以按第一列分组，然后按第二列对该分组进行分组：

```py
# Group rows, calculate mean
dataframe.groupby(['Sex','Survived'])['Age'].mean()
```

```py
Sex     Survived
female  0           24.901408
        1           30.867143
male    0           32.320780
        1           25.951875
Name: Age, dtype: float64
```

# 3.15 按时间分组行

## 问题

您需要按时间段对单独的行进行分组。

## 解决方案

使用`resample`按时间段分组行：

```py
# Load libraries
import pandas as pd
import numpy as np

# Create date range
time_index = pd.date_range('06/06/2017', periods=100000, freq='30S')

# Create DataFrame
dataframe = pd.DataFrame(index=time_index)

# Create column of random values
dataframe['Sale_Amount'] = np.random.randint(1, 10, 100000)

# Group rows by week, calculate sum per week
dataframe.resample('W').sum()
```

|  | 销售金额 |
| --- | --- |
| 2017-06-11 | 86423 |
| 2017-06-18 | 101045 |
| 2017-06-25 | 100867 |
| 2017-07-02 | 100894 |
| 2017-07-09 | 100438 |
| 2017-07-16 | 10297 |

## 讨论

我们的标准*Titanic*数据集不包含日期时间列，因此对于这个示例，我们生成了一个简单的 DataFrame，其中每行代表一次单独的销售。对于每个销售，我们知道其日期时间和金额（这些数据并不真实，因为销售间隔正好为 30 秒，金额为确切的美元数，但为了简单起见，我们假装是这样的）。

原始数据如下所示：

```py
# Show three rows
dataframe.head(3)
```

|  | 销售金额 |
| --- | --- |
| 2017-06-06 00:00:00 | 7 |
| 2017-06-06 00:00:30 | 2 |
| 2017-06-06 00:01:00 | 7 |

注意每次销售的日期和时间是 DataFrame 的索引；这是因为`resample`需要索引是类似日期时间的值。

使用`resample`，我们可以按照各种时间段（偏移）对行进行分组，然后可以在每个时间组上计算统计信息：

```py
# Group by two weeks, calculate mean
dataframe.resample('2W').mean()
```

|  | Sale_Amount |
| --- | --- |
| 2017-06-11 | 5.001331 |
| 2017-06-25 | 5.007738 |
| 2017-07-09 | 4.993353 |
| 2017-07-23 | 4.950481 |

```py
# Group by month, count rows
dataframe.resample('M').count()
```

|  | Sale_Amount |
| --- | --- | --- |
| 2017-06-30 | 72000 |
| 2017-07-31 | 28000 |

您可能注意到，在这两个输出中，日期时间索引是日期，即使我们按周和月进行分组。原因是默认情况下，`resample`返回时间组的右“边缘”标签（最后一个标签）。我们可以使用`label`参数控制此行为：

```py
# Group by month, count rows
dataframe.resample('M', label='left').count()
```

|  | Sale_Amount |
| --- | --- |
| 2017-05-31 | 72000 |
| 2017-06-30 | 28000 |

## 另请参阅

+   [pandas 时间偏移别名列表](https://oreil.ly/BURbR)

# 3.16 聚合操作和统计

## 问题

您需要对数据框中的每列（或一组列）进行聚合操作。

## 解决方案

使用 pandas 的`agg`方法。在这里，我们可以轻松地获得每列的最小值：

```py
# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Get the minimum of every column
dataframe.agg("min")
```

```py
Name        Abbing, Mr Anthony
PClass                       *
Age                       0.17
Sex                     female
Survived                     0
SexCode                      0
dtype: object
```

有时，我们希望将特定函数应用于特定列集：

```py
# Mean Age, min and max SexCode
dataframe.agg({"Age":["mean"], "SexCode":["min", "max"]})
```

|  | Age | SexCode |
| --- | --- | --- |
| mean | 30.397989 | NaN |
| min | NaN | 0.0 |
| max | NaN | 1.0 |

我们还可以将聚合函数应用于组，以获取更具体的描述性统计信息：

```py
# Number of people who survived and didn't survive in each class
dataframe.groupby(
    ["PClass","Survived"]).agg({"Survived":["count"]}
  ).reset_index()
```

| PClass | Survived | Count |
| --- | --- | --- |
| 0 | * | 0 | 1 |
| 1 | 1st | 0 | 129 |
| 2 | 1st | 1 | 193 |
| 3 | 2nd | 0 | 160 |
| 4 | 2nd | 1 | 119 |
| 5 | 3rd | 0 | 573 |
| 6 | 3rd | 1 | 138 |

## 讨论

在探索性数据分析中，聚合函数特别有用，可用于了解数据的不同子群体和变量之间的关系。通过对数据进行分组并应用聚合统计，您可以查看数据中的模式，这些模式在机器学习或特征工程过程中可能会很有用。虽然视觉图表也很有帮助，但有这样具体的描述性统计数据作为参考，有助于更好地理解数据。

## 另请参阅

+   [pandas agg 文档](https://oreil.ly/5xing)

# 3.17 遍历列

## 问题

您希望遍历列中的每个元素并应用某些操作。

## 解决方案

您可以像对待 Python 中的任何其他序列一样处理 pandas 列，并使用标准 Python 语法对其进行循环：

```py
# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Print first two names uppercased
for name in dataframe['Name'][0:2]:
    print(name.upper())
```

```py
ALLEN, MISS ELISABETH WALTON
ALLISON, MISS HELEN LORAINE
```

## 讨论

除了循环（通常称为`for`循环）之外，我们还可以使用列表推导：

```py
# Show first two names uppercased
[name.upper() for name in dataframe['Name'][0:2]]
```

```py
['ALLEN, MISS ELISABETH WALTON', 'ALLISON, MISS HELEN LORAINE']
```

尽管有诱惑使用`for`循环，更符合 Python 风格的解决方案应该使用 pandas 的`apply`方法，详见配方 3.18。

# 3.18 在列中的所有元素上应用函数

## 问题

您希望在一列的所有元素上应用某些函数。

## 解决方案

使用`apply`在列的每个元素上应用内置或自定义函数：

```py
# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Create function
def uppercase(x):
    return x.upper()

# Apply function, show two rows
dataframe['Name'].apply(uppercase)[0:2]
```

```py
0    ALLEN, MISS ELISABETH WALTON
1    ALLISON, MISS HELEN LORAINE
Name: Name, dtype: object
```

## 讨论

`apply`是进行数据清理和整理的好方法。通常会编写一个函数来执行一些有用的操作（将名字分开，将字符串转换为浮点数等），然后将该函数映射到列中的每个元素。

# 3.19 对组应用函数

## 问题

您已经使用`groupby`对行进行分组，并希望对每个组应用函数。

## 解决方案

结合`groupby`和`apply`：

```py
# Load library
import pandas as pd

# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/titanic.csv'

# Load data
dataframe = pd.read_csv(url)

# Group rows, apply function to groups
dataframe.groupby('Sex').apply(lambda x: x.count())
```

| 性别 | 姓名 | 舱位 | 年龄 | 性别 | 幸存 | 性别编码 |
| --- | --- | --- | --- | --- | --- | --- |
| 女性 | 462 | 462 | 288 | 462 | 462 | 462 |
| 男性 | 851 | 851 | 468 | 851 | 851 | 851 |

## 讨论

在 Recipe 3.18 中提到了`apply`。当你想对分组应用函数时，`apply`特别有用。通过结合`groupby`和`apply`，我们可以计算自定义统计信息或将任何函数分别应用于每个组。

# 3.20 合并数据框

## 问题

您想要将两个数据框连接在一起。

## 解决方案

使用`concat`和`axis=0`沿行轴进行连接：

```py
# Load library
import pandas as pd

# Create DataFrame
data_a = {'id': ['1', '2', '3'],
          'first': ['Alex', 'Amy', 'Allen'],
          'last': ['Anderson', 'Ackerman', 'Ali']}
dataframe_a = pd.DataFrame(data_a, columns = ['id', 'first', 'last'])

# Create DataFrame
data_b = {'id': ['4', '5', '6'],
          'first': ['Billy', 'Brian', 'Bran'],
          'last': ['Bonder', 'Black', 'Balwner']}
dataframe_b = pd.DataFrame(data_b, columns = ['id', 'first', 'last'])

# Concatenate DataFrames by rows
pd.concat([dataframe_a, dataframe_b], axis=0)
```

|  | id | first | last |
| --- | --- | --- | --- |
| 0 | 1 | Alex | Anderson |
| 1 | 2 | Amy | Ackerman |
| 2 | 3 | Allen | Ali |
| 0 | 4 | Billy | Bonder |
| 1 | 5 | Brian | Black |
| 2 | 6 | Bran | Balwner |

您可以使用`axis=1`沿列轴进行连接：

```py
# Concatenate DataFrames by columns
pd.concat([dataframe_a, dataframe_b], axis=1)
```

|  | id | first | last | id | first | last |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | Alex | Anderson | 4 | Billy | Bonder |
| 1 | 2 | Amy | Ackerman | 5 | Brian | Black |
| 2 | 3 | Allen | Ali | 6 | Bran | Balwner |

## 讨论

合并通常是一个你在计算机科学和编程领域听得较多的词汇，所以如果你以前没听说过，别担心。*concatenate*的非正式定义是将两个对象粘合在一起。在解决方案中，我们使用`axis`参数将两个小数据框粘合在一起，以指示我们是否想要将两个数据框叠加在一起还是并排放置它们。

# 3.21 合并数据框

## 问题

您想要合并两个数据框。

## 解决方案

要进行内连接，使用`merge`并使用`on`参数指定要合并的列：

```py
# Load library
import pandas as pd

# Create DataFrame
employee_data = {'employee_id': ['1', '2', '3', '4'],
                 'name': ['Amy Jones', 'Allen Keys', 'Alice Bees',
                 'Tim Horton']}
dataframe_employees = pd.DataFrame(employee_data, columns = ['employee_id',
                                                              'name'])

# Create DataFrame
sales_data = {'employee_id': ['3', '4', '5', '6'],
              'total_sales': [23456, 2512, 2345, 1455]}
dataframe_sales = pd.DataFrame(sales_data, columns = ['employee_id',
                                                      'total_sales'])

# Merge DataFrames
pd.merge(dataframe_employees, dataframe_sales, on='employee_id')
```

|  | 员工编号 | 姓名 | 总销售额 |
| --- | --- | --- | --- |
| 0 | 3 | Alice Bees | 23456 |
| 1 | 4 | Tim Horton | 2512 |

`merge`默认为内连接。如果我们想进行外连接，可以使用`how`参数来指定：

```py
# Merge DataFrames
pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='outer')
```

|  | 员工编号 | 姓名 | 总销售额 |
| --- | --- | --- | --- |
| 0 | 1 | Amy Jones | NaN |
| 1 | 2 | Allen Keys | NaN |
| 2 | 3 | Alice Bees | 23456.0 |
| 3 | 4 | Tim Horton | 2512.0 |
| 4 | 5 | NaN | 2345.0 |
| 5 | 6 | NaN | 1455.0 |

同一个参数可以用来指定左连接和右连接：

```py
# Merge DataFrames
pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='left')
```

|  | 员工编号 | 姓名 | 总销售额 |
| --- | --- | --- | --- |
| 0 | 1 | Amy Jones | NaN |
| 1 | 2 | Allen Keys | NaN |
| 2 | 3 | Alice Bees | 23456.0 |
| 3 | 4 | Tim Horton | 2512.0 |

我们还可以在每个数据框中指定要合并的列名：

```py
# Merge DataFrames
pd.merge(dataframe_employees,
         dataframe_sales,
         left_on='employee_id',
         right_on='employee_id')
```

|  | 员工编号 | 姓名 | 总销售额 |
| --- | --- | --- | --- |
| 0 | 3 | Alice Bees | 23456 |
| 1 | 4 | Tim Horton | 2512 |

如果我们希望不是在两个列上进行合并，而是在每个 DataFrame 的索引上进行合并，我们可以将`left_on`和`right_on`参数替换为`left_index=True`和`right_index=True`。

## 讨论

我们需要使用的数据通常很复杂；它不总是一次性出现。相反，在现实世界中，我们通常面对来自多个数据库查询或文件的不同数据集。为了将所有数据汇总到一个地方，我们可以将每个数据查询或数据文件作为单独的 DataFrame 加载到 pandas 中，然后将它们合并成一个单一的 DataFrame。

这个过程对于使用过 SQL 的人可能会很熟悉，SQL 是一种用于执行合并操作（称为*连接*）的流行语言。虽然 pandas 使用的确切参数会有所不同，但它们遵循其他软件语言和工具使用的相同一般模式。

任何`merge`操作都有三个方面需要指定。首先，我们必须指定要合并的两个 DataFrame。在解决方案中，我们将它们命名为`dataframe_employees`和`dataframe_sales`。其次，我们必须指定要合并的列名（们）-即，两个 DataFrame 之间共享值的列名。例如，在我们的解决方案中，两个 DataFrame 都有一个名为`employee_id`的列。为了合并这两个 DataFrame，我们将匹配每个 DataFrame 的`employee_id`列中的值。如果这两个列使用相同的名称，我们可以使用`on`参数。但是，如果它们有不同的名称，我们可以使用`left_on`和`right_on`。

什么是左 DataFrame 和右 DataFrame？左 DataFrame 是我们在`merge`中指定的第一个 DataFrame，右 DataFrame 是第二个。这种语言在我们将需要的下一组参数中再次出现。

最后一个方面，也是一些人难以掌握的最难的方面，是我们想要执行的合并操作的类型。这由`how`参数指定。`merge`支持四种主要类型的连接操作：

内部

仅返回在两个 DataFrame 中都匹配的行（例如，返回任何在`dataframe_employees`和`dataframe_sales`中的`employee_id`值都出现的行）。

外部

返回两个 DataFrame 中的所有行。如果一行存在于一个 DataFrame 中但不在另一个 DataFrame 中，则填充缺失值 NaN（例如，在`dataframe_employee`和`dataframe_sales`中返回所有行）。

左

返回左 DataFrame 中的所有行，但只返回与左 DataFrame 匹配的右 DataFrame 中的行。对于缺失值填充`NaN`（例如，从`dataframe_employees`返回所有行，但只返回`dataframe_sales`中具有出现在`dataframe_employees`中的`employee_id`值的行）。

右

返回右侧数据框的所有行，但仅返回左侧数据框中与右侧数据框匹配的行。对于缺失的值填充`NaN`（例如，返回`dataframe_sales`的所有行，但仅返回`dataframe_employees`中具有出现在`dataframe_sales`中的`employee_id`值的行）。

如果你还没有完全理解，我鼓励你在你的代码中尝试调整`how`参数，看看它如何影响`merge`返回的结果。

## 参见

+   [SQL 连接操作的可视化解释](https://oreil.ly/J1A4u)

+   [pandas 文档：合并、连接、串联和比较](https://oreil.ly/eNalU)
