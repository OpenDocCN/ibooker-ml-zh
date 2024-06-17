# 第七章：处理日期和时间

# 7.0 引言

处理日期和时间（datetime），例如特定销售的时间或公共卫生统计的日期，在机器学习预处理中经常遇到。*纵向数据*（或*时间序列数据*）是重复收集同一变量的数据，随时间点变化。在本章中，我们将构建处理时间序列数据的策略工具箱，包括处理时区和创建滞后时间特征。具体来说，我们将关注 pandas 库中的时间序列工具，该工具集集中了许多其他通用库（如 `datetime`）的功能。

# 7.1 字符串转换为日期

## 问题

给定一个表示日期和时间的字符串向量，你想要将它们转换为时间序列数据。

## 解决方案

使用 pandas 的 `to_datetime`，并指定日期和/或时间的格式在 `format` 参数中：

```py
# Load libraries
import numpy as np
import pandas as pd

# Create strings
date_strings = np.array(['03-04-2005 11:35 PM',
                         '23-05-2010 12:01 AM',
                         '04-09-2009 09:09 PM'])

# Convert to datetimes
[pd.to_datetime(date, format='%d-%m-%Y %I:%M %p') for date in date_strings]
```

```py
[Timestamp('2005-04-03 23:35:00'),
 Timestamp('2010-05-23 00:01:00'),
 Timestamp('2009-09-04 21:09:00')]
```

我们也可能想要向 `errors` 参数添加一个参数来处理问题：

```py
# Convert to datetimes
[pd.to_datetime(date, format="%d-%m-%Y %I:%M %p", errors="coerce")
for date in date_strings]
```

```py
[Timestamp('2005-04-03 23:35:00'),
 Timestamp('2010-05-23 00:01:00'),
 Timestamp('2009-09-04 21:09:00')]
```

如果 `errors="coerce"`，则任何发生的问题不会引发错误（默认行为），而是将导致错误的值设置为 `NaT`（缺失值）。这允许你通过将其填充为 null 值来处理异常值，而不是为数据中的每个记录进行故障排除。

## 讨论

当日期和时间以字符串形式提供时，我们需要将它们转换为 Python 能够理解的数据类型。虽然有许多用于将字符串转换为日期时间的 Python 工具，但在其他示例中使用 pandas 后，我们可以使用 `to_datetime` 进行转换。使用字符串表示日期和时间的一个障碍是，字符串的格式在数据源之间可能有很大的变化。例如，一个日期向量可能将 2015 年 3 月 23 日表示为“03-23-15”，而另一个可能使用“3|23|2015”。我们可以使用 `format` 参数来指定字符串的确切格式。以下是一些常见的日期和时间格式代码：

| 代码 | 描述 | 示例 |
| --- | --- | --- |
| `%Y` | 完整年份 | `2001` |
| `%m` | 带零填充的月份 | `04` |
| `%d` | 带零填充的日期 | `09` |
| `%I` | 带零填充的小时（12 小时制） | `02` |
| `%p` | 上午或下午 | `AM` |
| `%M` | 带零填充的分钟数 | `05` |
| `%S` | 带零填充的秒数 | `09` |

## 参见

+   [Python strftime 速查表（Python 字符串时间代码的完整列表）](https://oreil.ly/4-tN6)

# 7.2 处理时区

## 问题

你有时间序列数据，想要添加或更改时区信息。

## 解决方案

除非指定，否则 pandas 对象没有时区。我们可以在创建时使用 `tz` 添加时区：

```py
# Load library
import pandas as pd

# Create datetime
pd.Timestamp('2017-05-01 06:00:00', tz='Europe/London')
```

```py
Timestamp('2017-05-01 06:00:00+0100', tz='Europe/London')
```

我们可以使用 `tz_localize` 为先前创建的日期时间添加时区：

```py
# Create datetime
date = pd.Timestamp('2017-05-01 06:00:00')

# Set time zone
date_in_london = date.tz_localize('Europe/London')

# Show datetime
date_in_london
```

```py
Timestamp('2017-05-01 06:00:00+0100', tz='Europe/London')
```

我们也可以转换为不同的时区：

```py
# Change time zone
date_in_london.tz_convert('Africa/Abidjan')
```

```py
Timestamp('2017-05-01 05:00:00+0000', tz='Africa/Abidjan')
```

最后，pandas 的 `Series` 对象可以对每个元素应用 `tz_localize` 和 `tz_convert`：

```py
# Create three dates
dates = pd.Series(pd.date_range('2/2/2002', periods=3, freq='M'))

# Set time zone
dates.dt.tz_localize('Africa/Abidjan')
```

```py
0   2002-02-28 00:00:00+00:00
1   2002-03-31 00:00:00+00:00
2   2002-04-30 00:00:00+00:00
dtype: datetime64[ns, Africa/Abidjan]
```

## 讨论

pandas 支持两组表示时区的字符串；然而，建议使用`pytz`库的字符串。我们可以通过导入`all_timezones`查看表示时区的所有字符串：

```py
# Load library
from pytz import all_timezones

# Show two time zones
all_timezones[0:2]
```

```py
['Africa/Abidjan', 'Africa/Accra']
```

# 7.3 选择日期和时间

## 问题

您有一个日期向量，想要选择一个或多个。

## 解决方案

使用两个布尔条件作为开始和结束日期：

```py
# Load library
import pandas as pd

# Create data frame
dataframe = pd.DataFrame()

# Create datetimes
dataframe['date'] = pd.date_range('1/1/2001', periods=100000, freq='H')

# Select observations between two datetimes
dataframe[(dataframe['date'] > '2002-1-1 01:00:00') &
          (dataframe['date'] <= '2002-1-1 04:00:00')]
```

|  | 日期 |
| --- | --- |
| 8762 | 2002-01-01 02:00:00 |
| 8763 | 2002-01-01 03:00:00 |
| 8764 | 2002-01-01 04:00:00 |

或者，我们可以将日期列设置为 DataFrame 的索引，然后使用`loc`进行切片：

```py
# Set index
dataframe = dataframe.set_index(dataframe['date'])

# Select observations between two datetimes
dataframe.loc['2002-1-1 01:00:00':'2002-1-1 04:00:00']
```

| 日期 | 日期 |
| --- | --- |
| 2002-01-01 01:00:00 | 2002-01-01 01:00:00 |
| 2002-01-01 02:00:00 | 2002-01-01 02:00:00 |
| 2002-01-01 03:00:00 | 2002-01-01 03:00:00 |
| 2002-01-01 04:00:00 | 2002-01-01 04:00:00 |

## 讨论

是否使用布尔条件或索引切片取决于具体情况。如果我们想要进行一些复杂的时间序列操作，将日期列设置为 DataFrame 的索引可能值得开销，但如果我们只想进行简单的数据处理，布尔条件可能更容易。

# 7.4 将日期数据拆分为多个特征

## 问题

您有一个包含日期和时间的列，并且希望创建年、月、日、小时和分钟的特征。

## 解决方案

使用 pandas `Series.dt`中的时间属性：

```py
# Load library
import pandas as pd

# Create data frame
dataframe = pd.DataFrame()

# Create five dates
dataframe['date'] = pd.date_range('1/1/2001', periods=150, freq='W')

# Create features for year, month, day, hour, and minute
dataframe['year'] = dataframe['date'].dt.year
dataframe['month'] = dataframe['date'].dt.month
dataframe['day'] = dataframe['date'].dt.day
dataframe['hour'] = dataframe['date'].dt.hour
dataframe['minute'] = dataframe['date'].dt.minute

# Show three rows
dataframe.head(3)
```

|  | 日期 | 年 | 月 | 日 | 小时 | 分钟 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 2001-01-07 | 2001 | 1 | 7 | 0 | 0 |
| 1 | 2001-01-14 | 2001 | 1 | 14 | 0 | 0 |
| 2 | 2001-01-21 | 2001 | 1 | 21 | 0 | 0 |

## 讨论

有时将日期列分解为各个组成部分会很有用。例如，我们可能希望有一个特征仅包括观察年份，或者我们可能只想考虑某些观测的月份，以便无论年份如何都可以比较它们。

# 7.5 计算日期之间的差异

## 问题

您有两个日期时间特征，想要计算每个观测值之间的时间间隔。

## 解决方案

使用 pandas 减去两个日期特征：

```py
# Load library
import pandas as pd

# Create data frame
dataframe = pd.DataFrame()

# Create two datetime features
dataframe['Arrived'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-04-2017')]
dataframe['Left'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-06-2017')]

# Calculate duration between features
dataframe['Left'] - dataframe['Arrived']
```

```py
0   0 days
1   2 days
dtype: timedelta64[ns]
```

我们经常希望删除`days`的输出，仅保留数值：

```py
# Calculate duration between features
pd.Series(delta.days for delta in (dataframe['Left'] - dataframe['Arrived']))
```

```py
0    0
1    2
dtype: int64
```

## 讨论

有时我们需要的特征是两个时间点之间的变化（delta）。例如，我们可能有客户入住和退房的日期，但我们想要的特征是客户住店的持续时间。pandas 使用`TimeDelta`数据类型使得这种计算变得简单。

## 参见

+   [pandas 文档：时间差](https://oreil.ly/fbgp-)

# 7.6 编码星期几

## 问题

您有一个日期向量，想知道每个日期的星期几。

## 解决方案

使用 pandas `Series.dt`方法的`day_name()`：

```py
# Load library
import pandas as pd

# Create dates
dates = pd.Series(pd.date_range("2/2/2002", periods=3, freq="M"))

# Show days of the week
dates.dt.day_name()
```

```py
0    Thursday
1      Sunday
2     Tuesday
dtype: object
```

如果我们希望输出为数值形式，因此更适合作为机器学习特征使用，可以使用`weekday`，其中星期几表示为整数（星期一为 0）。

```py
# Show days of the week
dates.dt.weekday
```

```py
0    3
1    6
2    1
dtype: int64
```

## 讨论

如果我们想要比较过去三年每个星期日的总销售额，知道星期几可能很有帮助。pandas 使得创建包含星期信息的特征向量变得很容易。

## 另请参阅

+   [pandas Series 日期时间特性](https://oreil.ly/3Au86)

# 7.7 创建滞后特征

## 问题

你想要创建一个滞后*n*个时间段的特征。

## 解决方案

使用 pandas 的`shift`方法：

```py
# Load library
import pandas as pd

# Create data frame
dataframe = pd.DataFrame()

# Create data
dataframe["dates"] = pd.date_range("1/1/2001", periods=5, freq="D")
dataframe["stock_price"] = [1.1,2.2,3.3,4.4,5.5]

# Lagged values by one row
dataframe["previous_days_stock_price"] = dataframe["stock_price"].shift(1)

# Show data frame
dataframe
```

|  | 日期 | 股票价格 | 前几天的股票价格 |
| --- | --- | --- | --- |
| 0 | 2001-01-01 | 1.1 | NaN |
| 1 | 2001-01-02 | 2.2 | 1.1 |
| 2 | 2001-01-03 | 3.3 | 2.2 |
| 3 | 2001-01-04 | 4.4 | 3.3 |
| 4 | 2001-01-05 | 5.5 | 4.4 |

## 讨论

数据往往基于定期间隔的时间段（例如每天、每小时、每三小时），我们有兴趣使用过去的值来进行预测（通常称为*滞后*一个特征）。例如，我们可能想要使用前一天的价格来预测股票的价格。使用 pandas，我们可以使用`shift`将值按一行滞后，创建一个包含过去值的新特征。

在我们的解决方案中，`previous_days_stock_price`的第一行是一个缺失值，因为没有先前的`stock_price`值。

# 7.8 使用滚动时间窗口

## 问题

给定时间序列数据，你想要计算一个滚动时间的统计量。

## 解决方案

使用 pandas DataFrame `rolling`方法：

```py
# Load library
import pandas as pd

# Create datetimes
time_index = pd.date_range("01/01/2010", periods=5, freq="M")

# Create data frame, set index
dataframe = pd.DataFrame(index=time_index)

# Create feature
dataframe["Stock_Price"] = [1,2,3,4,5]

# Calculate rolling mean
dataframe.rolling(window=2).mean()
```

|  | 股票价格 |
| --- | --- |
| 2010-01-31 | NaN |
| 2010-02-28 | 1.5 |
| 2010-03-31 | 2.5 |
| 2010-04-30 | 3.5 |
| 2010-05-31 | 4.5 |

## 讨论

*滚动*（也称为*移动*）*时间窗口*在概念上很简单，但一开始可能难以理解。假设我们有一个股票价格的月度观察数据。经常有用的是设定一个特定月数的时间窗口，然后在观察数据上移动，计算时间窗口内所有观察数据的统计量。

例如，如果我们有一个三个月的时间窗口，并且想要一个滚动均值，我们可以计算：

1.  `mean(一月, 二月, 三月)`

1.  `mean(二月, 三月, 四月)`

1.  `mean(三月, 四月, 五月)`

1.  等等

另一种表达方式：我们的三个月时间窗口“漫步”过观测值，在每一步计算窗口的平均值。

pandas 的`rolling`方法允许我们通过使用`window`指定窗口大小，然后快速计算一些常见统计量，包括最大值（`max()`）、均值（`mean()`）、值的数量（`count()`）和滚动相关性（`corr()`）。

滚动均值通常用于平滑时间序列数据，因为使用整个时间窗口的均值可以抑制短期波动的影响。

## 另请参阅

+   [pandas 文档：滚动窗口](https://oreil.ly/a5gZQ)

+   [什么是移动平均或平滑技术？](https://oreil.ly/aoOSe)

# 7.9 处理时间序列中的缺失数据

## 问题

你在时间序列数据中有缺失值。

## 解决方案

除了前面讨论的缺失数据策略之外，当我们有时间序列数据时，我们可以使用插值来填补由缺失值引起的间隙：

```py
# Load libraries
import pandas as pd
import numpy as np

# Create date
time_index = pd.date_range("01/01/2010", periods=5, freq="M")

# Create data frame, set index
dataframe = pd.DataFrame(index=time_index)

# Create feature with a gap of missing values
dataframe["Sales"] = [1.0,2.0,np.nan,np.nan,5.0]

# Interpolate missing values
dataframe.interpolate()
```

|  | 销售 |
| --- | --- |
| 2010-01-31 | 1.0 |
| 2010-02-28 | 2.0 |
| 2010-03-31 | 3.0 |
| 2010-04-30 | 4.0 |
| 2010-05-31 | 5.0 |

或者，我们可以用最后一个已知值替换缺失值（即向前填充）：

```py
# Forward fill
dataframe.ffill()
```

|  | 销售 |
| --- | --- |
| 2010-01-31 | 1.0 |
| 2010-02-28 | 2.0 |
| 2010-03-31 | 2.0 |
| 2010-04-30 | 2.0 |
| 2010-05-31 | 5.0 |

我们还可以用最新的已知值替换缺失值（即向后填充）：

```py
# Backfill
dataframe.bfill()
```

|  | 销售 |
| --- | --- |
| 2010-01-31 | 1.0 |
| 2010-02-28 | 2.0 |
| 2010-03-31 | 5.0 |
| 2010-04-30 | 5.0 |
| 2010-05-31 | 5.0 |

## 讨论

*插值*是一种填补由缺失值引起的间隙的技术，实质上是在已知值之间绘制一条直线或曲线，并使用该线或曲线来预测合理的值。当时间间隔恒定、数据不易受到嘈杂波动影响、缺失值引起的间隙较小时，插值尤为有用。例如，在我们的解决方案中，两个缺失值之间的间隙由 `2.0` 和 `5.0` 所界定。通过在 `2.0` 和 `5.0` 之间拟合一条直线，我们可以推测出 `3.0` 到 `4.0` 之间的两个缺失值的合理值。

如果我们认为两个已知点之间的线是非线性的，我们可以使用 `interpolate` 的 `method` 参数来指定插值方法：

```py
# Interpolate missing values
dataframe.interpolate(method="quadratic")
```

|  | 销售 |
| --- | --- |
| 2010-01-31 | 1.000000 |
| 2010-02-28 | 2.000000 |
| 2010-03-31 | 3.059808 |
| 2010-04-30 | 4.038069 |
| 2010-05-31 | 5.000000 |

最后，我们可能有大量的缺失值间隙，但不希望在整个间隙内插值。在这些情况下，我们可以使用 `limit` 限制插值值的数量，并使用 `limit_direction` 来设置是从间隙前的最后已知值向前插值，还是反之：

```py
# Interpolate missing values
dataframe.interpolate(limit=1, limit_direction="forward")
```

|  | 销售 |
| --- | --- |
| 2010-01-31 | 1.0 |
| 2010-02-28 | 2.0 |
| 2010-03-31 | 3.0 |
| 2010-04-30 | NaN |
| 2010-05-31 | 5.0 |

向后填充和向前填充是一种朴素插值的形式，其中我们从已知值开始绘制一条平直线，并用它来填充缺失值。与插值相比，向后填充和向前填充的一个（轻微）优势在于它们不需要在缺失值的*两侧*都有已知值。
