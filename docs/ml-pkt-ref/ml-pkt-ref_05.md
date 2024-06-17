# 第五章：清理数据

我们可以使用通用工具如 pandas 和专业工具如 pyjanitor 来帮助清理数据。

# 列名

在使用 pandas 时，使用 Python 友好的列名可以进行属性访问。pyjanitor 的 `clean_names` 函数将返回一个列名为小写并用下划线替换空格的 DataFrame：

```py
>>> import janitor as jn
>>> Xbad = pd.DataFrame(
...     {
...         "A": [1, None, 3],
...         "  sales numbers ": [20.0, 30.0, None],
...     }
... )
>>> jn.clean_names(Xbad)
 a  _sales_numbers_
0  1.0             20.0
1  NaN             30.0
2  3.0              NaN
```

###### 提示

我建议使用索引赋值、`.assign` 方法、`.loc` 或 `.iloc` 赋值来更新列。我还建议不要使用属性赋值来更新 pandas 中的列。由于可能会覆盖同名列的现有方法，属性赋值不能保证可靠地工作。

pyjanitor 库很方便，但不能去除列周围的空白。我们可以使用 pandas 更精细地控制列的重命名：

```py
>>> def clean_col(name):
...     return (
...         name.strip().lower().replace(" ", "_")
...     )

>>> Xbad.rename(columns=clean_col)
 a  sales_numbers
0  1.0           20.0
1  NaN           30.0
2  3.0            NaN
```

# 替换缺失值

pyjanitor 中的 `coalesce` 函数接受一个 DataFrame 和一个要考虑的列列表。这类似于 Excel 和 SQL 数据库中的功能。它返回每行的第一个非空值：

```py
>>> jn.coalesce(
...     Xbad,
...     columns=["A", "  sales numbers "],
...     new_column_name="val",
... )
 val
0   1.0
1  30.0
2   3.0
```

如果我们想要用特定值填充缺失值，可以使用 DataFrame 的 `.fillna` 方法：

```py
>>> Xbad.fillna(10)
 A    sales numbers
0   1.0              20.0
1  10.0              30.0
2   3.0              10.0
```

或者 pyjanitor 的 `fill_empty` 函数：

```py
>>> jn.fill_empty(
...     Xbad,
...     columns=["A", "  sales numbers "],
...     value=10,
... )
 A    sales numbers
0   1.0              20.0
1  10.0              30.0
2   3.0              10.0
```

经常情况下，我们会使用更精细的方法在 pandas、scikit-learn 或 fancyimpute 中执行每列的空值替换。

在创建模型之前，可以使用 pandas 来进行健全性检查，确保处理了所有的缺失值。以下代码在 DataFrame 中返回一个布尔值，用于检查是否有任何缺失的单元格：

```py
>>> df.isna().any().any()
True
```
