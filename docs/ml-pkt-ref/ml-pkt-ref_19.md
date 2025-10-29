# 第十九章：管道

Scikit-learn 使用管道的概念。使用 `Pipeline` 类，您可以将转换器和模型链在一起，并像对待整个 scikit-learn 模型一样对待整个过程。甚至可以插入自定义逻辑。

# 分类管道

这是在管道内使用 `tweak_titanic` 函数的示例：

```py
>>> from sklearn.base import (
...     BaseEstimator,
...     TransformerMixin,
... )
>>> from sklearn.pipeline import Pipeline

>>> def tweak_titanic(df):
...     df = df.drop(
...         columns=[
...             "name",
...             "ticket",
...             "home.dest",
...             "boat",
...             "body",
...             "cabin",
...         ]
...     ).pipe(pd.get_dummies, drop_first=True)
...     return df

>>> class TitanicTransformer(
...     BaseEstimator, TransformerMixin
... ):
...     def transform(self, X):
...         # assumes X is output
...         # from reading Excel file
...         X = tweak_titanic(X)
...         X = X.drop(column="survived")
...         return X
...
...     def fit(self, X, y):
...         return self

>>> pipe = Pipeline(
...     [
...         ("titan", TitanicTransformer()),
...         ("impute", impute.IterativeImputer()),
...         (
...             "std",
...             preprocessing.StandardScaler(),
...         ),
...         ("rf", RandomForestClassifier()),
...     ]
... )
```

有了管道，我们可以对其调用 `.fit` 和 `.score`：

```py
>>> from sklearn.model_selection import (
...     train_test_split,
... )
>>> X_train2, X_test2, y_train2, y_test2 = train_test_split(
...     orig_df,
...     orig_df.survived,
...     test_size=0.3,
...     random_state=42,
... )

>>> pipe.fit(X_train2, y_train2)
>>> pipe.score(X_test2, y_test2)
0.7913486005089059
```

管道可以在网格搜索中使用。我们的 `param_grid` 需要将参数前缀设为管道阶段的名称，后跟两个下划线。在下面的示例中，我们为随机森林阶段添加了一些参数：

```py
>>> params = {
...     "rf__max_features": [0.4, "auto"],
...     "rf__n_estimators": [15, 200],
... }

>>> grid = model_selection.GridSearchCV(
...     pipe, cv=3, param_grid=params
... )
>>> grid.fit(orig_df, orig_df.survived)
```

现在我们可以提取最佳参数并训练最终模型。（在这种情况下，随机森林在网格搜索后没有改善。）

```py
>>> grid.best_params_
{'rf__max_features': 0.4, 'rf__n_estimators': 15}
>>> pipe.set_params(**grid.best_params_)
>>> pipe.fit(X_train2, y_train2)
>>> pipe.score(X_test2, y_test2)
0.7913486005089059
```

我们可以在使用 scikit-learn 模型的管道中使用：

```py
>>> metrics.roc_auc_score(
...     y_test2, pipe.predict(X_test2)
... )
0.7813688715131023
```

# 回归管道

这是在波士顿数据集上执行线性回归的管道示例：

```py
>>> from sklearn.pipeline import Pipeline

>>> reg_pipe = Pipeline(
...     [
...         (
...             "std",
...             preprocessing.StandardScaler(),
...         ),
...         ("lr", LinearRegression()),
...     ]
... )
>>> reg_pipe.fit(bos_X_train, bos_y_train)
>>> reg_pipe.score(bos_X_test, bos_y_test)
0.7112260057484934
```

如果我们想要从管道中提取部分来检查它们的属性，我们可以使用 `.named_steps` 属性进行操作：

```py
>>> reg_pipe.named_steps["lr"].intercept_
23.01581920903956
>>> reg_pipe.named_steps["lr"].coef_
array([-1.10834602,  0.80843998,  0.34313466,
 0.81386426, -1.79804295,  2.913858  ,
 -0.29893918, -2.94251148,  2.09419303,
 -1.44706731, -2.05232232,  1.02375187,
 -3.88579002])_
```

我们也可以在度量计算中使用管道：

```py
>>> from sklearn import metrics
>>> metrics.mean_squared_error(
...     bos_y_test, reg_pipe.predict(bos_X_test)
... )
21.517444231177205
```

# PCA 管道

Scikit-learn 管道也可以用于 PCA。

这里我们对泰坦尼克号数据集进行标准化并对其执行 PCA：

```py
>>> pca_pipe = Pipeline(
...     [
...         (
...             "std",
...             preprocessing.StandardScaler(),
...         ),
...         ("pca", PCA()),
...     ]
... )
>>> X_pca = pca_pipe.fit_transform(X)
```

使用 `.named_steps` 属性，我们可以从管道的 PCA 部分提取属性：

```py
>>> pca_pipe.named_steps[
...     "pca"
... ].explained_variance_ratio_
array([0.23917891, 0.21623078, 0.19265028,
 0.10460882, 0.08170342, 0.07229959,
 0.05133752, 0.04199068])
>>> pca_pipe.named_steps["pca"].components_[0]
array([-0.63368693,  0.39682566,  0.00614498,
 0.11488415,  0.58075352, -0.19046812,
 -0.21190808, -0.09631388])
```
