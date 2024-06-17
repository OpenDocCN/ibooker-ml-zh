# 第一章\. 介绍

这不太像是一本教学手册，而更像是关于机器学习的笔记、表格和示例。它是作者在培训期间作为额外资源创建的，旨在作为实体笔记本分发。参与者（喜欢纸质材料的物理特性）可以添加自己的笔记和想法，并获得经过筛选的示例的宝贵参考。

我们将介绍如何使用结构化数据进行分类。其他常见的机器学习应用包括预测连续值（回归）、创建聚类或试图降低维度等。本书不讨论深度学习技术。虽然这些技术在非结构化数据上表现良好，但大多数人推荐本书中的技术来处理结构化数据。

我们假设您具有 Python 的知识和熟悉度。学习如何使用[pandas 库](https://pandas.pydata.org)来处理数据非常有用。我们有许多使用 pandas 的示例，它是处理结构化数据的优秀工具。然而，如果您不熟悉 numpy，一些索引操作可能会令人困惑。完整覆盖 pandas 可能需要一本专著来讨论。

# 使用的库

本书使用了许多库。这既可能是好事，也可能是坏事。其中一些库可能难以安装或与其他库的版本冲突。不要觉得您需要安装所有这些库。使用“即时安装”并仅在需要时安装您想要使用的库。

```py
>>> import autosklearn, catboost,
category_encoders, dtreeviz, eli5, fancyimpute,
fastai, featuretools, glmnet_py, graphviz,
hdbscan, imblearn, janitor, lime, matplotlib,
missingno, mlxtend, numpy, pandas, pdpbox, phate,
pydotplus, rfpimp, scikitplot, scipy, seaborn,
shap, sklearn, statsmodels, tpot, treeinterpreter,
umap, xgbfir, xgboost, yellowbrick

>>> for lib in [
...     autosklearn,
...     catboost,
...     category_encoders,
...     dtreeviz,
...     eli5,
...     fancyimpute,
...     fastai,
...     featuretools,
...     glmnet_py,
...     graphviz,
...     hdbscan,
...     imblearn,
...     lime,
...     janitor,
...     matplotlib,
...     missingno,
...     mlxtend,
...     numpy,
...     pandas,
...     pandas_profiling,
...     pdpbox,
...     phate,
...     pydotplus,
...     rfpimp,
...     scikitplot,
...     scipy,
...     seaborn,
...     shap,
...     sklearn,
...     statsmodels,
...     tpot,
...     treeinterpreter,
...     umap,
...     xgbfir,
...     xgboost,
...     yellowbrick,
... ]:
...     try:
...         print(lib.__name__, lib.__version__)
...     except:
...         print("Missing", lib.__name__)
catboost 0.11.1
category_encoders 2.0.0
Missing dtreeviz
eli5 0.8.2
fancyimpute 0.4.2
fastai 1.0.28
featuretools 0.4.0
Missing glmnet_py
graphviz 0.10.1
hdbscan 0.8.22
imblearn 0.4.3
janitor 0.16.6
Missing lime
matplotlib 2.2.3
missingno 0.4.1
mlxtend 0.14.0
numpy 1.15.2
pandas 0.23.4
Missing pandas_profiling
pdpbox 0.2.0
phate 0.4.2
Missing pydotplus
rfpimp
scikitplot 0.3.7
scipy 1.1.0
seaborn 0.9.0
shap 0.25.2
sklearn 0.21.1
statsmodels 0.9.0
tpot 0.9.5
treeinterpreter 0.1.0
umap 0.3.8
xgboost 0.81
yellowbrick 0.9
```

###### 注意

大多数这些库可以使用`pip`或`conda`轻松安装。对于`fastai`，我需要使用`pip install --no-deps fastai`。`umap`库可以使用`pip install umap-learn`安装。`janitor`库可以使用`pip install pyjanitor`安装。`autosklearn`库可以使用`pip install auto-sklearn`安装。

我通常使用 Jupyter 进行分析。您也可以使用其他笔记本工具。请注意，一些工具如 Google Colab 已预安装了许多库（尽管它们可能是过时版本）。

在 Python 中安装库有两个主要选项。一个是使用`pip`（Python 包管理工具的缩写），这是一个随 Python 一起安装的工具。另一个选项是使用[Anaconda](https://anaconda.org)。我们将两者都介绍。

# 使用 Pip 进行安装

在使用`pip`之前，我们将创建一个沙盒环境，将我们的库安装到其中。这称为名为`env`的虚拟环境：

```py
$ python -m venv env
```

###### 注意

在 Macintosh 和 Linux 上，使用`python`；在 Windows 上，使用`python3`。如果 Windows 在命令提示符中未能识别它，请重新安装或修复您的安装，并确保选中“将 Python 添加到我的 PATH”复选框。

然后，您激活环境，这样当您安装库时，它们将进入沙盒环境而不是全局的 Python 安装中。由于许多这些库会发生变化并进行更新，最好在每个项目基础上锁定版本，这样您就知道您的代码将可以运行。

这是如何在 Linux 和 Macintosh 上激活虚拟环境：

```py
$ source env/bin/activate
```

注意到提示符已更新，表示我们正在使用虚拟环境：

```py
  (env) $ which python
  env/bin/python

```

在 Windows 上，您需要通过运行此命令激活环境：

```py
C:> env\Scripts\activate.bat
```

再次注意到，提示符已更新，表示我们正在使用虚拟环境：

```py
  (env) C:> where python
  env\Scripts\python.exe

```

在所有平台上，您可以使用`pip`安装包。要安装 pandas，键入：

```py
(env) $ pip install pandas
```

一些包名称与库名称不同。您可以使用以下命令搜索包：

```py
(env) $ pip search libraryname
```

安装了包之后，你可以使用`pip`创建一个包含所有包版本的文件：

```py
(env) $ pip freeze > requirements.txt
```

使用此`requirements.txt`文件，您可以轻松地将包安装到新的虚拟环境中：

```py
(other_env) $ pip install -r requirements.txt
```

# 使用 Conda 安装

`conda`工具与 Anaconda 捆绑，允许我们创建环境并安装包。

要创建一个名为`env`的环境，请运行：

```py
$ conda create --name env python=3.6
```

要激活此环境，请运行：

```py
$ conda activate env
```

这将在 Unix 和 Windows 系统上更新提示符。现在你可以使用以下命令搜索包：

```py
(env) $ conda search libraryname
```

要安装像 pandas 这样的包，请运行：

```py
(env) $ conda install pandas
```

要创建包含包需求的文件，请运行：

```py
(env) $ conda env export > environment.yml
```

要在新环境中安装这些要求，请运行：

```py
(other_env) $ conda create -f environment.yml
```

###### 警告

本书提到的一些库无法从 Anaconda 的仓库安装。不要担心。事实证明，你可以在 conda 环境中使用`pip`（无需创建新的虚拟环境），并使用`pip`安装这些库。
