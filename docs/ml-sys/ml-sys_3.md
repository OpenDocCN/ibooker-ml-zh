## 设置环境

### Scala

本书中的大部分代码都是用 Scala 编写的。了解如何在你的平台上设置 Scala 的最佳地方是 Scala 语言网站([www.scala-lang.org](http://www.scala-lang.org))，特别是下载部分([www.scala-lang.org/download](http://www.scala-lang.org/download))。本书使用的 Scala 版本是 Scala 2.11.7，但如果你在阅读本书时 2.11 系列有更新的版本，那么最新版本也应该同样适用。如果你已经使用 IntelliJ IDEA、NetBeans 或 Eclipse 等 IDE，安装该 IDE 的相关 Scala 支持可能最简单。

注意，本书提供的所有代码都结构化为类或对象，但并非所有代码都需要以这种方式执行。如果你想使用 Scala REPL 或 Scala 工作表来执行更独立的代码示例，这通常也能很好地工作。

### Git 代码仓库

本书展示的所有代码都可以从本书的网站([www.manning.com/books/reactive-machine-learning-systems](http://www.manning.com/books/reactive-machine-learning-systems))和 GitHub([`github.com`](https://github.com))以 Git 仓库的形式下载。*Reactive Machine Learning Systems*仓库([`github.com/jeffreyksmithjr/reactive-machine-learning-systems`](https://github.com/jeffreyksmithjr/reactive-machine-learning-systems))包含每个章节的项目。如果你对使用 Git 和 GitHub 进行版本控制不熟悉，可以查看入门文章([`help.github.com/categories/bootcamp`](https://help.github.com/categories/bootcamp))和/或初学者资源([`help.github.com/articles/good-resources-for-learning-git-and-github`](https://help.github.com/articles/good-resources-for-learning-git-and-github))来学习这些工具。

### sbt

本书使用了广泛的库。在 Git 仓库中提供的代码中，这些依赖项被指定为可以通过 sbt 解析的方式。许多 Scala 项目使用 sbt 来管理它们的依赖项并构建代码。尽管你不需要使用 sbt 来构建本书提供的多数代码，但通过安装它，你将能够利用 Git 仓库中的项目以及第七章中展示的构建代码的一些特定技术。有关如何开始使用 sbt 的说明，请参阅 sbt 网站([www.scala-sbt.org](http://www.scala-sbt.org))的下载部分([www.scala-sbt.org/download.html](http://www.scala-sbt.org))).本书使用的 sbt 版本是 sbt 13.9，但 13 系列中的任何后续版本都应该同样适用。

### Spark

本书的一些章节使用 Spark 构建机器学习系统的组件。在 GitHub 仓库提供的代码中，您可以像使用任何其他库依赖项一样使用 Spark。但在本地环境中安装完整的 Spark 可以帮助您学习更多。Spark 附带一个名为*Spark shell*的 REPL，这对于与 Spark 代码的探索性交互非常有帮助。有关下载和设置 Spark 的说明可以在 Spark 网站的下载部分([`spark.apache.org/downloads.html`](http://spark.apache.org/downloads.html))找到。本书使用的 Spark 版本是 2.2.0，但 Spark 通常具有非常稳定的 API，因此各种版本应该几乎相同。

### Couchbase

本书使用的数据库是 Couchbase。它是开源的，并且拥有强大的商业支持。要开始安装和设置 Couchbase，最佳起点是访问 Couchbase 网站的开发者部分([`developer.couchbase.com/server`](http://developer.couchbase.com/server))。Couchbase Server 的免费社区版足以满足本书中所有示例的需求。本书使用的版本是 4.0，但 4 系列中的任何后续版本也应能正常工作。

### Docker

第七章介绍了如何使用 Docker，这是一个用于处理容器的工具。它可以在所有常见的桌面操作系统上安装，但安装过程会根据您选择的操作系统有所不同。此外，工具也在快速演变。有关如何在您的计算机上设置 Docker 的最佳信息，请访问 Docker 网站：[www.docker.com](http://www.docker.com)。
