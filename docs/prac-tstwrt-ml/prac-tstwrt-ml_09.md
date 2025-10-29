# 附录 A. 合成数据生成工具

对于特定领域的数据生成工具，您有多种选择。

在 GAN 和基于流模型的精神中，有许多项目在真实世界数据上训练生成模型，然后将生成器用作合成数据的源。表格 A-1 列出了几种基于 GAN 的方法。

表格 A-1\. 数据驱动方法和工具^(a)

| 方法和工具 | 描述 | 进一步阅读 | 类型 |
| --- | --- | --- | --- |
| [CTGAN](https://oreil.ly/CL8JQ) | 基于 GAN 的数据合成器，可以高度保真地生成合成表格数据 | [“使用条件 GAN 建模表格数据”](https://arxiv.org/pdf/1907.00503.pdf) | 表格 |
| [TGAN](https://oreil.ly/hVQrf) | 已过时，已被 CTGAN 取代 |  | 表格 |
| [gretel](https://oreil.ly/drIwO) | 创建具有增强隐私保证的虚假合成数据集 |  | 表格 |
| WGAN-GP | 推荐用于训练 GAN；比其他基于 GAN 的数据生成工具更少受模式崩溃影响，并且损失更有意义 | [“关于使用 GAN 生成和评估合成表格数据”](https://oreil.ly/VH753) | 表格 |
| [DataSynthesizer](https://oreil.ly/z6r5y) | “生成模拟给定数据集的合成数据，并应用差分隐私技术以实现强隐私保证” |  | 表格 |
| [MedGAN](https://oreil.ly/5YGkd) | “[一种] 生成对抗网络，用于生成多标签离散患者记录 [可以生成二进制和计数变量（例如，诊断代码、药物代码或程序代码等医疗代码）]” | [“使用生成对抗网络生成多标签离散患者记录”](https://arxiv.org/abs/1703.06490) | 表格 |
| [MC-MedGAN (多分类 GANs)](https://oreil.ly/Pcm7P) | 生成具有多个标签的合成数据实例，特别注重合成医疗数据 | [“使用生成对抗网络生成多类别样本”](https://arxiv.org/pdf/1807.01202.pdf) | 表格 |
| [tableGAN](https://oreil.ly/D1Mfa) | 基于 GAN 架构（DCGAN）的合成数据生成技术 | [“基于生成对抗网络的数据合成”](https://oreil.ly/ZYguO) | 表格 |
| [VEEGAN](https://oreil.ly/SQFfJ) | 使用隐式变分学习减少 GAN 中的模式崩溃 | [“VEEGAN：使用隐式变分学习减少 GAN 中的模式崩溃”](https://arxiv.org/abs/1705.07761) | 表格 |
| [DP-WGAN](https://oreil.ly/Civtk) | “训练一个 Wasserstein GAN（WGAN），该模型在真实的私密数据集上进行训练。通过清洗（范数剪辑和添加高斯噪声）鉴别器的梯度应用差分隐私训练。通过向生成器输入随机噪声来生成合成数据集。” |  | 表格 |
| [DP-GAN（差分私有 GAN）](https://oreil.ly/q0lMJ) | 描述了“差分私有释放语义丰富数据” | [“通过深度生成模型进行差分私有释放”](https://arxiv.org/abs/1801.01594) | 表格 |
| [DP-GAN 2](https://oreil.ly/f6aBy) | 改进了原始的 DP-GAN | [“差分私有生成对抗网络”](https://arxiv.org/abs/1802.06739) | 表格 |
| PateGAN | 修改了教师集成的私有聚合（PATE）框架，并应用于 GAN | [“PATE-GAN: 生成具有差分私有性保证的合成数据”](https://oreil.ly/U82Bl) | 表格 |
| [bnomics](https://oreil.ly/bX4eM) | 使用概率贝叶斯网络生成合成数据 | [“使用概率贝叶斯网络生成合成数据”](https://doi.org/10.1101/2020.06.14.151084) | 表格 |
| [CLGP（分类潜在高斯过程）](https://oreil.ly/TE3sI) | 多元分类数据生成的生成模型 | [“用于多元分类数据分布估计的潜在高斯过程”](https://oreil.ly/XHoc0) | 表格 |
| [COR-GAN](https://oreil.ly/einjl) | 使用“捕获相关性的卷积生成对抗网络生成合成医疗记录” | [“CorGAN: 用于生成合成医疗记录的捕获相关性卷积生成对抗网络”](https://arxiv.org/pdf/2001.09346v2.pdf) | 表格 |
| [synergetr](https://oreil.ly/4FwjN) | “用于生成带有经验概率分布的合成数据的 R 包” |  | 表格 |
| [DPautoGAN](https://oreil.ly/v3Kt9) | 用于生成混合类型数据的差分私有无监督学习任务工具 | [“差分私有混合类型数据生成用于无监督学习”](https://oreil.ly/awvag) | 表格 |
| [SynC](https://oreil.ly/Zu97Q) | “用高斯 Copula 生成合成人口的统一框架” | [“SynC: 用高斯 Copula 生成合成人口的统一框架”](https://arxiv.org/abs/1904.07998) | 表格 |
| [Bn-learn Latent Model](https://oreil.ly/SMYUy) | “为评估机器学习医疗保健软件生成高保真度合成患者数据” | [“为评估机器学习医疗保健软件生成高保真度合成患者数据”](https://oreil.ly/rszbN) | 表格 |
| [SAP Security research sample](https://oreil.ly/mkoFZ) | 使用生成式深度学习模型生成差分私有合成数据集 |  | 表格 |
| [Python synthpop](https://oreil.ly/Z4tr9) | “R 包 synthpop 的 Python 实现” |  | 表格 |
| [synthia](https://oreil.ly/Sw6uf) | Python 中的“多维合成数据生成” |  | 表格 |
| [Synthetic_Data_System](https://oreil.ly/orewB) | “用于思想收集、测试和评论的 SDS Alpha 版” |  | 表格 |
| [QUIPP](https://oreil.ly/uqVbF) | 创建“隐私保护合成数据生成工作流” |  | 表格 |
| [MSFT 合成数据展示](https://oreil.ly/wt9c4) | 展示“生成合成数据和用于隐私保护数据共享和分析的 UI” |  | 表格 |
| [扩展的 MedGan](https://oreil.ly/kh3bu) | 使用 GAN 创建“合成患者数据” |  | 表格 |
| [合成数据](https://oreil.ly/HiC6H) | 提供来自私人健康研究的资产 |  | 表格 |
| [贝叶斯合成生成器](https://oreil.ly/6CIDu) | 一个软件系统的仓库，用于基于贝叶斯网络块结构生成合成个人数据 |  | 表格 |
| [使用 GAN 生成合成数据](https://oreil.ly/TXxXY) | 解决如何安全高效地共享加密数据并使用 GAN 生成虚假图像生成合成表格数据的机制的问题 |  | 表格 |
| [HoloClean](https://oreil.ly/twei4) | 机器学习系统，利用“质量规则、值相关性、参考数据和多种其他信号构建概率模型，以扩展现有数据集” |  | 表格 |
| [SYNDATA](https://oreil.ly/xRcXi) | 生成和评估合成患者数据 | [“生成和评估合成患者数据”](https://doi.org/10.1186/s12874-020-00977-1) | 表格 |
| [SDV 评估函数](https://oreil.ly/DEfOY) | 提供“表格、关系和时间序列数据的合成数据生成” |  | 多种格式 |
| [MTSS-GAN](https://oreil.ly/81kPn) | “使用 GAN 生成的多变量时间序列模拟” | [“MTSS-GAN：多变量时间序列模拟生成对抗网络”](https://dx.doi.org/10.2139/ssrn.3616557) | 时间序列 |
| [数据生成](https://oreil.ly/AmHRl) | “一个实现圆柱钟形漏斗时间序列数据生成器的数据生成，用于不同长度、维度和样本的数据生成” |  | 时间序列 |
| [RGAN](https://oreil.ly/gEM0q) | 用于生成实值时间序列数据的“递归（条件）GAN” | [“使用递归条件 GAN 生成实值（医学）时间序列的文章”](https://arxiv.org/abs/1706.02633) | 时间序列 |
| [机器学习交易 Chapter 21](https://oreil.ly/jEhj1) | 用于算法交易中创建合成时间序列数据的教程 |  | 时间序列 |
| [tsBNgen](https://oreil.ly/68p1s) | “一个 Python 库，根据任意贝叶斯网络结构生成时间序列数据” | [“tsBNgen：根据任意动态贝叶斯网络结构生成时间序列数据的 Python 库”](https://arxiv.org/pdf/2009.04595.pdf) | 时间序列 |
| [合成数据生成](https://oreil.ly/GIA3x) | 用于 QuantUniversity 关于“金融中的合成数据生成”的讲座材料 | [“金融中的合成数据生成”](https://oreil.ly/us9xl) | 时间序列 |
| [LSTM GAN model](https://oreil.ly/KcjzL) | “LSTM GAN 模型可用于生成合成的多维时间序列数据” |  | Time series |
| [synsys](https://oreil.ly/fyev8) | 提供传感器数据 |  | Sensor data |
| ^(a) 我们添加了描述，其中我们希望详细说明，并从 GitHub 仓库中直接复制了创作者或作者自行提供的最佳说明。 |

如前所述，一些合成数据方法依赖于由人类输入的领域知识引导的过程。表 A-2 列出了这些方法。

表 A-2\. 基于过程的方法和工具^(a)

| 方法和工具 | 描述 | 类型 |
| --- | --- | --- |
| [plaitpy](https://oreil.ly/ZEt1T) | “从可组合的 yaml 模板生成假数据的程序” | Tabular |
| [pySyntheticDatasetGenerator](https://oreil.ly/qKFvI) | 基于 YAML 配置文件生成类型检查的合成数据表格的工具 | Tabular |
| [SimPop](https://oreil.ly/zefmG) | “用于基于辅助数据模拟调查人群的工具和方法：基于模型的方法、校准和组合优化算法” | Tabular |
| [datasynthR](https://oreil.ly/CCznP) | “用于测试和协作中在 R 中程序生成合成数据的函数集合” | Tabular |
| [synner](https://oreil.ly/CKxli) | “生成逼真的合成数据” | Tabular |
| [synthea](https://oreil.ly/HzU6Q) | “合成患者人口模拟器 | 患者和医疗数据” |
| [BadMedicine](https://oreil.ly/qYrgO) | “用于随机生成类似电子健康记录（EHR）系统输出的医疗数据的库和 CLI” | 患者和医疗数据” |
| ^(a) 我们添加了描述，其中我们希望详细说明，并从 GitHub 仓库中直接复制了创作者或作者自行提供的最佳说明。 |

以下列表介绍了用于评估合成数据质量的工具：

+   [datagene](https://oreil.ly/05HUS)

+   [SDMetrics](https://oreil.ly/J2E2J)

+   [table-evaluator](https://oreil.ly/PmsWt)

+   [Statistical-Similarity-Measurement](https://oreil.ly/IsPsv)

+   [SDGym](https://oreil.ly/un19m)

+   [virtualdatalab](https://oreil.ly/UHf5D)

开始学习合成数据生成相对容易。做好这件事情则需要更加仔细的过程。

表 A-3 展示了其他不容易适应前述类别的工具和资源。

表 A-3\. 用于生成合成数据的其他工具和资源^(a)

| 工具和资源 | 描述 | 进一步阅读 |
| --- | --- | --- |
| [pomegranate](https://oreil.ly/pPZKM) | 用于构建概率模型的软件包 |  |
| [“生成表格合成数据”](https://oreil.ly/RKjbs) | 一个关于使用最先进的 GAN 架构“生成表格合成数据”的教程 |  |
| [ydata-synthetic](https://oreil.ly/r4TTN) | “与 GAN 相关的合成数据生成材料，特别是常规表格数据和时间序列” |  |
| [jclymo/DataGen_NPBE](https://oreil.ly/eJQCU) | 用于让神经网络编写程序等任务，这是一个用于生成训练数据的工具，为训练这样的网络提供所需数据 | [“通过示例生成神经编程数据”](https://arxiv.org/abs/1911.02624) |
| [SynthMedTopia](https://oreil.ly/7oHjD) | 一个项目，用于“生成合成医疗数据并将真实和合成数据转换为机器学习格式” |  |
| [spiros/tofu](https://oreil.ly/GF58P) | 一个“用于生成合成 UK Biobank 数据的 Python 工具” |  |
| [chasebos91/GAN-for-Synthetic-EEG-Data](https://oreil.ly/4h82D) | 用于生成合成 EEG 数据的 GAN |  |
| [jgalilee/data](https://oreil.ly/Hky2p) | 一个用 Go 编写的“合成数据生成工具” |  |
| [blt2114/overpruning_in_variational_bnns](https://oreil.ly/72QoX) | 用于“变分贝叶斯神经网络过度修剪的合成数据实验代码” |  |
| [avensolutions/synthetic-cdc-data-generator](https://oreil.ly/ZSqDG) | 一个“生成变更集的应用程序，可用于开发和测试用于源变更数据捕获（CDC）处理的合成数据” |  |
| [nikk-nikaznan/SSVEP-Neural-Generative-Models](https://oreil.ly/O6rBb) | 使用生成神经网络创建合成 EEG 数据的应用程序 | [“模拟脑信号：通过基于神经的生成模型创建合成 EEG 数据以改善 SSVEP 分类”](https://arxiv.org/abs/1901.07429) |
| ^(a) 我们添加了描述，希望能更详细地说明，并且直接从 GitHub 存储库复制，因为我们认为创建者或作者最能准确描述自己的内容。 |
