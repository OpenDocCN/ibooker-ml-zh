# 第十三章：后记

1966 年，麻省理工学院教授西摩·帕帕特（Seymour Papert）为他的学生们[启动了一个暑期项目](https://oreil.ly/AC3Xh)。项目的最终目标是通过将物体与已知物体词汇进行匹配，来命名图像中的对象。他为他们详细分解了任务，并期望小组能在几个月内完成。可以说，帕帕特博士稍微低估了这个问题的复杂性。

我们从探讨像全连接神经网络这样的天真机器学习方法开始这本书，这些方法未能利用图像的特殊特征。在第二章中，尝试这些天真的方法使我们学会了如何读取图像，以及如何使用机器学习模型进行训练、评估和预测。

然后，在第三章中，我们介绍了许多创新概念——卷积滤波器、最大池化层、跳跃连接、模块、挤压激活等，这些概念使得现代机器学习模型能够在从图像中提取信息方面表现出色。在实践中实现这些模型通常涉及使用内置的 Keras 模型或 TensorFlow Hub 层。我们还详细讨论了迁移学习和微调。

在第四章中，我们探讨了如何使用在第三章中涵盖的计算机视觉模型来解决计算机视觉中的两个更基本的问题：物体检测和图像分割。

本书的接下来几章深入探讨了创建生产级计算机视觉机器学习模型涉及的每个阶段：

+   在第五章中，我们讨论了如何创建一个对机器学习高效的数据集格式。我们还讨论了标签创建的选项，以及为模型评估和超参数调整保留独立数据集的选项。

+   在第六章中，我们深入研究了预处理和防止训练-服务偏差。预处理可以在`tf.data`输入流水线中完成，在 Keras 层中完成，在`tf.transform`中完成，或者使用这些方法的混合。我们涵盖了每种方法的实施细节及其利弊。

+   在第七章中，我们讨论了模型训练，包括如何在多个 GPU 和工作节点上分布训练。

+   在第八章中，我们探讨了如何监控和评估模型。我们还研究了如何进行分片评估，以诊断模型中的不公平和偏见。

+   在第九章中，我们讨论了用于部署模型的各种选项。我们实现了批处理、流式处理和边缘预测。我们能够在本地和通过网络调用我们的模型。

+   在第十章中，我们向您展示了如何将所有这些步骤结合成一个机器学习流水线。我们还尝试了一个无代码图像分类系统，以利用机器学习的民主化进程。

在第十一章中，我们将视野扩展到图像分类之外。我们看到了计算机视觉的基本构建模块如何用于解决各种问题，包括计数、姿态检测和其他用例。最后，在第十二章中，我们探讨了如何生成图像和字幕。

在整本书中，讨论的概念、模型和流程都伴随着在[GitHub 上的实现](https://github.com/GoogleCloudPlatform/practical-ml-vision-book)。我们强烈建议您不仅阅读本书，还要动手编写代码并进行实验。学习机器学习的最佳方式就是亲自实践。

计算机视觉正处于一个激动人心的阶段。底层技术已经足够成熟，以至于今天，在帕珀特博士向他的学生提出这个问题 50 多年后，我们终于达到了图像分类*可以*完成为期两个月的项目的阶段！我们祝您在将这项技术应用于改善人类生活方面取得巨大成功，并希望您能像我们一样，通过使用计算机视觉来解决真实世界的问题，为此带来无限的乐趣。