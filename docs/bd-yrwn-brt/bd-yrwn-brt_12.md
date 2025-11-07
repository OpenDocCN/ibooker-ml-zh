# 附录 A. 硬件购买指南

本附录涵盖了构建本书中所述机器人所需的各类硬件组件。它提供了书项目中所需产品的具体型号细节，以及销售这些产品的在线零售商的产品页面链接。本书中使用了三种不同的机器人配置，本附录涵盖了所有这些配置所需的硬件。在购买硬件之前，请务必查阅本指南。还值得注意的是，附录 D 提供了一种模拟机器人硬件的机制，可以在任何笔记本电脑或台式计算机上运行本书中的所有代码。

## A.1 Raspberry Pi 组件

Raspberry Pi 是由 Raspberry Pi 基金会创建的小型单板计算机 ([`raspberrypi.org`](https://raspberrypi.org))。它是本书中所有机器人项目的核心。Pi 还支持广泛的附加硬件板，这些板可以增加计算机的额外功能。我们项目需要以下 Raspberry Pi 硬件：

+   Raspberry Pi 4 ([`mng.bz/A8aK`](http://mng.bz/A8aK))，带有 2 GB 或更多 RAM

+   Raspberry Pi 摄像头模块 2 ([`mng.bz/ZRVO`](http://mng.bz/ZRVO))，标准版或 NoIR 版本

+   Adafruit 摄像头外壳 ([`www.adafruit.com/product/3253`](https://www.adafruit.com/product/3253))，用于 Pi 摄像头

+   Adafruit CRICKIT HAT for Raspberry Pi ([`www.adafruit.com/product/3957`](https://www.adafruit.com/product/3957))，用于控制直流和伺服电机

+   Pimoroni Pibow Coupe 4 ([`shop.pimoroni.com/products/pibow-coupe-4`](https://shop.pimoroni.com/products/pibow-coupe-4))，Raspberry Pi 4 的机箱

全球有许多本地和在线零售商销售这些产品。以下是一些有用的提示和网站，以帮助您选择最适合您所在地区的最佳选项：

+   Raspberry Pi 基金会在其每个产品页面上列出官方零售商，您可以在其网站上点击购买产品时找到这些零售商。在线工具列出了特定产品和国家/地区的官方零售商。

+   Adafruit 产品可以在网上购买 ([`www.adafruit.com/`](https://www.adafruit.com/)) 或通过其官方分销商之一 ([`www.adafruit.com/distributors`](https://www.adafruit.com/distributors))。

+   Pimoroni 产品可以在网上 ([`shop.pimoroni.com/`](https://shop.pimoroni.com/)) 或通过分销商 ([`mng.bz/RmN0`](http://mng.bz/RmN0)) 购买。

Raspberry Pi 需要使用 microSD 卡或 USB 闪存驱动器作为存储。只要满足安装 Raspberry Pi OS 的空间要求，本书中的项目就没有特定的存储要求。以下是一些关于不同可用存储选项需要注意的要点：

+   如果这是您第一次使用树莓派，那么购买套件会提供许多对初学者有帮助的额外物品，并且通常物有所值。套件通常会包括用于存储的内存卡、电源和用于视频输出的 HDMI 线。其中一个选项是树莓派 4 桌面套件([`mng.bz/27gd`](http://mng.bz/27gd))。Pimoroni 树莓派 4 基础套件([`mng.bz/1JaV`](http://mng.bz/1JaV))也是另一个受欢迎的选择。如果您发现常规的树莓派 4 缺货，这个套件也是一个不错的选择。

+   USB 闪存驱动器在树莓派上可以比 microSD 卡快得多，这使得在计算机上安装和升级软件的速度也更快。关于树莓派存储性能的这篇文章提供了更多关于磁盘基准测试和快速 USB 闪存驱动器的详细信息([`mng.bz/PRN9`](http://mng.bz/PRN9))。

+   由于 USB 闪存驱动器在树莓派上的位置，它们通常比 microSD 卡更容易更换。当树莓派完全组装成机器人底盘时，这一点尤其正确，因为 USB 端口比 microSD 插槽更容易访问。您可能想要移除存储设备，以便能够轻松地在另一台计算机上备份整个系统，或者您可能有多个 USB 闪存驱动器，每个驱动器都有不同的软件设置，可以互换使用。

## A.2 电机、底盘套件和摇杆控制器

两种最常见的电机类型是直流电机和伺服电机。这两种类型在本书中都有使用。机器人底盘也需要用来连接计算机、电机和电池。推荐的底盘套件有三层，比较小的两层底盘套件提供了更多的空间用于电路板和电池：

+   Adafruit 三层机器人底盘套件([`www.adafruit.com/product/3244`](https://www.adafruit.com/product/3244))包含两个直流电机和轮子。

+   Adafruit 迷你万向节套件([`www.adafruit.com/product/1967`](https://www.adafruit.com/product/1967))已经组装好，包含两个执行俯仰和倾斜动作的微型伺服电机。

这两个套件都非常灵活，支持许多不同的硬件平台。它们的尺寸、连接性和电源需求非常适合使用 CRICKIT HAT 的树莓派。

对于第七章，该章节涉及使用摇杆控制机器人，控制器有多种硬件选项。可以使用原始的索尼 PlayStation 4 或 5 控制器。也可以使用原始的 Xbox 或兼容控制器。以下两个兼容 Xbox 的控制器已经过测试，可以在 Linux 和树莓派上使用：

+   与 Xbox 360 兼容的 W&O 无线控制器([`a.co/d/7FA95aj`](https://a.co/d/7FA95aj))

+   Xbox 360 的 YAEYE 控制器([`a.co/d/8lsabwI`](https://a.co/d/8lsabwI))

需要注意的一点是，无线蓝牙连接仅适用于 PlayStation 控制器。然而，你可以使用无线网络连接和任何控制器来控制机器人，这可以通过第七章中介绍的方法，即通过 Wi-Fi 网络连接远程控制机器人。

## A.3 电源和布线

树莓派和 CRICKIT HAT 各自需要一个电源。电源选项很多，从电池组到连接电源线到电源插座都有。推荐的方法是使用一个单 USB 电源宝为两个设备供电。有一些电源宝允许同时连接并给两个设备供电。电源宝是可充电和便携的。我们需要一个便携式电源，以便我们的机器人在没有连接电线的情况下行驶。任何支持同时为两个设备充电的 USB 电源宝都可以使用。以下是一个经过测试并且工作良好的电源宝：

+   安克 PowerCore Select 10000 ([`walmart.com/ip/Anker/211593977`](https://walmart.com/ip/Anker/211593977))，具有双 12W 输出端口和 10000mAh 的电量

CRICKIT HAT 通过一个圆柱形插头连接器接收电源，因此使用 USB 到圆柱形插头线缆将其连接到电源宝。我们还需要扩展跳线，因为机器人底盘套件中的电缆长度不足以连接到 CRICKIT HAT，一旦我们组装了所有必需的部件。以下是一些推荐的电缆：

+   USB 到圆柱形插头线缆 ([`www.adafruit.com/product/2697`](https://www.adafruit.com/product/2697))

+   高级 M/M 扩展跳线 ([`www.adafruit.com/product/1956`](https://www.adafruit.com/product/1956))

## A.4 可选购买

有许多物品可以帮助改善你的机器人构建体验，但这些不是必需的。你经常会想要拆解和重新配置你的机器人硬件以适应不同的布局。你可能正在尝试不同的电机布局，或者修改电池的位置以改变机器人的重心。在这个过程中，你将希望能够轻松地将树莓派和电源宝从底盘上连接和断开。涤纶粘合方形是解决这个问题的绝佳方案。当与树莓派、CRICKIT HAT 和机器人底盘一起工作时，每个板子和底盘上都有许多位置可以牢固地拧紧部件。尼龙螺丝和支架套件提供了许多不同长度和类型的螺丝和支架，用于此特定目的。磁吸 USB 线缆提供了一种干净且易于连接和断开电源宝到树莓派以及连接电源宝到 USB 充电器的方法。SlimRun 以太网电缆比标准网络电缆轻便且细薄，这在使用有线网络连接时为机器人提供了更多的机动性：

+   涤纶粘合方形 ([`a.co/d/8Sz6OMi`](https://a.co/d/8Sz6OMi))

+   尼龙螺丝和支架套装 ([`www.adafruit.com/product/3658`](https://www.adafruit.com/product/3658))

+   七件套磁吸 USB 线 ([`a.co/d/cYc3waP`](https://a.co/d/cYc3waP))

+   Monoprice SlimRun 网络线 ([`a.co/d/0GBLsyQ`](https://a.co/d/0GBLsyQ))
