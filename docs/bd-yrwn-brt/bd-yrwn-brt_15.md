# 附录 D. 模拟 CRICKIT 库

本附录涵盖了模拟 Adafruit Python CRICKIT 库的主题。模拟是测试软件时常用的一种机制。它允许我们用模拟对象替换软件的一部分。在我们的情况下，这样做有许多好处：

+   *在不使用机器人硬件的情况下运行代码*——本附录中的模拟库将允许我们在没有任何机器人硬件的情况下运行本书中的所有代码。这很有用，因为它提供了对代码中不需要机器人硬件的计算机视觉、网络、操纵杆和网络部分的更深入的了解。

+   *在不使用树莓派的情况下执行代码*——代码被编写为在许多不同的 Linux 系统上运行。所有代码都在 Ubuntu 22.04 上进行了测试，它可以在任何装有 Windows 或 Mac 的笔记本电脑或虚拟机上运行。

+   *更好的编码体验*——在现代计算机上运行代码通常比在较慢的机器（如树莓派）上执行代码要快得多，也更舒适。例如，当在机器人应用程序的 Web 前端进行大量工作时，在笔记本电脑上开发周期可以更快、更高效。

## D.1 安装模拟 CRICKIT 库

将作为 Python 标准库一部分提供的模拟对象库将用于模拟 CRICKIT 库的不同功能。实现将解决模拟本书中使用的特定功能，而不是整个 CRICKIT 库。本附录将主要关注使用 `mock_crickit` 库，而不会深入探讨实现细节。将以下脚本保存到名为 `adafruit_crickit.py` 的文件中。

列表 D.1 `adafruit_crickit.py`：模拟 CRICKIT 库

```
#!/usr/bin/env python3
import os
from unittest.mock import Mock, PropertyMock
from functools import partial

DEBUG = bool(os.environ.get('ROBO_DEBUG'))
PROP_VALUES = {'touch_1.value': True}

def print_msg(msg):
    if DEBUG:
        print('MOCK_CRICKIT:', msg)

def prop_access(name, *args):
    action = 'set' if args else 'get'
    if action == 'set':
        PROP_VALUES[name] = args[0]
    val = PROP_VALUES.get(name)
    print_msg(f'{action} crickit.{name}: {val!r}')
    return val

def pixel_fill(val):
    print_msg(f'call crickit.onboard_pixel.fill({val!r})')

def add_property(name):
    parent, child = name.split('.')
    property_mock = PropertyMock(side_effect=partial(prop_access, name))
    setattr(type(getattr(crickit, parent)), child, property_mock)

crickit = Mock()
crickit.onboard_pixel.fill = Mock(side_effect=pixel_fill)
names = [
    'onboard_pixel.brightness', 'touch_1.value', 'dc_motor_1.throttle',
    'dc_motor_2.throttle', 'servo_1.angle', 'servo_2.angle',
    'servo_1.actuation_range', 'servo_2.actuation_range']
for name in names:
    add_property(name)

def demo():
    print('starting mock_crickit demo...')
    crickit.onboard_pixel.brightness = 0.01
    crickit.onboard_pixel.fill(0xFF0000)
    crickit.touch_1.value
    crickit.dc_motor_1.throttle = 1
    crickit.dc_motor_2.throttle = -1
    crickit.servo_1.angle = 70
    crickit.servo_1.angle
    crickit.servo_2.angle = 90
    crickit.servo_2.angle
    crickit.servo_1.actuation_range = 142
    crickit.servo_2.actuation_range = 180

if __name__ == "__main__":
    demo()
```

该库被编写为 `adafruit_crickit` 库的直接替代品，这就是为什么它有相同的名字。我们可以用它来替代 Adafruit 库，而无需更改我们的 Python 代码。正如我们在整本书中所做的那样，我们可以设置 `ROBO_DEBUG` 环境变量，使模拟库打印出它接收到的每个模拟调用。当库直接执行时，它将执行 `demo` 函数，该函数演示了它模拟的 CRICKIT 库的所有不同部分。以下会话显示了库的示例运行：

```
$ export ROBO_DEBUG=1
$ ./adafruit_crickit.py
starting mock_crickit demo...
MOCK_CRICKIT: set crickit.onboard_pixel.brightness: 0.01
MOCK_CRICKIT: call crickit.onboard_pixel.fill(16711680)
MOCK_CRICKIT: get crickit.touch_1.value: True
MOCK_CRICKIT: set crickit.dc_motor_1.throttle: 1
MOCK_CRICKIT: set crickit.dc_motor_2.throttle: -1
MOCK_CRICKIT: set crickit.servo_1.angle: 70
MOCK_CRICKIT: get crickit.servo_1.angle: 70
MOCK_CRICKIT: set crickit.servo_2.angle: 90
MOCK_CRICKIT: get crickit.servo_2.angle: 90
MOCK_CRICKIT: set crickit.servo_1.actuation_range: 142
MOCK_CRICKIT: set crickit.servo_2.actuation_range: 180
```

我们还可以将模拟库安装到我们选择的任何 Python 虚拟环境中。库的代码和安装程序可以在 GitHub 上找到（[`github.com/marwano/robo`](https://github.com/marwano/robo)）。在下一个会话中，我们将使用 `pip install` 命令安装 `mock_crickit` 库。请确保在设置脚本所在的目录中运行 `pip install` 命令：

```
(main) robo@robopi:/tmp$ cd mock_crickit
(main) robo@robopi:/tmp/mock_crickit$ pip install .
Processing /tmp/mock_crickit
 Installing build dependencies ... done
 Getting requirements to build wheel ... done
 Preparing metadata (pyproject.toml) ... done
Building wheels for collected packages: mock-crickit
 Building wheel for mock-crickit (pyproject.toml) ... done
 Created wheel for mock-crickit: filename=mock_crickit-1.0-py3-none-any.whl
Successfully built mock-crickit
Installing collected packages: mock-crickit
Successfully installed mock-crickit-1.0
```

我们现在可以运行 `pip list` 来获取我们虚拟环境中安装的包列表。我们可以看到我们已经安装了 `mock-crickit` 库的 `1.0` 版本：

```
(main) robo@robopi:/tmp/mock_crickit$ pip list
Package      Version
------------ -------
mock-crickit 1.0
pip          23.1.2
setuptools   59.6.0
```

我们可以通过以下命令调用`demo`函数来验证库是否正常工作：

```
(main) robo@robopi:~$ python -m adafruit_crickit
starting mock_crickit demo...
MOCK_CRICKIT: set crickit.onboard_pixel.brightness: 0.01
MOCK_CRICKIT: call crickit.onboard_pixel.fill(16711680)
MOCK_CRICKIT: get crickit.touch_1.value: True
MOCK_CRICKIT: set crickit.dc_motor_1.throttle: 1
MOCK_CRICKIT: set crickit.dc_motor_2.throttle: -1
MOCK_CRICKIT: set crickit.servo_1.angle: 70
MOCK_CRICKIT: get crickit.servo_1.angle: 70
MOCK_CRICKIT: set crickit.servo_2.angle: 90
MOCK_CRICKIT: get crickit.servo_2.angle: 90
MOCK_CRICKIT: set crickit.servo_1.actuation_range: 142
MOCK_CRICKIT: set crickit.servo_2.actuation_range: 180
```

书中的项目可以使用这个库在各种硬件上执行。附录 A 中提到的操纵杆硬件可以与任何运行 Linux 的笔记本电脑或台式计算机一起使用。此外，任何标准摄像头都可以替代书中的树莓派摄像头模块使用，无需对书中的代码进行任何修改。这使得计算机视觉功能能够实现人脸和二维码检测。
