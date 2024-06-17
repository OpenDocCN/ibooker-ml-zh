# 第二十章 PyTorch 中的张量

# 20.0 简介

就像 NumPy 是机器学习堆栈中数据操作的基础工具一样，PyTorch 是深度学习堆栈中处理张量的基础工具。在深入学习之前，我们应该熟悉 PyTorch 张量，并创建许多与 NumPy 中执行的操作类似的操作（见 第一章）。

虽然 PyTorch 只是多个深度学习库之一，但在学术界和工业界都非常流行。PyTorch 张量与 NumPy 数组非常相似。然而，它们还允许我们在 GPU（专门用于深度学习的硬件）上执行张量操作。在本章中，我们将熟悉 PyTorch 张量的基础知识和许多常见的低级操作。

# 20.1 创建张量

## 问题

您需要创建一个张量。

## 解决方案

使用 PyTorch 创建张量：

```py
# Load library
import torch

# Create a vector as a row
tensor_row = torch.tensor([1, 2, 3])

# Create a vector as a column
tensor_column = torch.tensor(
    [
        [1],
        [2],
        [3]
    ]
)
```

## 讨论

PyTorch 中的主要数据结构是张量，在许多方面，张量与多维 NumPy 数组（见 第一章）完全相同。就像向量和数组一样，这些张量可以水平（即行）或垂直（即列）表示。

## 参见

+   [PyTorch 文档：张量](https://oreil.ly/utaTD)

# 20.2 从 NumPy 创建张量

## 问题

您需要从 NumPy 数组创建 PyTorch 张量。

## 解决方案

使用 PyTorch 的 `from_numpy` 函数：

```py
# Import libraries
import numpy as np
import torch

# Create a NumPy array
vector_row = np.array([1, 2, 3])

# Create a tensor from a NumPy array
tensor_row = torch.from_numpy(vector_row)
```

## 讨论

正如我们所看到的，PyTorch 在语法上与 NumPy 非常相似。此外，它还允许我们轻松地将 NumPy 数组转换为可以在 GPU 和其他加速硬件上使用的 PyTorch 张量。在撰写本文时，PyTorch 文档中频繁提到 NumPy，并且 PyTorch 本身甚至提供了一种使 PyTorch 张量和 NumPy 数组可以共享内存以减少开销的方式。

## 参见

+   [PyTorch 文档：与 NumPy 的桥接](https://oreil.ly/zEJo6)

# 20.3 创建稀疏张量

## 问题

给定数据，其中非零值非常少，您希望以张量的方式高效表示它。

## 解决方案

使用 PyTorch 的 `to_sparse` 函数：

```py
# Import libraries
import torch

# Create a tensor
tensor = torch.tensor(
[
[0, 0],
[0, 1],
[3, 0]
]
)

# Create a sparse tensor from a regular tensor
sparse_tensor = tensor.to_sparse()
```

## 讨论

稀疏张量是表示由大多数 0 组成的数据的内存高效方法。在 第一章 中，我们使用 `scipy` 创建了一个压缩稀疏行（CSR）矩阵，它不再是 NumPy 数组。

`torch.Tensor` 类允许我们使用同一个对象创建常规矩阵和稀疏矩阵。如果我们检查刚刚创建的两个张量的类型，我们可以看到它们实际上都属于同一类：

```py
print(type(tensor))
print(type(sparse_tensor))
```

```py
<class 'torch.Tensor'>
<class 'torch.Tensor'>
```

## 参见

+   [PyTorch 文档：稀疏张量](https://oreil.ly/8J3IO)

# 20.4 在张量中选择元素

## 问题

我们需要选择张量的特定元素。

## 解决方案

使用类似于 NumPy 的索引和切片返回元素：

```py
# Load library
import torch

# Create vector tensor
vector = torch.tensor([1, 2, 3, 4, 5, 6])

# Create matrix tensor
matrix = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
)

# Select third element of vector
vector[2]
```

```py
tensor(3)
```

```py
# Select second row, second column
matrix[1,1]
```

```py
tensor(5)
```

## 讨论

像 NumPy 数组和 Python 中的大多数内容一样，PyTorch 张量也是从零开始索引的。索引和切片也都受支持。一个关键区别是，对 PyTorch 张量进行索引以返回单个元素仍然会返回一个张量，而不是对象本身的值（该值将是整数或浮点数）。切片语法也与 NumPy 相同，并将以张量对象的形式返回：

```py
# Select all elements of a vector
vector[:]
```

```py
array([1, 2, 3, 4, 5, 6])
```

```py
# Select everything up to and including the third element
vector[:3]
```

```py
tensor([1, 2, 3])
```

```py
# Select everything after the third element
vector[3:]
```

```py
tensor([4, 5, 6])
```

```py
# Select the last element
vector[-1]
```

```py
tensor(6)
```

```py
# Select the first two rows and all columns of a matrix
matrix[:2,:]
```

```py
tensor([[1, 2, 3],
       [4, 5, 6]])
```

```py
# Select all rows and the second column
matrix[:,1:2]
```

```py
tensor([[2],
       [5],
       [8]])
```

一个关键区别是，PyTorch 张量在切片时不支持负步长。因此，尝试使用切片反转张量会产生错误：

```py
# Reverse the vector
vector[::-1]
```

```py
ValueError: step must be greater than zero
```

相反，如果我们希望反转张量，我们可以使用 `flip` 方法：

```py
vector.flip(dims=(-1,))
```

```py
tensor([6, 5, 4, 3, 2, 1])
```

## 参见

+   [PyTorch 文档：张量操作](https://oreil.ly/8-xj7)

# 20.5 描述张量

## 问题

您想描述张量的形状、数据类型和格式以及它所使用的硬件。

## 解决方案

检查张量的 `shape`、`dtype`、`layout` 和 `device` 属性：

```py
# Load library
import torch

# Create a tensor
tensor = torch.tensor([[1,2,3], [1,2,3]])

# Get the shape of the tensor
tensor.shape
```

```py
torch.Size([2, 3])
```

```py
# Get the data type of items in the tensor
tensor.dtype
```

```py
torch.int64
```

```py
# Get the layout of the tensor
tensor.layout
```

```py
torch.strided
```

```py
# Get the device being used by the tensor
tensor.device
```

```py
device(type='cpu')
```

## 讨论

PyTorch 张量提供了许多有用的属性，用于收集关于给定张量的信息，包括：

形状

返回张量的维度

Dtype

返回张量中对象的数据类型

布局

返回内存布局（最常见的是用于稠密张量的 `strided`）

设备

返回张量存储的硬件（CPU/GPU）

再次，张量与数组的主要区别在于 *设备* 这样的属性，因为张量为我们提供了像 GPU 这样的硬件加速选项。

# 20.6 对元素应用操作

## 问题

您想对张量中的所有元素应用操作。

## 解决方案

利用 PyTorch 进行 *广播*：

```py
# Load library
import torch

# Create a tensor
tensor = torch.tensor([1, 2, 3])

# Broadcast an arithmetic operation to all elements in a tensor
tensor * 100
```

```py
tensor([100, 200, 300])
```

## 讨论

PyTorch 中的基本操作将利用广播并行化，使用像 GPU 这样的加速硬件。这对于 Python 中支持的数学运算符（+、-、×、/）和 PyTorch 内置函数是真实的。与 NumPy 不同，PyTorch 不包括用于在张量上应用函数的 `vectorize` 方法。然而，PyTorch 配备了所有必要的数学工具，以分发和加速深度学习工作流程中所需的常规操作。

## 参见

+   [PyTorch 文档：广播语义](https://oreil.ly/NsPpa)

+   [PyTorch 中的向量化和广播](https://oreil.ly/dfzIJ)

# 20.7 查找最大值和最小值

## 问题

您需要在张量中找到最大值或最小值。

## 解决方案

使用 PyTorch 的 `max` 和 `min` 方法：

```py
# Load library
import torch

# Create a tensor
torch.tensor([1,2,3])

# Find the largest value
tensor.max()
```

```py
tensor(3)
```

```py
# Find the smallest value
tensor.min()
```

```py
tensor(1)
```

## 讨论

张量的 `max` 和 `min` 方法帮助我们找到该张量中的最大值或最小值。这些方法在多维张量上同样适用：

```py
# Create a multidimensional tensor
tensor = torch.tensor([[1,2,3],[1,2,5]])

# Find the largest value
tensor.max()
```

```py
tensor(5)
```

# 20.8 改变张量的形状

## 问题

您希望改变张量的形状（行数和列数）而不改变元素值。

## 解决方案

使用 PyTorch 的 `reshape` 方法：

```py
# Load library
import torch

# Create 4x3 tensor
tensor = torch.tensor([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9],
                       [10, 11, 12]])

# Reshape tensor into 2x6 tensor
tensor.reshape(2, 6)
```

```py
tensor([[ 1,  2,  3,  4,  5,  6],
        [ 7,  8,  9, 10, 11, 12]])
```

## 讨论

在深度学习领域中，操作张量的形状可能很常见，因为神经网络中的神经元通常需要具有非常特定形状的张量。由于给定神经网络中的神经元之间所需的张量形状可能会发生变化，因此了解深度学习中输入和输出的低级细节是很有好处的。

# 20.9 转置张量

## 问题

您需要转置一个张量。

## 解决方案

使用 `mT` 方法：

```py
# Load library
import torch

# Create a two-dimensional tensor
tensor = torch.tensor([[[1,2,3]]])

# Transpose it
tensor.mT
```

```py
tensor([[1],
        [2],
        [3]])
```

## 讨论

使用 PyTorch 进行转置与 NumPy 略有不同。用于 NumPy 数组的 `T` 方法仅支持二维张量，在 PyTorch 中对于其他形状的张量时，该方法在写作时已被弃用。用于转置批处理张量的 `mT` 方法更受欢迎，因为它适用于超过两个维度的张量。

除了使用 `permute` 方法之外，还可以使用 PyTorch 中的另一种方式来转置任意形状的张量：

```py
tensor.permute(*torch.arange(tensor.ndim - 1, -1, -1))
```

```py
tensor([[1],
        [2],
        [3]])
```

这种方法也适用于一维张量（其中转置张量的值与原始张量相同）。

# 20.10 张量展平

## 问题

您需要将张量转换为一维。

## 解决方案

使用 `flatten` 方法：

```py
# Load library
import torch

# Create tensor
tensor = torch.tensor([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])

# Flatten tensor
tensor.flatten()
```

```py
tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
```

## 讨论

张量展平是将多维张量降维为一维的一种有用技术。

# 20.11 计算点积

## 问题

您需要计算两个张量的点积。

## 解决方案

使用 `dot` 方法：

```py
# Load library
import torch

# Create one tensor
tensor_1 = torch.tensor([1, 2, 3])

# Create another tensor
tensor_2 = torch.tensor([4, 5, 6])

# Calculate the dot product of the two tensors
tensor_1.dot(tensor_2)
```

```py
tensor(32)
```

## 讨论

计算两个张量的点积是深度学习空间以及信息检索空间中常用的操作。您可能还记得本书中我们使用两个向量的点积执行基于余弦相似度的搜索。在 PyTorch 上使用 GPU（而不是在 CPU 上使用 NumPy 或 scikit-learn）执行此操作可以在信息检索问题上获得显著的性能优势。

## 参见

+   [使用 PyTorch 进行向量化和广播](https://oreil.ly/lIjtB)

# 20.12 乘法张量

## 问题

您需要将两个张量相乘。

## 解决方案

使用基本的 Python 算术运算符：

```py
# Load library
import torch

# Create one tensor
tensor_1 = torch.tensor([1, 2, 3])

# Create another tensor
tensor_2 = torch.tensor([4, 5, 6])

# Multiply the two tensors
tensor_1 * tensor_2
```

```py
tensor([ 4, 10, 18])
```

## 讨论

PyTorch 支持基本算术运算符，如 ×、+、- 和 /。虽然在深度学习中，张量乘法可能是最常用的操作之一，但了解张量也可以进行加法、减法和除法是很有用的。

将一个张量加到另一个张量中：

```py
tensor_1+tensor_2
```

```py
tensor([5, 7, 9])
```

从一个张量中减去另一个张量：

```py
tensor_1-tensor_2
```

```py
tensor([-3, -3, -3])
```

将一个张量除以另一个张量：

```py
tensor_1/tensor_2
```

```py
tensor([0.2500, 0.4000, 0.5000])
```
