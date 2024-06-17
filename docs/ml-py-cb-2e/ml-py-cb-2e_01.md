# 第一章：在 NumPy 中处理向量、矩阵和数组

# 1.0 介绍

NumPy 是 Python 机器学习堆栈的基础工具。NumPy 允许在机器学习中经常使用的数据结构（向量、矩阵和张量）上进行高效操作。虽然本书的重点不是 NumPy，但在接下来的章节中会经常出现。本章涵盖了我们在机器学习工作流中可能遇到的最常见的 NumPy 操作。

# 1.1 创建向量

## 问题

您需要创建一个向量。

## 解决方案

使用 NumPy 创建一维数组：

```py
# Load library
import numpy as np

# Create a vector as a row
vector_row = np.array([1, 2, 3])

# Create a vector as a column
vector_column = np.array([[1],
                          [2],
                          [3]])
```

## 讨论

NumPy 的主要数据结构是多维数组。向量只是一个单维数组。要创建向量，我们只需创建一个一维数组。就像向量一样，这些数组可以水平表示（即行）或垂直表示（即列）。

## 参见

+   [向量, Math Is Fun](https://oreil.ly/43I-b)

+   [欧几里得向量, 维基百科](https://oreil.ly/er78t)

# 1.2 创建矩阵

## 问题

您需要创建一个矩阵。

## 解决方案

使用 NumPy 创建二维数组：

```py
# Load library
import numpy as np

# Create a matrix
matrix = np.array([[1, 2],
                   [1, 2],
                   [1, 2]])
```

## 讨论

要创建矩阵，我们可以使用 NumPy 的二维数组。在我们的解决方案中，矩阵包含三行和两列（一列为 1，一列为 2）。

实际上，NumPy 有一个专用的矩阵数据结构：

```py
matrix_object = np.mat([[1, 2],
                        [1, 2],
                        [1, 2]])
```

```py
matrix([[1, 2],
        [1, 2],
        [1, 2]])
```

然而，矩阵数据结构由于两个原因而不推荐使用。首先，数组是 NumPy 的事实标准数据结构。其次，绝大多数 NumPy 操作返回数组，而不是矩阵对象。

## 参见

+   [矩阵, 维基百科](https://oreil.ly/tnRJw)

+   [矩阵, Wolfram MathWorld](https://oreil.ly/76jUS)

# 1.3 创建稀疏矩阵

## 问题

鉴于数据中只有很少的非零值，您希望以高效的方式表示它。

## 解决方案

创建稀疏矩阵：

```py
# Load libraries
import numpy as np
from scipy import sparse

# Create a matrix
matrix = np.array([[0, 0],
                   [0, 1],
                   [3, 0]])

# Create compressed sparse row (CSR) matrix
matrix_sparse = sparse.csr_matrix(matrix)
```

## 讨论

在机器学习中经常遇到的情况是有大量数据；然而，数据中大多数元素都是零。例如，想象一下一个矩阵，其中列是 Netflix 上的每部电影，行是每个 Netflix 用户，值是用户观看该特定电影的次数。这个矩阵将有成千上万的列和数百万的行！然而，由于大多数用户不会观看大多数电影，大多数元素将为零。

*稀疏矩阵* 是一个大部分元素为 0 的矩阵。稀疏矩阵仅存储非零元素，并假设所有其他值都为零，从而显著节省计算资源。在我们的解决方案中，我们创建了一个具有两个非零值的 NumPy 数组，然后将其转换为稀疏矩阵。如果查看稀疏矩阵，可以看到只存储了非零值：

```py
# View sparse matrix
print(matrix_sparse)
```

```py
  (1, 1)    1
  (2, 0)    3
```

有许多类型的稀疏矩阵。然而，在*压缩稀疏行*（CSR）矩阵中，`(1, 1)` 和 `(2, 0)` 表示非零值 `1` 和 `3` 的（从零开始计数的）索引。例如，元素 `1` 在第二行第二列。如果我们创建一个具有更多零元素的更大矩阵，然后将其与我们的原始稀疏矩阵进行比较，我们可以看到稀疏矩阵的优势：

```py
# Create larger matrix
matrix_large = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [3, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# Create compressed sparse row (CSR) matrix
matrix_large_sparse = sparse.csr_matrix(matrix_large)
```

```py
# View original sparse matrix
print(matrix_sparse)
```

```py
  (1, 1)    1
  (2, 0)    3
```

```py
# View larger sparse matrix
print(matrix_large_sparse)
```

```py
  (1, 1)    1
  (2, 0)    3
```

正如我们所见，尽管在更大的矩阵中添加了更多的零元素，但其稀疏表示与我们原始的稀疏矩阵完全相同。也就是说，添加零元素并没有改变稀疏矩阵的大小。

正如前面提到的，稀疏矩阵有许多不同的类型，如压缩稀疏列、列表列表和键字典。虽然解释这些不同类型及其影响超出了本书的范围，但值得注意的是，虽然没有“最佳”稀疏矩阵类型，但它们之间存在显著差异，我们应该意识到为什么选择一种类型而不是另一种类型。

## 另请参阅

+   [SciPy 文档：稀疏矩阵](https://oreil.ly/zBBRB)

+   [存储稀疏矩阵的 101 种方法](https://oreil.ly/sBQhN)

# 1.4 预分配 NumPy 数组

## 问题

您需要预先分配给定大小的数组，并使用某些值。

## 解决方案

NumPy 具有使用 0、1 或您选择的值生成任意大小向量和矩阵的函数：

```py
# Load library
import numpy as np

# Generate a vector of shape (1,5) containing all zeros
vector = np.zeros(shape=5)

# View the matrix
print(vector)
```

```py
array([0., 0., 0., 0., 0.])
```

```py
# Generate a matrix of shape (3,3) containing all ones
matrix = np.full(shape=(3,3), fill_value=1)

# View the vector
print(matrix)
```

```py
array([[1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.]])
```

## 讨论

使用预填充数据生成数组对于许多目的非常有用，例如使代码更具性能或使用合成数据来测试算法。在许多编程语言中，预先分配一个带有默认值（例如 0）的数组被认为是常见做法。

# 1.5 选择元素

## 问题

您需要在向量或矩阵中选择一个或多个元素。

## 解决方案

NumPy 数组使得选择向量或矩阵中的元素变得很容易：

```py
# Load library
import numpy as np

# Create row vector
vector = np.array([1, 2, 3, 4, 5, 6])

# Create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Select third element of vector
vector[2]
```

```py
3
```

```py
# Select second row, second column
matrix[1,1]
```

```py
5
```

## 讨论

像大多数 Python 中的事物一样，NumPy 数组是从零开始索引的，这意味着第一个元素的索引是 0，而不是 1。除此之外，NumPy 提供了大量方法来选择（即索引和切片）数组中的元素或元素组：

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
array([1, 2, 3])
```

```py
# Select everything after the third element
vector[3:]
```

```py
array([4, 5, 6])
```

```py
# Select the last element
vector[-1]
```

```py
6
```

```py
# Reverse the vector
vector[::-1]
```

```py
array([6, 5, 4, 3, 2, 1])
```

```py
# Select the first two rows and all columns of a matrix
matrix[:2,:]
```

```py
array([[1, 2, 3],
       [4, 5, 6]])
```

```py
# Select all rows and the second column
matrix[:,1:2]
```

```py
array([[2],
       [5],
       [8]])
```

# 1.6 描述矩阵

## 问题

您想要描述矩阵的形状、大小和维度。

## 解决方案

使用 NumPy 对象的 `shape`、`size` 和 `ndim` 属性：

```py
# Load library
import numpy as np

# Create matrix
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

# View number of rows and columns
matrix.shape
```

```py
(3, 4)
```

```py
# View number of elements (rows * columns)
matrix.size
```

```py
12
```

```py
# View number of dimensions
matrix.ndim
```

```py
2
```

## 讨论

这可能看起来很基础（而且确实如此）；然而，一次又一次地，检查数组的形状和大小都是非常有价值的，无论是为了进一步的计算还是仅仅作为操作后的直觉检查。

# 1.7 对每个元素应用函数

## 问题

您想将某些函数应用于数组中的所有元素。

## 解决方案

使用 NumPy 的 `vectorize` 方法：

```py
# Load library
import numpy as np

# Create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Create function that adds 100 to something
add_100 = lambda i: i + 100

# Create vectorized function
vectorized_add_100 = np.vectorize(add_100)

# Apply function to all elements in matrix
vectorized_add_100(matrix)
```

```py
array([[101, 102, 103],
       [104, 105, 106],
       [107, 108, 109]])
```

## 讨论

NumPy 的`vectorize`方法将一个函数转换为可以应用于数组或数组切片的所有元素的函数。值得注意的是，`vectorize`本质上是对元素的`for`循环，不会提高性能。此外，NumPy 数组允许我们在数组之间执行操作，即使它们的维度不同（这称为*广播*）。例如，我们可以使用广播创建一个更简单的版本：

```py
# Add 100 to all elements
matrix + 100
```

```py
array([[101, 102, 103],
       [104, 105, 106],
       [107, 108, 109]])
```

广播不适用于所有形状和情况，但它是在 NumPy 数组的所有元素上应用简单操作的常见方法。

# 1.8 查找最大值和最小值

## 问题

您需要在数组中找到最大或最小值。

## 解决方案

使用 NumPy 的`max`和`min`方法：

```py
# Load library
import numpy as np

# Create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Return maximum element
np.max(matrix)
```

```py
9
```

```py
# Return minimum element
np.min(matrix)
```

```py
1
```

## 讨论

我们经常想知道数组或数组子集中的最大值和最小值。这可以通过`max`和`min`方法来实现。使用`axis`参数，我们还可以沿着特定轴应用操作：

```py
# Find maximum element in each column
np.max(matrix, axis=0)
```

```py
array([7, 8, 9])
```

```py
# Find maximum element in each row
np.max(matrix, axis=1)
```

```py
array([3, 6, 9])
```

# 1.9 计算平均值、方差和标准差

## 问题

您希望计算数组的一些描述性统计信息。

## 解决方案

使用 NumPy 的`mean`、`var`和`std`：

```py
# Load library
import numpy as np

# Create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Return mean
np.mean(matrix)
```

```py
5.0
```

```py
# Return variance
np.var(matrix)
```

```py
6.666666666666667
```

```py
# Return standard deviation
np.std(matrix)
```

```py
2.5819888974716112
```

## 讨论

就像使用`max`和`min`一样，我们可以轻松地获得关于整个矩阵的描述性统计信息，或者沿着单个轴进行计算：

```py
# Find the mean value in each column
np.mean(matrix, axis=0)
```

```py
array([ 4.,  5.,  6.])
```

# 1.10 重新塑形数组

## 问题

您希望更改数组的形状（行数和列数），而不更改元素值。

## 解决方案

使用 NumPy 的`reshape`：

```py
# Load library
import numpy as np

# Create 4x3 matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]])

# Reshape matrix into 2x6 matrix
matrix.reshape(2, 6)
```

```py
array([[ 1,  2,  3,  4,  5,  6],
       [ 7,  8,  9, 10, 11, 12]])
```

## 讨论

`reshape`允许我们重构一个数组，以便我们保持相同的数据但将其组织为不同数量的行和列。唯一的要求是原始矩阵和新矩阵的形状包含相同数量的元素（即，大小相同）。我们可以使用`size`来查看矩阵的大小：

```py
matrix.size
```

```py
12
```

`reshape`中一个有用的参数是`-1`，它实际上意味着“需要多少就多少”，因此`reshape(1, -1)`意味着一行和所需的列数：

```py
matrix.reshape(1, -1)
```

```py
array([[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]])
```

最后，如果我们提供一个整数，`reshape`将返回一个长度为该整数的一维数组：

```py
matrix.reshape(12)
```

```py
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])
```

# 1.11 转置向量或矩阵

## 问题

您需要转置向量或矩阵。

## 解决方案

使用`T`方法：

```py
# Load library
import numpy as np

# Create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Transpose matrix
matrix.T
```

```py
array([[1, 4, 7],
       [2, 5, 8],
       [3, 6, 9]])
```

## 讨论

*转置*是线性代数中的常见操作，其中每个元素的列和行索引被交换。在线性代数课程之外通常被忽略的一个微妙的点是，从技术上讲，向量不能被转置，因为它只是一组值：

```py
# Transpose vector
np.array([1, 2, 3, 4, 5, 6]).T
```

```py
array([1, 2, 3, 4, 5, 6])
```

然而，通常将向量的转置称为将行向量转换为列向量（请注意第二对括号）或反之亦然：

```py
# Transpose row vector
np.array([[1, 2, 3, 4, 5, 6]]).T
```

```py
array([[1],
       [2],
       [3],
       [4],
       [5],
       [6]])
```

# 1.12 扁平化矩阵

## 问题

您需要将矩阵转换为一维数组。

## 解决方案

使用`flatten`方法：

```py
# Load library
import numpy as np

# Create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Flatten matrix
matrix.flatten()
```

```py
array([1, 2, 3, 4, 5, 6, 7, 8, 9])
```

## 讨论

`flatten`是一个简单的方法，将矩阵转换为一维数组。或者，我们可以使用`reshape`创建一个行向量：

```py
matrix.reshape(1, -1)
```

```py
array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
```

另一种常见的数组展平方法是`ravel`方法。与返回原始数组副本的`flatten`不同，`ravel`直接操作原始对象，因此速度稍快。此外，`ravel`还允许我们展平数组列表，而`flatten`方法则无法做到。这种操作对于展平非常大的数组和加速代码非常有用：

```py
# Create one matrix
matrix_a = np.array([[1, 2],
                     [3, 4]])

# Create a second matrix
matrix_b = np.array([[5, 6],
                     [7, 8]])

# Create a list of matrices
matrix_list = [matrix_a, matrix_b]

# Flatten the entire list of matrices
np.ravel(matrix_list)
```

```py
array([1, 2, 3, 4, 5, 6, 7, 8])
```

# 1.13 矩阵的秩

## 问题

你需要知道一个矩阵的秩。

## 解决方案

使用 NumPy 的线性代数方法`matrix_rank`：

```py
# Load library
import numpy as np

# Create matrix
matrix = np.array([[1, 1, 1],
                   [1, 1, 10],
                   [1, 1, 15]])

# Return matrix rank
np.linalg.matrix_rank(matrix)
```

```py
2
```

## 讨论

矩阵的*秩*是由其列或行张成的向量空间的维数。在 NumPy 中由于`matrix_rank`函数，计算矩阵的秩非常容易。

## 参见

+   [矩阵的秩，CliffsNotes](https://oreil.ly/Wg9ZG)

# 1.14 获取矩阵的对角线

## 问题

你需要获取一个矩阵的对角线元素。

## 解决方案

使用 NumPy 的`diagonal`：

```py
# Load library
import numpy as np

# Create matrix
matrix = np.array([[1, 2, 3],
                   [2, 4, 6],
                   [3, 8, 9]])

# Return diagonal elements
matrix.diagonal()
```

```py
array([1, 4, 9])
```

## 讨论

使用 NumPy 轻松获取矩阵的对角线元素，使用`diagonal`函数。还可以通过使用`offset`参数获取主对角线之外的对角线：

```py
# Return diagonal one above the main diagonal
matrix.diagonal(offset=1)
```

```py
array([2, 6])
```

```py
# Return diagonal one below the main diagonal
matrix.diagonal(offset=-1)
```

```py
array([2, 8])
```

# 1.15 计算一个矩阵的迹

## 问题

你需要计算一个矩阵的迹。

## 解决方案

使用`trace`：

```py
# Load library
import numpy as np

# Create matrix
matrix = np.array([[1, 2, 3],
                   [2, 4, 6],
                   [3, 8, 9]])

# Return trace
matrix.trace()
```

```py
14
```

## 讨论

一个矩阵的*迹*是其对角线元素的和，通常在机器学习方法中被广泛使用。对于给定的 NumPy 多维数组，我们可以使用`trace`函数来计算迹。或者，我们可以返回一个矩阵的对角线并计算其总和：

```py
# Return diagonal and sum elements
sum(matrix.diagonal())
```

```py
14
```

## 参见

+   [方阵的迹](https://oreil.ly/AhX1b)

# 1.16 计算点积

## 问题

你需要计算两个向量的点积。

## 解决方案

使用 NumPy 的`dot`函数：

```py
# Load library
import numpy as np

# Create two vectors
vector_a = np.array([1,2,3])
vector_b = np.array([4,5,6])

# Calculate dot product
np.dot(vector_a, vector_b)
```

```py
32
```

## 讨论

两个向量<math display="inline"><mi>a</mi></math>和<math display="inline"><mi>b</mi></math>的*点积*定义如下：

<math display="block"><mrow><munderover><mo>∑</mo> <mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow> <mi>n</mi></munderover> <msub><mi>a</mi> <mi>i</mi></msub> <msub><mi>b</mi> <mi>i</mi></msub></mrow></math>

其中<math display="inline"><msub><mi>a</mi><mi>i</mi></msub></math>是向量<math display="inline"><mi>a</mi></math>的第<math display="inline"><mi>i</mi></math>个元素，<math display="inline"><msub><mi>b</mi><mi>i</mi></msub></math>是向量<math display="inline"><mi>b</mi></math>的第<math display="inline"><mi>i</mi></math>个元素。我们可以使用 NumPy 的`dot`函数来计算点积。或者，在 Python 3.5+ 中，我们可以使用新的`@`运算符：

```py
# Calculate dot product
vector_a @ vector_b
```

```py
32
```

## 参见

+   [向量点积和向量长度，Khan Academy](https://oreil.ly/MpBt7)

+   [点积，Paul’s Online Math Notes](https://oreil.ly/EprM1)

# 1.17 矩阵的加法和减法

## 问题

你想要对两个矩阵进行加法或减法。

## 解决方案

使用 NumPy 的`add`和`subtract`：

```py
# Load library
import numpy as np

# Create matrix
matrix_a = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 2]])

# Create matrix
matrix_b = np.array([[1, 3, 1],
                     [1, 3, 1],
                     [1, 3, 8]])

# Add two matrices
np.add(matrix_a, matrix_b)
```

```py
array([[ 2,  4,  2],
       [ 2,  4,  2],
       [ 2,  4, 10]])
```

```py
# Subtract two matrices
np.subtract(matrix_a, matrix_b)
```

```py
array([[ 0, -2,  0],
       [ 0, -2,  0],
       [ 0, -2, -6]])
```

## 讨论

或者，我们可以简单地使用`+`和`–`运算符：

```py
# Add two matrices
matrix_a + matrix_b
```

```py
array([[ 2,  4,  2],
       [ 2,  4,  2],
       [ 2,  4, 10]])
```

# 1.18 矩阵的乘法

## 问题

你想要对两个矩阵进行乘法运算。

## 解决方案

使用 NumPy 的`dot`：

```py
# Load library
import numpy as np

# Create matrix
matrix_a = np.array([[1, 1],
                     [1, 2]])

# Create matrix
matrix_b = np.array([[1, 3],
                     [1, 2]])

# Multiply two matrices
np.dot(matrix_a, matrix_b)
```

```py
array([[2, 5],
       [3, 7]])
```

## 讨论

或者，在 Python 3.5+ 中我们可以使用`@`运算符：

```py
# Multiply two matrices
matrix_a @ matrix_b
```

```py
array([[2, 5],
       [3, 7]])
```

如果我们想要进行逐元素乘法，可以使用`*`运算符：

```py
# Multiply two matrices element-wise
matrix_a * matrix_b
```

```py
array([[1, 3],
       [1, 4]])
```

## 参见

+   [数组 vs. 矩阵运算，MathWorks](https://oreil.ly/_sFx5)

# 1.19 矩阵求逆

## 问题

您想要计算一个方阵的逆。

## 解决方案

使用 NumPy 的线性代数 `inv` 方法：

```py
# Load library
import numpy as np

# Create matrix
matrix = np.array([[1, 4],
                   [2, 5]])

# Calculate inverse of matrix
np.linalg.inv(matrix)
```

```py
array([[-1.66666667,  1.33333333],
       [ 0.66666667, -0.33333333]])
```

## 讨论

方阵 **A** 的逆，**A**^(–1)，是第二个矩阵，满足以下条件：

<math display="block"><mrow><mi mathvariant="bold">A</mi> <msup><mi mathvariant="bold">A</mi> <mrow><mo>-</mo><mn>1</mn></mrow></msup> <mo>=</mo> <mi mathvariant="bold">I</mi></mrow></math>

其中 **I** 是单位矩阵。在 NumPy 中，如果存在的话，我们可以使用 `linalg.inv` 计算 **A**^(–1)。为了看到这一点，我们可以将一个矩阵乘以它的逆，结果是单位矩阵：

```py
# Multiply matrix and its inverse
matrix @ np.linalg.inv(matrix)
```

```py
array([[ 1.,  0.],
       [ 0.,  1.]])
```

## 参见

+   [矩阵的逆](https://oreil.ly/SwRXC)

# 1.20 生成随机值

## 问题

您想要生成伪随机值。

## 解决方案

使用 NumPy 的 `random`：

```py
# Load library
import numpy as np

# Set seed
np.random.seed(0)

# Generate three random floats between 0.0 and 1.0
np.random.random(3)
```

```py
array([ 0.5488135 ,  0.71518937,  0.60276338])
```

## 讨论

NumPy 提供了生成随机数的多种方法，远远超出了此处所能涵盖的范围。在我们的解决方案中，我们生成了浮点数；然而，生成整数也很常见：

```py
# Generate three random integers between 0 and 10
np.random.randint(0, 11, 3)
```

```py
array([3, 7, 9])
```

或者，我们可以通过从分布中抽取数字来生成数字（请注意，这在技术上不是随机的）：

```py
# Draw three numbers from a normal distribution with mean 0.0
# and standard deviation of 1.0
np.random.normal(0.0, 1.0, 3)
```

```py
array([-1.42232584,  1.52006949, -0.29139398])
```

```py
# Draw three numbers from a logistic distribution with mean 0.0 and scale of 1.0
np.random.logistic(0.0, 1.0, 3)
```

```py
array([-0.98118713, -0.08939902,  1.46416405])
```

```py
# Draw three numbers greater than or equal to 1.0 and less than 2.0
np.random.uniform(1.0, 2.0, 3)
```

```py
array([ 1.47997717,  1.3927848 ,  1.83607876])
```

最后，有时候返回相同的随机数多次以获取可预测的、可重复的结果是有用的。我们可以通过设置伪随机生成器的“种子”（一个整数）来实现这一点。具有相同种子的随机过程将始终产生相同的输出。我们将在本书中使用种子，以确保您在书中看到的代码和在您的计算机上运行的代码产生相同的结果。
