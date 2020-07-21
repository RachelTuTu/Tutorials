# 从零开始学python数据分析

## numpy入门和实战

### 创建ndarray数组

```
import numpy as np
d1 = [1,2,3,4] # 列表
d2 = (1,2,3,4) # 元组
d3 = [[1,2,3,4], [5,6,7,8]] # 多维数据

arr1 = np.array(d1)
arr2 = np.array(d2)
arr3 = np.array(d3)

arr1.shape
arr1.dtype
```

```
np.zeros(10) 
np.zeros((3,4))
np.ones(10)
np.ones((3,4))

np.ones_like(arr1) # 以另一个数组为参看，根据其形状和dtype创建一个全1数组
np.zeros_like(arr1)
np.empty_like(arr1)
np.eye(10)
np.identity(10)
```

### ndarray对象属性
```
arr1.ndim # 秩，即数据轴的个数
arr1.shape
arr1.size
arr1.dtype
```

### ndarray数据类型
```
arr1 = np.arange(5)
arr2 = np.arange(5, dtype='float64')
arr3 = arr1.astype(np.float64)
arr4 = arr1.astype('string_')
```

### 数组变换
```
arr = np.arange(9)
arr2 = arr.reshape(3,3)
arr3 = np.arange(9).reshape(3,3)
arr4 = arr2.ravel() # 数据散开
arr5 = arr2.flatten() # 数据扁平化
```

- 数组合并
```
arr1 = np.arange(12).reshape(3,4)
arr2 = np.arange(12,24).reshape(3,4)
np.concatenate([arr1, arr2], axis=0) # 按行拼接
np.concatenate([arr1, arr2], axis=1) # 按列拼接
```

```
np.vstack((arr1, arr2)) # 按行拼接
np.hstack((arr1, arr2)) # 按列拼接
```
- 数组拆分
```
arr = np.arange(12).reshape(6,2)
>> array([[ 0, 1],
[ 2, 3],
[ 4, 5],
[ 6, 7],
[ 8, 9],
[10, 11]])

np.split(arr, [2,4])
>> [array([[0, 1],
           [2, 3]]), 
    array([[4, 5],
           [6, 7]]), 
    array([[ 8, 9],
           [10, 11]])]
```
- 数组的转置和轴对换
```
arr = np.arange(12).reshape(3,4)
arr.transpose((1,0))
arr.T
```

```
arr = np.arange(16).reshape(2,2,4) # (2,2,4)
arr.swapaxes(1,2)                  # (2,4,2)
```

### numpy的随机数函数
```
arr = np.random.randint(100, 200, size=(5,4))
arr = np.random.randn(2,3,5)
arr = np.random.normal(4,5,size(3,5)) # 指定均值和标准差的正态分布
```

```
np.random.rand # 产生均匀分布的样本值
np.random.randint # 给定范围内取随机整数
np.random.randn # 产生正态分布的样本值
np.random.seed # 随机数种子
np.random.permutation # 对一个序列随机排序，不改变原数组
np.random.shuffle # 对一个序列随机排序，改变原数组
np.random.uniform(low, high, size) # 产生均匀分布的数组
np.random.normal(loca, scale, size) # 产生具有正态分布的数组， loc均值，scale标准差
np.random.poisson(lam, size) # 产生具有泊松分布的数组， lam表示随机事件发生率
```

### 花式索引
```
arr = np.arange(12).reshape(4,3)
arr[[3,2]][:,[2,1]]
arr[np.ix_([3,2], [2,1])]

array([[11, 10],
       [ 8, 7]])
```

### 数组的运算
```
arr = np.random.randn(3,3)
arr1= np.random.randn(3,3)
arr*10
arr*arr
arr-arr
np.abs(arr)
np.square(arr)
np.modf(arr) # 输出arr的小数部分和整数部分
np.add(arr1,arr2)
np.minimum(arr1,arr2)
```

### 条件运算
```
arr1 = np.array([1,2,3,4])
arr2 = np.array([5,6,7,8])
cond = np.array([True, False, False. True])
```
- 通过cond的值选取arr1和arr2的值，当cond为True时，选择arr1的值，否则选择arr2的值
```
result = [(x if c else y) for x, y, c in zip(arr1, arr2, cond)]  # 慢

result  = np.where(cond, arr1, arr2)  # 快
```

```
arr = np.random.randn(4,4)
new_arr = np.where(arr>0, 1, -1)
```

### 统计运算
```
arr = np.random.randn(4,4)
arr.sum(),  arr.sum(0)
arr.mean(), arr.mean(axis=1)
arr.std()
```
```
arr.any() # 用于测试数组中是否存在一个或多个True
arr.all() # 用于检测数组中的所有值是否为True
```

### 排序
```
arr = np.random.rand(10)
arr.sort()
```
```
arr = np.random.randn(5,3)
arr.sort(1) # 通过指定轴方向进行排序
```

### 集合运算
```
np.unique(arr)
```

- `np.in1d`用于测试几个数组中是否包含相同的值，返回布尔值数组
```
arr = np.array(2,3,5,7)
np.in1d(arr, [2,7])
>> array([True, False, False, True], dtype=bool)
```
```
np.intersect1d(x, y)  # 交集
np.union(x, y)  # 并集
np.in1d(x， y)  # x的元素是否在y中
np.setdiff1d（x, y） # 集合的差
np.setxor1d(x, y)  # 集合取返
```

### 线性代数
```
arr1 = np.array([[1,2,3], [4,5,6]])
arr2 = np.arange(9).reshape(3,3)
np.dot(arr1, arr2) # 矩阵乘法
```
- 更多的矩阵计算可通过`np.linalg`模块完成
```
from numpy.linalg import det
arr = np.array([[1,2], [3,4]])
det(arr)  # 行列式
```

### 数组的存储/读取
```
arr = np.arange(12).shape(3,4)
np.savetxt('aa.csv', arr, fmt='%d', delimiter=',')
np.loadtxt('aa.csv', delimiter=',')
```

### 综合示例--图像变换
```
from PIL import Image
import numpy as np
im = np.array(Image.open('1.jpg'))
print(im.shape, im.dtype)

b = [255, 255, 255] - im 
new_im = Image.fromarray(b.astype('uint8'))
new_im.save('2.jpg')
```

