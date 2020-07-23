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

## pandas入门和实战

### pandas数据结构
- pandas两个基本的数据结构： Series和DataFrame
- Series数据结构类似于一维数组，由一组数据和对应的索引组成
```
from pandas import Series, DataFrame
import pandas as pd

obj = Series([1,-2,3,-4])

import pandas as pd

>>> from pandas import Series, DataFrame

obj = Series([1,-2,3,-4])
>>> obj
0 1
1 -2
2 3
3 -4
dtype: int64

obj2 = Series([1,-2,3,-4], index=['a', 'b', 'c', 'd'])
>>> obj2
a 1
b -2
c 3
d -4
dtype: int64

>>> obj2.values
array([ 1, -2, 3, -4], dtype=int64)
>>> obj2.index
Index(['a', 'b', 'c', 'd'], dtype='object')

obj2['b']
obj2[['b', 'c']]
```
- 支持运算
```
obj2[obj2<0]
obj2 * 2
np.abs(obj2)
```
```
data = {'张三': 92,
        '李四': 78,
        ‘王五’: 68,
        '小明': 82 }
obj3 = Series(data)

names = ['张三', '李四'， ‘王五’, '小明']
obj4 = Series(data, index=names)
obj4.name = 'math' # Series对象和索引都有name属性
obj4.index.name = 'students'
```
- DataFrame 数据结构为二维表格结构，类比excel
```
data = {
    'name':['张三', '李四', '王五', '小明'],
    'sex': ['female', 'female', 'male', 'male'],
    'year':[2001, 2001, 2003, 2002],
    'city':['北京', '上海', '广州', '北京']
}
df = DataFrame(data)
>>> df
name sex year city
0 张三 female 2001 北京
1 李四 female 2001 上海
2 王五 male 2003 广州
3 小明 male 2002 北京

df = DataFrame(data, columns=['name', 'sex', 'year', 'city']) # 通过columns指定列索引的排列顺序
df = DataFrame(data, columns=['name', 'sex', 'year', 'city'], index=['a', 'b', 'c', 'd']) 
# 使用其他数据作为行索引
```
```
data2 = {'sex':{'张三': 'female', '李四': 'female', '王五': 'male'},
         'city':{'张三': '北京', '李四': '上海', '王五': '广州'}}
df2 = DataFrame(data2)
>>> df1
      sex   city
张三 female  北京
李四 female  上海
王五 male    广州
```
```
df.index.name = 'id'
df.columns.name = 'std_info'
>>> df
std_info name sex year city
id
0 张三 female 2001 北京
1 李四 female 2001 上海
2 王五 male 2003 广州
3 小明 male 2002 北京

>>> df.values  # values属性可将DataFrame数据转换为二维数组
array([['张三 ', 'female', 2001, '北京 '],
       ['李四 ', 'female', 2001, '上海 '],
       ['王五 ', 'male', 2003, '广州 '],
	   ['小明 ', 'male', 2002, '北京 ']], dtype=object)
```
- Series和DataFrame的行列索引都是索引对象, 管理轴标签和元数据, 不可修改

### pandas索引操作
- 重新索引, 不是给索引重命名, 而是对索引重新排序, 如果某个索引值不存在, 就会引入缺失值
```
obj = Series([1, -2, 3, -4], index = ['b', 'a', 'c', 'd'])
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
>>> obj
b 1
a -2
c 3
d -4
dtype: int64
>>> obj2
a -2.0
b 1.0
c 3.0
d -4.0
e NaN
dtype: float64
```
- 如需对插入的缺失值进行填充, 可通过method函数实现, 参数值为ffill或pad时向前填充, 参数值为bfill或backfill时为向后填充
```
obj = Series([1, -2, 3, -4], index=[0, 2, 3, 5])
obj2 = obj.reindex(range(6), method='ffill')
>>> obj
0 1
2 -2
3 3
5 -4
dtype: int64
>>> obj2
0 1
1 1
2 -2
3 3
4 3
5 -4
dtype: int64
```
- DataFrame数据的行列索引都可以重新索引
```
df = DataFrame(np.range(9).reshape(3,3), index=['a', 'c', 'd'], columns=['name', 'id', 'sex'])
df2 = df.reindex(['a', 'b', 'c', 'd'])
df3 = df.reindex(columns=['name', 'year', 'id'], fill_value=0) # full _value填充值
```
- 更换索引 `set_index`, `reset_index`
```
df2 = df.set_index('name')
df3 = df2.reset_index()
>>> df
city name sex year
0 北 京 张 三 female 2001
1 上 海 李 四 female 2001
2 广 州 王 五 male 2003
3 北 京 小 明 male 2002
>>> df2
city sex year
name
张 三 北 京 female 2001
李 四 上 海 female 2001
王 五 广 州 male 2003
小 明 北 京 male 2002
>>> df3
name city sex year
0 张 三 北 京 female 2001
1 李 四 上 海 female 2001
2 王 五 广 州 male 2003
3 小 明 北 京 male 2002
```
- 排序, 行索引会改变
```
data = {'name': ['张三', '李四', '王五', '小明'],
        'grade': [68, 78, 63, 92]}
df = DataFrame(data)
df2 = df.sort_values(by='grade')
>>> df
name grade
0 张 三 68
1 李 四 78
2 王 五 63
3 小 明 92
>>> df2
name grade
2 王 五 63
0 张 三 68
1 李 四 78
3 小 明 92
```
- 原索引可用drop参数进行删除
```
df3 = df2.reset_index()
df4 = df2.reset_index(drop=True)
>>> df
name grade
0 张 三 68
1 李 四 78
2 王 五 63
3 小 明 92
>>> df2
name grade
2 王 五 63
0 张 三 68
1 李 四 78
3 小 明 92
>>> df3
index name grade
0 2 王 五 63
1 0 张 三 68
2 1 李 四 78
3 3 小 明 92
>>> df4
name grade
0 王 五 63
1 张 三 68
2 李 四 78
3 小 明 92
```
- 索引和选取
- Series切片
```
obj[0:2]
obj['a':'c']
```
- DataFrame, 选取列不能使用切片, 选取行可用切片
```
df['city']
df.name
df[['city', 'sex']]
```
```
df2 = df.set_index('name')
df2[0:2]
df2['李四': '王五']
```
- 想获取单独的几行, 通过loc和iloc方法实现. loc方法是安行索引标签选取数据; iloc方法是按行索引位置选取数据
```
df2.loc['张三']
df2.loc[['张三', '王五']]
df2.iloc[1]
df2.iloc[[1, 3]]
```
- 选取行和列`.ix`
```
df2.ix[['张三', '王五'], 0:2]
df2.ix[:,['sex', 'year']]
df2.ix[[1,3], :]
```
- 布尔选择
```
df2['sex'] == 'female'
df2[df2['sex'] == 'female']
df2[(df2['sex']=='female') & (df2['city']=='北京')]
```
- 操作行和列
- 增加行
```
new_data = {'city': '武汉', 
            'name': '小李', 
            'sex': 'male', 
            'year': 2002}
df = df.append(new_data, ignore_index=True) # 忽略索引值
>>> df
name sex year city
0 张 三 female 2001 北 京
1 李 四 female 2001 上 海
2 王 五 male 2003 广 州
3 小 明 male 2002 北 京
4 小 李 male 2002 武 汉
```
- 增加列
```
df['class'] = 2018
df['math'] = [92,78,63,85,56]
>>> df
name sex year city class math
0 张 三 female 2001 北 京 2018 92
1 李 四 female 2001 上 海 2018 78
2 王 五 male 2003 广 州 2018 63
3 小 明 male 2002 北 京 2018 85
4 小 李 male 2002 武 汉 2018 56
```
- 删除
```
new_df= df.drop(2) # 删除行
>>> new_df
name sex year city class math
0 张 三 female 2001 北 京 2018 92
1 李 四 female 2001 上 海 2018 78
3 小 明 male 2002 北 京 2018 85
4 小 李 male 2002 武 汉 2018 56

new_df = new_df.drop('class', axis=1) # 删除列
>>> new_df
name sex year city math
0 张 三 female 2001 北 京 92
1 李 四 female 2001 上 海 78
3 小 明 male 2002 北 京 85
4 小 李 male 2002 武 汉 56
```
- 修改(行列索引标签的修改`.rename`)
```
new_df.rename(index={3:2, 4:3}, columns={'math':'Math'}, inplace=True) # inplace可在原数据上修改
>>> new_df
name sex year city Math
0 张 三 female 2001 北 京 92
1 李 四 female 2001 上 海 78
2 小 明 male 2002 北 京 85
3 小 李 male 2002 武 汉 56
```

