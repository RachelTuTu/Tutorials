# 从零开始学python数据分析
- [numpy入门和实战](#jump)
- [pandas入门和实战](#jump)
- [外部数据的读取与存储](#jump)
- [数据清洗与整理](#jump)


## <span id="jump">numpy入门和实战</span>
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

## <span id="jump">pandas入门和实战</span>

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

### pandas数据运算
- 算数运算
```
obj1 = Series([3.2, 5.3, -4.4, -3.7], index=['a', 'c', 'g', 'f'])
obj2 = Series([5.0, -2, 4.4, 3.4], index=['a', 'b', 'c', 'd'])
obj1 + obj2
>>> obj1
a 3.2
c 5.3
g -4.4
f -3.4
dtype: float64
>>> obj2
a 5.0
b -2.0
c 4.4
d 3.4
dtype: float64
>>> obj1+obj2
a 8.2
b NaN
c 9.7
d NaN
f NaN
g NaN
dtype: float64
```
```
df1 = DataFrame(np.arange(9).reshape(3,3), columns=['a', 'b', 'c'], index=['apple', 'tea', 'banana'])
df2 = DataFrame(np.arange(9).reshape(3,3), columns=['a', 'b', 'd'], index=['apple', 'tea', 'coco'])
df1 + df2
>>> df1
a b c
apple 0 1 2
tea 3 4 5
banana 6 7 8
>>> df2
a b d
apple 0 1 2
tea 3 4 5
coco 6 7 8
>>> df1+df2
a b c d
apple 0.0 2.0 NaN NaN
banana NaN NaN NaN NaN
coco NaN NaN NaN NaN
tea 6.0 8.0 NaN NaN
```
- DataFrame 和 Series数据在进行运算时, 先通过Series的索引匹配到相应的DataFrame列索引上, 然后沿行向下运算(广播)
```
s = df1.loc['apple']
df1 - s
>>> s
a 0
b 1
c 2
Name: apple, dtype: int32
>>> df1-s
a b c
apple 0 0 0
tea 3 3 3
banana 6 6 6
```
- 函数应用和映射: 数据分析时可定义函数, 并应用到pandas数据中.
- map函数用在Series的每个元素中;
- apply函数用在DataFrame的行与列上;
- applymap函数用在DataFrame的每个元素上
```
data= {'fruit': ['apple', 'orange', 'grape', 'banana'],
       'price': ['25元', '42元', '36元', '14元']}
df = DataFrame(data)

def f(x):
    return x.split('元')[0]

df['price'] = df['price'].map(f)
>>> df
fruit price
0 apple 25元
1 orange 42元
2 grape 36元
3 banana 14元
>>>df
fruit price
0 apple 25
1 orange 42
2 grape 36
3 banana 14
```
```
df2 = DataFrame(np.random.randn(3,3), columns=['a', 'b', 'c'], index=['app', 'win', 'mac'])
>>> df2
a b c
app 0.133191 -1.184032 0.342867
win 1.848022 0.444231 0.112673
mac -0.829696 -0.876299 -0.111810

f = lambda x:x.max()-x.min() # lambda为匿名函数,和定义好的函数一样,可以节省代码量
df2.apply(f)
>>> df2.apply(f)
a 2.677718
b 1.628263
c 0.454678
dtype: float64
```
```
df2 = DataFrame(np.random.randn(3,3), columns=['a', 'b', 'c'], index=['app', 'win', 'mac'])
df2.applymap(lambda x:'%.2f' %x)
>>>df2
a b c
app 0.07 -0.24 -0.65
win 1.69 -0.83 -1.49
mac 1.77 -0.89 0.99
```
- 排序
-- Series: `.sort_index`
--DataFrame: `.sort_values(by='a')`

- 汇总与统计
```
df = DataFrame(np.random.randn(9).reshape(3,3), columns=['a', 'b', 'c'])
>>> df
a b c
0 -0.637101 0.304616 -1.381508
1 0.587885 0.938603 0.064915
2 0.593575 -1.495701 -0.843845

>>> df.sum()
a 0.544359
b -0.252483
c -2.160439
dtype: float64

>>> df.sum(axis=1)
0 -1.713993
1 1.591402
2 -1.745972
dtype: float64
```
- describe可对每个数值型列进行统计,常用于对数据的初步观察时使用
```
data = {'name':['张三 ', '李四 ', '王五 ', '小明 '],
        'sex': ['female', 'female', 'male', 'male'],
        'year':[2001, 2001, 2003, 2002],
        'city':['北京 ', '上海 ', '广州 ', '北京 ']}
df= DataFrame(data)
df.describe()
>>> df.describe()
year
count 4.000000
mean 2001.750000
std 0.957427
min 2001.000000
25% 2001.000000
50% 2001.500000
75% 2002.250000
max 2003.000000
```
- 唯一值和值计数
```
obj= Series(['a', 'b', 'a', 'c', 'b'])
obj.unique()
obj.value_counts()

>>> obj
0 a
1 b
2 a
3 c
4 b
dtype: object
>>> obj.unique()
array(['a', 'b', 'c'], dtype=object)
>>> obj.value_counts()
b 2
a 2
c 1
dtype: int64
```

### 层次化索引
- 层次化索引就是轴上有多个级别索引
```
obj= Series(np.random.randn(9), 
            index=[['one','one','one','two','two','two','three','three','three'],
                   ['a', 'b', 'c','a', 'b', 'c','a', 'b', 'c']])
>>> obj
one   a -2.309879
      b -0.080024
      c -0.320460
two   a -1.084222
      b 0.249383
      c -0.626188
three a 0.489591
      b -0.543118
      c -0.791813
dtype: float64

>>> obj.index  # 索引对象为MultiIndex对象
MultiIndex([( 'one', 'a'),
( 'one', 'b'),
( 'one', 'c'),
( 'two', 'a'),
( 'two', 'b'),
( 'two', 'c'),
('three', 'a'),
('three', 'b'),
('three', 'c')],
)

>>> obj['two']
a -1.084222
b 0.249383
c -0.626188
dtype: float64

>>> obj[:, 'a']
one -2.309879
two -1.084222
three 0.489591
dtype: float64
```
```
df = DataFrame(np.arange(16).reshape(4,4),
               index=[['one','one','two','two'], ['a','b','a','b']],
               columns=[['apple','apple','orange','orange'], ['red','green','red','green']])
>>> df
    apple     orange
    red green red green
one a 0 1 2 3
    b 4 5 6 7
two a 8 9 10 11
    b 12 13 14 15

>>> df['apple']
    red green
one a 0 1
    b 4 5
two a 8 9
    b 12 13
```
- 通过`swaplevel`可对层次化索引进行重排
```
df.swaplevel(0,1)
>>> 
      apple     orange
      red green red green
a one 0 1 2 3
b one 4 5 6 7
a two 8 9 10 11
b two 12 13 14 15
```
- 汇总统计, 可通过level参数指定在某层次上进行汇总统计
```
>>> df
    apple     orange
    red green red green
one a 0 1 2 3
    b 4 5 6 7
two a 8 9 10 11
    b 12 13 14 15
    
>>> df.sum(level=0)
    apple     orange
    red green red green
one 4 6 8 10
two 20 22 24 26

>>> df.sum(level=1, axis=1)
      red green
one a 2 4
    b 10 12
two a 18 20
    b 26 28
```

### pandas可视化
- 线形图
```
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline # 魔法函数,使用该函数绘制的图片会直接显示在notebook中

s = Series(np.random.normal(size=10))
s.plot()
plt.show()
```
\
![](data/image1.png?v=1&type=image)

```
df = DataFrame({'normal': np.random.normal(size=100),
                'gamma': np.random.gamma(1, size=100),
                'poisson': np.random.poisson(size=100)})
df.cumsum().plot()
plt.show()
```
![](data/image2.png?v=1&type=image)

- 柱状图. 只需在plot函数中加入`kind='bar'`. 如类别较多,可会知水平柱状图`kind='barh'`, `stacted=True`可绘制堆积柱状图
```
data = {'name': ['张三','李四','王五','小明','Peter'],
        'sex': ['female','female','male','male','male'],
        'city': ['北京','上海','广州','北京','北京']}
df = DataFrame(data)
>>> df['sex'].value_counts()
male 3
female 2
Name: sex, dtype: int64

df['sex'].value_counts().plot(kind='bar')
```
![](data/image3.png?v=1&type=image)

```
df2 = DataFrame(np.random.randint(0,100,size=(3,3)),
                index=['one','two','three'],
                columns=['A','B','C'])
>>> df2
A B C
one 36 51 98
two 62 70 46
three 91 63 8

df2.plot(kind='barh')
```
![](data/image4.png?v=1&type=image)

```
df2.plot(kind='barh', stacked=True, alpha=0.5)
```
![](data/image5.png?v=1&type=image)

- 直方图和密度图
-- 直方图 `grid`添加网格, `bins`将值分为多少段, 默认为16
```
s = Series(np.random.normal(size=100))
s.hist(bins=20, grid=False)
```
![](data/image6.png?v=1&type=image)

-- 密度图: 核密度估计(kernel Density Estimate, KDE)是对振实密度的估计, 即将数据的分布近似为一组核(如正态分布)
```
s.plot(kind='kde')
```
![](data/image7.png?v=1&type=image)

- 散点图
```
df3 = DataFrame(np.arange(10), columns=['X'])
df3['Y'] = 2 * df3['X'] + 5
>>> df3
X Y
0 0 5
1 1 7
2 2 9
3 3 11
4 4 13
5 5 15
6 6 17
7 7 19
8 8 21
9 9 23

df3.plot(kind='scatter', x='X', y='Y')
```
![](data/image8.png?v=1&type=image)

### 综合示例--小费数据集
- 数据分析流程
-- 收集数据
-- 定义问题
-- 数据清洗与整理
-- 数据探索
-- 数据展示
- 数据来源: seaborn为python第三方数据库,用于绘图
```
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import seaborn as sns
tips = sns.load_dataset('tips')
tips.head()   # 返回前五条数据,也可指定返回数据行数

# total_bill列为消费总金额; tip列为小费金额; sex列为顾客性别; smoker列为顾客是否抽烟; 
# day为消费的星期;size列为聚餐人数

>>> tips.head()
total_bill tip sex smoker day time size
0 16.99 1.01 Female No Sun Dinner 2
1 10.34 1.66 Male No Sun Dinner 3
2 21.01 3.50 Male No Sun Dinner 3
3 23.68 3.31 Male No Sun Dinner 2
4 24.59 3.61 Female No Sun Dinner 4
```

- 定义问题
-- 消费金额与消费总金额是否存在相关性
-- 性别, 是否吸烟, 星期几, 中/晚餐, 聚餐人数和小费金额是否有一定的关联
-- 小费金额站消费总金额的百分比服从正态分布吗

- 数据清洗
```
# 对数据进行简单描述,看是否有缺失值或异常值
>>> tips.shape
(244, 7)

>>> tips.describe()
total_bill tip size
count 244.000000 244.000000 244.000000
mean 19.785943 2.998279 2.569672
std 8.902412 1.383638 0.951100
min 3.070000 1.000000 1.000000
25% 13.347500 2.000000 2.000000
50% 17.795000 2.900000 2.000000
75% 24.127500 3.562500 3.000000
max 50.810000 10.000000 6.000000

>>> tips.info()
```
![](data/image9.png?v=1&type=image)

总共有244个数据, 通过统计暂时看不出是否有缺失值. 通过打印数据的info信息可以看出每列数据的类型和缺失值. 本例中没有缺失值

- 数据探索
-- 分析小费金额与消费总金额, 是否有关联, 绘制散点图
```
tips.plot(kind='scatter', x='total_bill', y='tip')
# 小费金额与消费总金额存在着正相关的关系
```
![](data/image.png?v=1&type=image)

-- 性别不一样是否会影响小费的金额, 用柱状图, 通过布尔选择男女性别, 对小费数据进行平均后绘制柱状图
```
male_tip = tips[tips['sex']=='Male']['tip'].mean()
female_tip = tips[tips['sex']=='Female']['tip'].mean()
s = Series([male_tip, female_tip], index=['male', 'female'])
>>> s
male 3.089618
female 2.833448
dtype: float64

s.plot(kind='bar')
# 女性消费金额小于男性消费金额
```
![](data/image10.png?v=1&type=image)

-- 其他字段与小费的关系也是类似的方法. 如小费与日期的关系
```
>>> tips['day'].unique()
[Sun, Sat, Thur, Fri]
Categories (4, object): [Sun, Sat, Thur, Fri]

sun = tips[tips['day']=='Sun']['tip'].mean()
sat = tips[tips['day']=='Sat']['tip'].mean()
thur = tips[tips['day']=='Thur']['tip'].mean()
fri = tips[tips['day']=='Fri']['tip'].mean()

s = Series([thur, fri, sat, sun], index=['Thur', 'Fri', 'Sat', 'Sun'])
>>> s
Thur 2.771452
Fri 2.734737
Sat 2.993103
Sun 3.255132
dtype: float64

s.plot(kind='bar')

# 周六 周日的小费比周四 周五的小费高
```
![](data/image11.png?v=1&type=image)

-- 分析小费百分比的分布情况
```
tips['percent_tip'] = tips['tip'] / (tips['total_bill'] + tips['tip'])
tips.head(10)

total_bill tip sex smoker day time size percent_tip
0 16.99 1.01 Female No Sun Dinner 2 0.056111
1 10.34 1.66 Male No Sun Dinner 3 0.138333
2 21.01 3.50 Male No Sun Dinner 3 0.142799
3 23.68 3.31 Male No Sun Dinner 2 0.122638
4 24.59 3.61 Female No Sun Dinner 4 0.128014
5 25.29 4.71 Male No Sun Dinner 4 0.157000
6 8.77 2.00 Male No Sun Dinner 2 0.185701
7 26.88 3.12 Male No Sun Dinner 4 0.104000
8 15.04 1.96 Male No Sun Dinner 2 0.115294
9 14.78 3.23 Male No Sun Dinner 2 0.179345

tips['percent_tip'].hist(bins=50, grid=True)

# 基本上符合正态分布

```
![](data/image12.png?v=1&type=image)


## <span id="jump">外部数据的读取与存储</span>
### 文本数据的读取与存储
- CSV文件的读取
-- `read_csv`: 从文件中加载带分隔符的数据, 默认分隔符为逗号
-- `read_table`: 默认分隔符为制表符
```
# 创建csv文件
import csv
fp = open('D:\myproject\ch4ex1.csv', 'w', newline='')
writer = csv.writer(fp)
writer.writerow(('id', 'name', 'grade'))
writer.writerow(('1', 'lucky', '87'))
writer.writerow(('2', 'peter', '92'))
writer.writerow(('3', 'lili', '85'))
fp.close()

# 查看数据
type D:\myproject\ch4ex1.csv  # linux下是cat
id,name,grade
1,lucky,87
2,peter,92
3,lili,85

#读csv文件
import pandas as pd
df = pd.read_csv(open('D:\myproject\ch4ex1.csv'))
>>> df
id name grade
0 1 lucky 87
1 2 peter 92
2 3 lili 85

df = pd.read_table(open('D:\myproject\ch4ex1.csv'), sep=',') # 指定分隔符
```
```
# 指定列作为索引
df = pd.read_csv(open('D:\myproject\ch4ex1.csv'), index_col='id')
>>> df
  name grade
id
1 lucky 87
2 peter 92
3 lili 85
```
```
# 多列层次化索引
school,id,name,grade
a,1,lucky,87
a,2,peter,92
a,3,lili,85
b,1,coco,78
b,2,kevin,87
b,3,heven,96

df = pd.reand_csv(open('aa.csv'), index_col=[0, 'id'])
>>> df
          name grade
school id
a       1 lucky 87
        2 peter 92
        3 lili 85
b       1 coco 78
        2 kevin 87
        3 heven 96
```
```
# 标题行设置
# 默认读取会指定第一行为标题行
df = pd.read_csv(open('ch4ex3.csv'))
  1 lucky 87
0 2 peter 92
1 3 lili 85

# 通过header参数分配默认的标题行
df = pd.read_csv(open('ch4ex3.csv'), header=None)
>>> df
  0 1     2
0 1 lucky 87
1 2 peter 92
2 3 lili 85

# 通过names参数指定列名
df = pd.read_csv(open('ch4ex3.csv'), names=['id', 'name', 'grade'])
>>> df
  id name grade
0 1 lucky 87
1 2 peter 92
2 3 lili 85
```
```
# 自定义读取, 跳过一些行
df = pd.read_csv(open('ch4ex3.csv'), skiprows=[0,5])

# 只读取部分数据
df = pd.read_csv(open('ch4ex3.csv'), nrows=10)

# 选取部分列
df = pd.read_csv(open('ch4ex3.csv'), usecols=['Survived', 'Sex'])

# 逐步读取文件
chunker = pd.read_csv(open('titanic.csv', chunksize=100)) # 返回可迭代的TextFileReader
# 通过迭代对sex列进行统计

chunker = pd.read_csv(open('titanic.csv', chunksize=100)) # 返回可迭代的TextFileReader
sex = Series([])
for i in chunker:
    sex=sex.add(i['Sex'].value_counts(), fill_value=0)
>>> sex
>
male 577.0
female 314.0
dtype: float64
```

- txt文件的读取
```
# 创建txt文件, 分隔符为?
fp = open('1.txt', 'a+')
fp.writelines('id?name?grade'+'\n')
fp.writelines('1?lucky?87'+'\n')
fp.writelines('2?peter?92'+'\n')
fp.writelines('3?lili?85'+'\n')
fp.close

df = pd.read_table(open('1.txt'), sep='?')  # sep 指定分隔符
df = pd.read_table(open('1.txt'), sep='\s+') # 适用于没有固定分隔符的情况
```

- 文本数据的存储
```
df.to_csv('out.csv')  # 以逗号为分隔符
df.to_csv('out1.csv', sep='?')  # sep指定分隔符
df.to_csv('out2.csv', index=False) # 通过设置index和header分别处理行和列索引
```

- JSON的读取与存储
-- json数据是一种轻量级的数据交换合适
```
import json
f = open('1.json')
obj = f.read()
result = json.loads(obj)

# 将数据输入DataFrame构造器,完成对JSON数据的读取
df = DataFrame(result)

# 另一种读取方法
df = pd.read_json('1.json')

df = df.sort_index()   # 读取时会乱序, 重新排序

# 对DataFrame进行存储
df.to_json('2.json')
```

- Excel数据的读取与存储
```
# 读取
df = pd.read_excel('1.xlsx', sheetname='Sheet1') # sheetname指定读取的工作簿

# 存储
df.to_excel('2.xlsx', sheetname='out', index=None)
```

### Web数据库读取
- 读取html表格
```
import pandas as pd
df = pd.read_html('http://worldcup.2014.163.com/schedule/')
```

- 网络爬虫. 把爬虫到的数据转换成DataFrame数据格式
```
# 爬虫代码
import requests  
from bs4 import BeautifulSoup  
from pandas import DataFrame  

data = []  
wb_data = requests.get('http://www.kugou.com/yy/rank/home/1-8888.html')  
soup = BeautifulSoup(wb_data.text, 'lxml')  
ranks = soup.select('span.pc_temp_num')  
titles = soup.select('div.pc_temp_songlist > ul > li > a')  
times = soup.select('span.pc_temp_tips_r > span')  
for rank, title, time in zip(ranks, titles, times):  
    a = {  
        'rank': rank.get_text().strip(),  
        'singer': title.get_text().split('-')[0],  
        'song': title.get_text().split('-')[1],  
        'time': time.get_text().strip()  
    }  
    data.append(a)  
print(data)  
  
df = DataFrame(data)  
print(df)

[{'rank': '1', 'singer': '李荣浩 ', 'song': ' 爸爸妈妈 ', 'time': '4:44'}, {'rank': '2', 'singer': '华莎 ', 'song': ' 마리아(María)', 'time': '3:19'}, {'rank': '3', 'singer': '白小白 ', 'song': ' 我爱你不问归期 ', 'time': '4:14'}, {'rank': '4', 'singer': 'Ava Max ', 'song': ' Salt', 'time': '3:02'}, {'rank': '5', 'singer':'王靖雯不胖 ', 'song': ' 爱,存在 ', 'time': '4:38'}, {'rank': '6', 'singer': '张韶涵 ', 'song': ' 破 茧 ', 'time': '3:31'}, {'rank': '7', 'singer': '伊格赛听 、 叶里 ', 'song': ' 谪仙 ', 'time': '2:58'}, {'rank': '8', 'singer': '任然 ', 'song': ' 飞鸟和蝉 ', 'time': '4:56'}, {'rank': '9', 'singer': '刘大壮 ', 'song': ' 信仰(吉他版 )', 'time': '2:06'}, {'rank': '10', 'singer': '任然 ', 'song': ' 无人之岛 ', 'time': '4:45'}, {'rank': '11', 'singer': '海来阿木 ', 'song': ' 你的万水千山 ', 'time': '4:09'}, {'rank': '12', 'singer': '阿悠悠 ', 'song': ' 旧梦一场 ', 'time': '2:54'}, {'rank': '13', 'singer': '蔡徐坤 ', 'song': ' 情人 ', 'time': '3:15'}, {'rank': '14', 'singer': '蒋雪儿 ', 'song': ' 莫问归期 ', 'time': '3:39'}, {'rank': '15', 'singer': '蔡健雅 ', 'song': ' 红色高跟鞋 ', 'time': '3:26'}, {'rank': '16', 'singer': '程响 ', 'song': ' 想起了你 ', 'time': '3:36'},{'rank': '17', 'singer': '添儿呗 ', 'song': ' 烟火人间 ', 'time': '4:25'}, {'rank': '18', 'singer': '等什么君 ', 'song': ' 辞九门回忆 ', 'time': '4:00'}, {'rank': '19', 'singer': '王韵', 'song': ' 思念成沙 ', 'time':'4:46'}, {'rank': '20', 'singer': '纸砚 zyan ', 'song': ' 画皮 ', 'time': '3:44'}, {'rank': '21', 'singer': '刘大壮 ', 'song': ' 一吻天荒 (吉他版 )', 'time': '4:04'}, {'rank': '22', 'singer': '松紧先生 （李宗锦） ', 'song': ' 你走 ', 'time': '4:04'}]

rank singer        song time
0 1 李荣浩          爸爸妈妈         4:44
1 2 华莎            마리아 (María)  3:19 
2 3 白小白          我爱你不问归期    4:14
3 4 Ava Max        Salt            3:02
4 5 王靖雯          不胖爱,存在       4:38
5 6 张韶涵          破茧             3:31
6 7 伊格赛听、叶里   谪仙              2:58
7 8 任然           飞鸟和蝉          4:56
8 9 刘大壮          信仰(吉他版)      2:06
9 10 任然           无人之岛         4:45
10 11 海来阿木      你的万水千山       4:09
11 12 阿悠悠        旧梦一场          2:54
12 13 蔡徐坤        情人              3:15
13 14 蒋雪儿        莫问归期          3:39
14 15 蔡健雅        红色高跟鞋        3:26
15 16 程响          想起了你          3:36
16 17 添儿呗        烟火人间          4:25
17 18 等什么君      辞九门回忆         4:00
18 19 王韵          思念成沙          4:46
19 20 纸砚zyan      画皮              3:44
20 21 刘大壮        一吻天荒(吉他版)   4:04
21 22 松紧先生（李宗锦） 你走          4:04
```

## <span id="jump">数据清洗与整理</span>
### 数据清洗
- 处理缺失值
```
# 创建有缺失值的数据
from pandas import Series, DataFrame
import pandas a spd
import numpy as np

df1 = DataFrame([[3,5,3], [1,6,np.nan], ['lili', np.nan,'pop'], [np.nan,'a','b'])
>>> df1
  0    1   2
0 3    5   3
1 1    6   NaN
2 lili NaN pop
3 NaN  a   b

>>> df1.isnull()  # True为缺失值
  0     1     2
0 False False False
1 False False True
2 False True  False
3 True  False False

>>> df1.notnull() # False为缺失值
  0     1     2
0 True  True  True
1 True  True  False
2 True  False True
3 False True  True

>>> df1.isnull().sum() # 每列的缺失值数量
0 1
1 1
2 1
dtype: int64

>>> df1.isnull().sum().sum() #整个表的缺失值数量
3

df1.info() # 也可看到每行缺失值信息
```

- 删除缺失值
```
df1.dropna() # 删除具有缺失值的行
>>> df1.dropna()
  0 1 2
0 3 5 3

df1.dropna(how='all') # 只删除全为缺失值的行
df2.dropna(how='all', axis=1) # 删除列
```

- 填充缺失值
```
# 将缺失值填充为常数值
df1.fillna(0)

>>> df2
  0   1   2   3
0 0.0 1.0 2.0 NaN
1 4.0 5.0 6.0 NaN
2 NaN NaN NaN NaN

# 在fillna中传入字典结构数据, 可以针对不同列填充不同值
>>> df2.fillna({1:6, 3:0}, inplace=True)
  0   1   2   3
0 0.0 1.0 2.0 0.0
1 4.0 5.0 6.0 0.0
2 NaN 6.0 NaN 0.0

>>> df2.fillna(method='ffill')
0 1 2 3
0 0.0 1.0 2.0 0.0
1 4.0 5.0 6.0 0.0
2 4.0 6.0 6.0 0.0

df2[0]=df2[0].fillna(df2[0].mean()) # 填充平均值
>>> df2
0 1 2 3
0 0.0 1.0 2.0 0.0
1 4.0 5.0 6.0 0.0
2 2.0 6.0 NaN 0.0
```

- 移除重复数据
```
>>> df1
  name  sex   year  city
0 张三  female 2001  北京
1 李四  male   2002  上海
2 张三  female 2001  北京
3 小明  male   2002  北京

>>> df1.duplicated() # 判断是否为重复数据
0 False
1 False
2 True
3 False
dtype: bool

>>> df1.drop_duplicates() # 删除多余的重复项
  name  sex   year   city
0 张三  female 2001  北京
1 李四  male   2002  上海
3 小明  male   2002  北京

>>> df1.drop_duplicates(['sex', 'year']) # 指定判断重复列
  name  sex   year   city
0 张三  female 2001  北京
1 李四  male   2002  上海
``` 

- 替换值
```
df1.replace('', '不详')
df1.replace(['', 2001], ['不详', 2002]) # ''-->'不详', 2001-->2002
```

- 利用函数或映射进行数据转换
```
data = {'name':['张三', '李四', '王五', '小明'], 
        'math': [79, 52, 63, 92]}
df = DataFrame(data)
>>> df
  name math
0 张 三 79
1 李 四 52
2 王 五 63
3 小 明 92

def f(x):
    if x >= 90:
        return '优秀'
    elif 70<=x<=90:
        return '良好'
    elif 60<=x<=70:
        return '合格'
    else:
        return '不合格'

df['class'] = df['math'].map(f)
```

- 检测异常值: 可通过画图找到离群点 (并非所有离群点都是异常值)
- 虚拟变量: 在计算中需用数值型数据, 因此需将分类变量转化为虚拟变量(哑变量, 即0,1 矩阵)
```
df = DataFrame({'朝向': ['东', '南', '东', '西', '北'],
                '价格': [1200, 2100, 2300, 2900, 1400]})
>>> df
  朝向 价格
0 东 1200
1 南 2100
2 东 2300
3 西 2900
4 北 1400

>>> pd.get_dummies(df['朝向'])
  东 北 南 西
0 1 0 0 0
1 0 0 1 0
2 1 0 0 0
3 0 0 0 1
4 0 1 0 0

# 对于多类别数据,需用apply函数实现
df = DataFrame({'朝向': ['东/北', '西/南', '东', '西/北', '北'],
                '价格': [1200, 2100, 2300, 2900, 1400]})
>>> df
  朝向   价格
0 东/北  1200
1 西/南  2100
2 东     2300
3 西/北  2900
4 北     1400

>>> dummies = df['朝向'].apply(lambda x:Series(x.split('/')).value_counts())

>>> dummies
东 北 南 西
0 1.0 1.0 NaN NaN
1 NaN NaN 1.0 1.0
2 1.0 NaN NaN NaN
3 NaN 1.0 NaN 1.0
4 NaN 1.0 NaN NaN

dummies = dummies.fillna(0).astype(int)
>>> dummies
东 北 南 西
0 1 1 0 0
1 0 0 1 1
2 1 0 0 0
3 0 1 0 1
4 0 1 0 0
```
