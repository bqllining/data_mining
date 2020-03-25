# Task3 特征工程
赛题：零基础入门数据挖掘-二手车交易价格预测
地址：
https://tianchi.aliyun.com/competition/entrance/231784/introduction?spm=5176.12281957.1004.1.38b02448ausjSX

## 1 特征工程目标
- 对特征进行进一步分析，并对于数据进行处理
- 完成对于特征工程的分析，并对于数据进行一些图表或文字总结。
  
## 2 常见特征工程
### 1.异常处理
- 通过箱线图或3-Sigma分析删除异常值
- BOX-COX转换（处理有偏分布）
  - 广义幂变换，让数据满足线性模型的基本假定，即线性、正态性及方差齐性。
  - 参考
    https://baike.baidu.com/item/box-cox%E5%8F%98%E6%8D%A2/10278422?fr=aladdin 
- 长尾截断 http://www.woshipm.com/pmd/440974.html


### 2.特征归一化/标准化
- 标准化（转换为标准正态分布）
- 归一化（转换到[0,1]区间
- 针对幂律分布，可以采用公式$log(\frac{1+x}{1+median})$


### 3.数据分桶（类似分类/分组，将连续变量离散化，将多状态的离散变量合并成少状态。）
- 等频分桶
- 等距分桶
- Best-KS分桶（类似利用基尼系数进行二分类） https://blog.csdn.net/hxcaifly/article/details/84593770
- 卡方分桶（依赖于卡方检验，具有最小卡方值的相邻区间合并在一起）https://blog.csdn.net/hxcaifly/article/details/80203663


### 4.缺失值处理
- 不处理（针对类似XGBoost等树模型）
- 删除（缺失数据太多）
- 差值补全，均值/中位数/众数/建模预测/多重插补/压缩感知补全/矩阵补全等
- 分箱（缺失值一个箱）


### 5.特征构造
- 构造统计量特征，包括计数、求和、比例、标准差等
- 时间特征，包括相对时间和绝对时间，节假日，双休日
- 地理信息，包括分箱、分布编码等方法
- 非线性变换，包括log、平方、根号
- 特征组合，特征交叉
- 仁者见仁智者见智


### 6.特征筛选
- 过滤式（filter）：现对数据进行特征选择，然后再训练学习器，常见的方法有Relief/方差选择法/相关系数法/卡方检验法/互信息法
- 包裹式（wrapper）：直接把最终要使用的学习器的性能作为特征子集的评价准则，常见方法有LVM（Las Vegas Wrapper)
- 嵌入式（embedding):结合过滤式和包裹式，学习器训练过程中自动进行了特征选择，常见的有lasso


### 7.降维
- PCA/LDA/ICA
- 特征选择也是一种降维


## 3 代码
### 3.0 导入数据和需要的库
```python
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from operator import itemgetter

%matplotlib inline
```
```python
train = pd.read_csv('E:/Git-repository/data_mining/二手车价格预测组队学习/data/used_car_train_20200313.csv', sep=' ')
test = pd.read_csv('E:/Git-repository/data_mining/二手车价格预测组队学习/data/used_car_testA_20200313.csv', sep=' ')
print(train.shape)
print(test.shape)
```
```
(150000, 31)
(50000, 30)
```

```python
train.head()
```

```python
train.columns
```

```python
test.columns
```

### 3.1 删除异常值
这里包装了一个异常值处理的代码，可以随便调用。
```python
def outliers_proc(data, col_name, scale=3):
    '''
    用于清洗异常值，默认用box_plot(scale=3)进行清洗
    ：param data：接受pandas数据格式
    ：param col_name：pandas列名
    ：param scale：尺度
    ：return：
    '''

    def box_plot_outliers(data_ser, box_scale):
        '''
        利用箱线图去除异常值
        ：param data_ser：接收pandas.Series数据格式
        ：param box_scale：箱线图尺度
        ：return：
        '''
        iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
        val_low = data_ser.quantile(0.25) - iqr
        val_up = data_ser.quantile(0.75) + iqr
        rule_low = (data_ser < val_low)
        rule_up = (data_ser > val_up)
        return(rule_low, rule_up), (val_low, val_up)

    data_n = data.copy()
    data_series = data_n[col_name]
    rule, value = box_plot_outliers(data_series, box_scale=scale)
    index = np.arange(data_series.shape[0])[rule[0] | rule[1]]
    print('Delete number is: {}'.format(len(index)))
    data_n = data_n.drop(index)
    data_n.reset_index(drop=True, inplace=True)
    print('Now column number is: {}'.format(data_n.shape[0]))
    index_low = np.arange(data_series.shape[0])[rule[0]]
    outliers = data_series.iloc[index_low]
    print('Description of data less than the lower bound is:')
    print(pd.Series(outliers).describe())
    index_up = np.arange(data_series.shape[0])[rule[1]]
    outliers = data_series.iloc[index_up]
    print('Description of data larger than the upper bound is:')
    print(pd.Series(outliers).describe())

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    sns.boxplot(y=data[col_name], data=data, palette='Set1', ax=ax[0])
    sns.boxplot(y=data_n[col_name], data=data_n, palette='Set1', ax=ax[1])
    return data_n
```

```python
# 删掉power中的一些异常数据
# 注意不能删除test中的数据
train = outliers_proc(train, 'power', scale=3)
```

### 3.2 特征构造
```python
#训练集和测试集放在一起，方便构造特征
train['train'] = 1
test['train'] = 0
data = pd.concat([train, test], ignore_index=True, sort=False)
```

```python
# 使用时间：data['creatDate'] - data['regDate']反映汽车使用时间，一般来说价格与使用时间成反比
# 不过要注意，数据里有时间出错的格式，所以我们需要errors='coerce'
data['used_time'] = (pd.to_datetime(data['creatDate'], format='%Y%m%d', errors='coerce') - pd.to_datetime(data['regDate'], format='%Y%m%d', errors='coerce')).dt.days
```

```python
# 看一下空数据
data['used_time'].isnull().sum()
```
```
15072
```

* 这里不建议删除空值，因为缺失数据占总样本量比例过大，15072/(150000+50000) = 7.5% 
* 可以先放着，因为如果有XGBoost之类的决策树，其本身就能处理缺失值，所以可以不用管

```python
# 从邮编中提取城市信息，因为是德国的数据，所以参考德国的邮编，相当于加入了先验知识
data['city'] = data['regionCode'].apply(lambda x : str(x)[:-3])
```

```python
# 计算某品牌的销售统计量，也可以计算其他特征的统计量
# 这里要以train的数据计算统计量
train_gb = train.groupby('brand')
all_info = {}
for kind, kind_data in train_gb:
    info = {}
    kind_data = kind_data[kind_data['price'] > 0]
    info['brand_amount'] = len(kind_data)
    info['brand_price_max'] = kind_data.price.max()
    info['brand_price_median'] = kind_data.price.median()
    info['brand_price_min'] = kind_data.price.min()
    info['brand_price_sum'] = kind_data.price.sum()
    info['brand_price_std'] = kind_data.price.std()
    info['brand_price_average'] = round(kind_data.price.sum() / (len(kind_data) + 1), 2)
    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={'index':'brand'})
data = data.merge(brand_fe, how='left', on='brand')
```

接下来是数据分桶，以power为例。
做数据分桶的原因：
- 离散后稀疏向量内积乘法运算速度更快，计算结果页方便存储，容易扩展
- 离散后的特征对异常值更具鲁棒性，如age>30为1否则为0，对于年龄为200的也不会对魔性造成很大的干扰
- LR属于广义线性模型，表达能力有限，经过离散化后，每个变量有单独的权重，这相当于引入了非线性，能够提升模型的表达能力，加大拟合
- 离散后特征可以进行特征交叉，提升表达能力，由M+N个变量变成M*N个变量，进一步引入非线性，提升表达能力
- 特征离散后模型更稳定，如用户年龄区间不会因为用户增长了一岁就变化
- 其他原因，例如LightGBM在改进XGBoost时就增加了数据分桶，增强了模型的泛化性

```python
# 对power进行分桶
bin = [i*10 for i in range(31)]
data['power_bin'] = pd.cut(data['power'], bin, labels=False)
data[['power_bin', 'power']].head()
```

```python
# 利用好了，就可以删掉原始数据了
data = data.drop(['creatDate', 'regDate', 'regionCode'], axis=1)
```

```python
print(data.shape)
data.columns
```

```python
# 目前的数据其实已经可以给树模型使用了，导出数据
data.to_csv('data_for_tree.csv', index=0)
```
再构造一份特征给LR NN之类的模型用,因为不同模型对数据集的要求不同，因此分开构造

```python
# 首先看下数据分布
data['power'].plot.hist()
```

前面已经对train进行异常值处理了，但是现在还有这么奇怪的分布是因为test中的power异
常值，所以其实前面train中的power异常值不删为好，可以用长尾截断分布来代替
```python
train['power'].plot.hist()
```

对其取log，再做归一化
```python
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
data['power'] = np.log(data['power'] + 1)
data['power'] = ((data['power'] - np.min(data['power'])) / (np.max(data['power']) - np.min(data['power'])))
data['power'].plot.hist()
```

```python
data['kilometer'].plot.hist()
```

km的分布比较正常，应该是已经做过分桶了,直接做归一化
```python
data['kilometer'] = ((data['kilometer'] - np.min(data['kilometer'])) / (np.max(data['kilometer']) - np.min(data['kilometer'])))
data['kilometer'].plot.hist()
```

还有前面刚刚构造的统计量特征：'brand_amount', 'brand_price_average','brand_price_max','brand_price_median', 'brand_price_min','brand_price_std','brand_price_sum'，这里不再一一举例分析了，直接做变换。
```python
def max_min(x): 
    return (x - np.min(x)) / (np.max(x) - np.min(x))

data['brand_amount'] = max_min(data['brand_amount'])
data['brand_price_average'] = max_min(data['brand_price_average'])
data['brand_price_max'] = max_min(data['brand_price_max'])
data['brand_price_median'] = max_min(data['brand_price_median'])
data['brand_price_min'] = max_min(data['brand_price_min'])
data['brand_price_std'] = max_min(data['brand_price_std'])
data['brand_price_sum'] = max_min(data['brand_price_sum'])
```

对类别进行 OneEncoder
```python
data = pd.get_dummies(data, columns=['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'power_bin'])
```

```python
print(data.shape)
data.columns
```

这份数据可以给LR用
```python
data.to_csv('data_for_lr.csv', index=0)
```

### 3.3 特征筛选
#### 1） 过滤式
```python
# 相关性分析
print(data['power'].corr(data['price'], method='spearman'))
print(data['kilometer'].corr(data['price'], method='spearman'))
print(data['brand_amount'].corr(data['price'], method='spearman'))
print(data['brand_price_average'].corr(data['price'], method='spearman'))
print(data['brand_price_max'].corr(data['price'], method='spearman'))
print(data['brand_price_median'].corr(data['price'], method='spearman'))
```

```
0.5728285196051496
-0.4082569701616764
0.058156610025581514
0.3834909576057687
0.259066833880992
0.38691042393409447
```

```python
# 也可以直接看图
data_numeric = data[['power', 'kilometer', 'brand_amount', 'brand_price_average', 'brand_price_max', 'brand_price_median']]
correlation = data_numeric.corr()

f, ax = plt.subplots(figsize=(7,7))
plt.title('Correlation of Numeric Feature with Price', y=1, size=16)
sns.heatmap(correlation, square=True, vmax=0.8)
```

#### 2)包裹式
用到mlxtend包。

```python
# k_feature 太大会很难跑，没服务器，所以提前 interrupt 了
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
sfs = SFS(LinearRegression(),
           k_features=10,
           forward=True,
           floating=False,
           scoring = 'r2',
           cv = 0)
x = data.drop(['price'], axis=1)
x = x.fillna(0)
y = data['price']
sfs.fit(x, y)
sfs.k_feature_names_ 
```
装了还是报错不存在这个包……

```python
# 画出来，可以看到边际效益
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
plt.grid()
plt.show()
```

#### 3)嵌入式
在第四部分再做。
大部分情况下都是用嵌入式做特征选择。

## 4 经验总结
- 特征工程是比赛中最至关重要的一块，特别是传统的比赛，大家的模型可能都差不多，调参带来的效果增幅是非常有限的，但特征工程的好坏往往会决定了最终的排名和成绩。
- 特征工程主要目的是将数据转换为能更好地表示潜在问题的特征，从而提高机器学习的性能。比如，异常值处理是为了去除噪声，填补缺失值可以加入先验知识等。
- 特征构造也属于特征工程的一部分，目的是为了增强数据的表达。
- 匿名特征的处理：并不清楚特征之间直接的关联性，这时我们就只有单纯基于特征进行处理，比如装箱，groupby，agg 等这样一些操作进行一些特征统计，此外还可以对特征进行进一步的 log，exp 等变换，或者对多个特征进行四则运算（如上面我们算出的使用时长），多项式组合等然后进行筛选。由于特性的匿名性其实限制了很多对于特征的处理，当然有些时候用 NN 去提取一些特征也会达到意想不到的良好效果。
- 费匿名特征需要结合背景的特征进行构建，要深入分析背后的业务逻辑或者说是物理原理。
- 特征工程是和模型结合在一起的，所以要为不同的模型做不同的特征处理，对于特征的处理效果和特征的重要性等往往要通过模型来验证。
- 在做Task3的过程中发现我的pip出了问题，报错pip is configured with locations that require TLS/SSL, however the ssl module in Pyth...，原因是环境变量有问题，
  https://stackoverflow.com/questions/45954528/pip-is-configured-with-locations-that-require-tls-ssl-however-the-ssl-module-in
  参考这个添加环境变量解决了。
