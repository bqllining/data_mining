# Task1: 赛题理解

## 赛题：零基础入门数据挖掘-二手车交易价格预测
地址：https://tianchi.aliyun.com/competition/entrance/231784/introduction?spm=5176.12281957.1004.1.38b02448ausjSX

---
学习目标：理解赛提数据和目标，清楚评分体系。

---
### 1.1 了解赛题
- 赛题概况
- 数据概况
- 预测指标
- 分析赛题

---

#### 1.1.1 赛题概况
根据给定的数据集，建立模型，预测二手车的交易价格。

----

#### 1.1.2 数据概况
数据来自某交易平台的二手车交易记录，总数据量超过40w，包含31列变量信息，其中15列
为匿名变量（比赛界面不会对其有数据概况介绍，性质未知）。为了保证比赛公平性，将会
从中抽取15万条作为训练集，5万条作为测试集A，5万条作为测试集B，同时会对name、
model、brand和regionCode等信息进行脱敏。

----------------
train.csv

- SaleID - 交易ID，唯一编码
- name - 汽车交易名称，已脱敏
- regDate - 汽车注册时间，例如20160101，,016年01月01日
- model - 车型编号，已脱敏
- brand - 品牌，已脱敏
- bodyType - 车身类型：豪华轿车0，微型车1，厢型车,，大巴车3，敞篷车4，双门汽车5，商务车6，搅拌车7
- fuelType - 燃油类型：汽油0，柴油1，液化石油气2，天然气3，混合动力4，其他5，电动6
- gearbox - 变速箱：手动0，自动1
- power - 汽车功率，范围[0,600]
- kilometer - 汽车已行驶公里,单位万km
- notRepairedDamage - 汽车有尚未修复的损坏：是1，否0
- regionCode - 看车地区编码，已脱敏
- seller - 销售方：个体0，非个体1
- offerType - 报价类型：提供0，请求1
- creatDate - 汽车上线时间，即开始售卖时间
- price - 二手车交易价格（预测目标）
- v系列特征 - 匿名特征，包含v_0-14在内15个匿名特征

**注：变量含义在赛题里有解释。**

-----

#### 1.1.3 预测指标

评价标准为MAE（Mean Absolute Error)。
若真实值为$y=(y_1,y_2,...,y_n)$,模型的预测值为$\hat{y}=(\hat{y_1},\hat{y_2},...,\hat{y_n})$，那么该模型的MAE计算公式为
$$MAE=\frac{\sum_{i=1}^n\left|y_i-\hat{y_n}\right|}{n}.$$
MAE越小，说明模型预测越准确。

*注：一般问题评估指标说明*
评估指标就是对一个模型效果的数值化量化，类似于对一个商品评价打分，这是针对于模型效果和理想效果之间的一个打分。

- 分类算法常见的评估指标
  - 二分类器/算法：accuracy、[Precision、Recall、F-score、Pr曲线]，ROC-AUC曲线。
    TP(真正例)、FP（假正例）、FN（假反例）、TN(真反例)，TP+FP+TN+FN=样例总数
    - accuracy（准确率）：分类错误的样本数占样本总数的比例
      $$E(f;D)=\frac{1}{m}{\sum_{i=1}^{m}I(f(x_i)\not= y_i)}$$
    - precision（精确度、查准率）：分类正确的样本数占样本总数的比例
      $$P=\frac{TP}{TP+FP}$$
    - recall（召回率、查全率）： 
      $$R=\frac{TP}{TP+FN}$$
    - F-score:
     $$F1=\frac{2×P×R}{P+R}=\frac{2×TP}{样例总数+TP-TN}$$
  
  - 多分类器/算法：accuracy、[宏平均和微平均，F-score]。
- 回归预测算法常见的评估指标
  - 平均绝对误差MAE、均方误差MSE、平均绝对百分误差MAPE、均方根误差RMSE、$R^2$。
    - 平均绝对误差  
     $$ MAE=\frac{1}{N} \sum_{i=1}^{N}\left|y_{i}-\hat{y}_{i}\right| $$
    - 均方误差
     $$ MSE=\frac{1}{N} \sum_{i=1}^{N}\left(y_{i}-\hat{y}_{i}\right)^{2} $$
    - $R^2$ 
       $$ SS_{res}=\sum\left(y_{i}-\hat{y}{i}\right)^{2} $$  
       $$ SS{tot}=\sum\left(y_{i}-\overline{y}_{i}\right)^{2} $$  
       $$ R^{2}=1-\frac{SS_{res}}{SS_{tot}}=1-\frac{\sum\left(y_{i}-\hat{y}{i}\right)^{2}}{\sum\left(y{i}-\overline{y}\right)^{2}} $$ 

---
#### 1.1.4 分析赛题
1.此题为传统的数据挖掘问题，通过数据科学以及机器学习深度学习的办法来进行建模得到结果。
2.此题是一个典型的回归问题。
3.主要应用xgb、lgb、catboost，以及pandas、numpy、matplotlib、seabon、sklearn、keras等等数据挖掘常用库或者框架来进行数据挖掘任务。
4.通过EDA来挖掘数据的联系和自我熟悉数据。

---
### 1.2 代码示例
（数据读取代码经常用到，已经会了，这里主要是指标评价的代码。代码示例也可见代码示例.ipynb）

---
#### 1.2.1 分类指标评价计算示例
``` python
# accuracy
import numpy as np
from sklearn.metrics import accuracy_score
y_pred = [0, 1, 0, 1]
y_true = [0, 1, 1, 1]
print('ACC:',accuracy_score(y_true, y_pred))
``` 
```
ACC:0.75
```

```python
# Precision,Recall,F1-score
from sklearn import metrics
y_pred = [0, 1, 0, 1]
y_true = [0, 1, 1, 1]
print('Precision',metrics.precision_score(y_true,y_pred))
print('Recall',metrics.recall_score(y_true,y_pred))
print('F1-score:',metrics.f1_score(y_true,y_pred))
```
```
Precision 1.0
Recall 0.6666666666666666
F1-score:0.8
```
```python
# AUC
import numpy as np
from sklearn.metrics import roc_auc_score
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
print('AUC score:',roc_auc_score(y_true,y_scores))
```
```
AUC score:0.75
```
#### 1.2.2 回归指标评价计算实例

```python
# coding=utf-8
import numpy as np
from sklearn import metrics

# MAPE需要自己实现
def mape(y_true,y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))

y_true = np.array([1.0, 5.0, 4.0, 3.0, 2.0, 5.0, -3.0])
y_pred = np.array([1.0, 4.5, 3.8, 3.2, 3.0, 4.8, -2.2])

# MSE
print('MSE:',metrics.mean_squared_error(y_true,y_pred))

# RMSE
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_true,y_pred)))

# MAE
print('MAE:',metrics.mean_absolute_error(y_true,y_pred))

# MAPE
print('MAPE:',mape(y_true,y_pred))
```
```
MSE:0.2871428571428571
RMSE: 0.5358571238146014
MAE: 0.4142857142857143
MAPE: 0.1461904761904762
```
``` python
# R2-score
from sklearn.metrics import r2_score
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print('R2-score:',r2_score(y_true,y_pred))
```
```
R2-score:0.9486081370449679
```

---
### 1.3 经验总结
赛题理解是切入一道赛题的基础，类似于我们在解题之前通读题干，会影响到后续的特征工程构建以及模型的选择，最主要的是会影响后续发展工作的方向，比如挖掘特征的方向或者存在问题解决问题的方向。对于赛题背后的思想以及赛题业务逻辑的清晰，也很有利于花费更少时间构建更为有效的特征模型。

- 1）赛题理解究竟是理解什么：
  - 该赛题符合的问题是什么问题，大概要去用哪些指标，做到线上线下的一致性，是否有效的利于我们进一步的探索更高线上分数的线下验证方法，在业务上，你是否对很多原始特征有很深刻的了解，并且可以通过EDA来寻求他们直接的关系，最后构造出满意的特征。

- 2） 有了赛题理解后能做什么： 
 - 有一些相应的理解分析，比如这题的难点可能在哪里，关键点可能在哪里，哪些地方可以挖掘更好的特征，用什么样的线下验证方式更为稳定，出现了过拟合或者其他问题怎么解决，哪些数据是可靠的，哪些数据是需要精密的处理的，哪部分数据应该是关键数据（背景的业务逻辑下，比如CTR的题，一个寻常顾客大体会有怎么样的购买行为逻辑规律，或者风电那种题，如果机组比较邻近，相关一些风速，转速特征是否会很近似）。这时是在一个宏观的大体下分析的，有助于摸清整个题的思路脉络，以及后续的分析方向。

- 3） 赛题理解的评价指标： 
  -  1．本地模型的验证方式，很多情况下，线上验证是有一定的时间和次数限制的，所以在比赛中构建一个合理的本地的验证集和验证的评价指标是很关键的步骤，能有效的节省很多时间。 
  -  2．不同的指标对于同样的预测结果是具有误差敏感的差异性的，比如AUC，logloss, MAE，RSME，或者一些特定的评价函数。是会有很大可能会影响后续一些预测的侧重点。

- 4） 赛题背景中可能潜在隐藏的条件： 
  - 其实赛题中有些说明是很有利益-都可以在后续答辩中以及问题思考中所体现出来的，比如高效性要求，比如对于数据异常的识别处理，比如工序流程的差异性，比如模型运行的时间，比模型的鲁棒性，有些的意识是可以贯穿问题思考，特征，模型以及后续处理的，也有些会对于特征构建或者选择模型上有很大益处，反过来如果在模型预测效果不好，其实有时也要反过来思考，是不是赛题背景有没有哪方面理解不清晰或者什么其中的问题没考虑到。
---

### 1.4 心得体会
在拿到一份数据之后，或者是拿到一道题目之前，首先需要通读赛题理解，一般数据量非常大，可能直接也无法打开，这时候就需要通过赛题说明里的信息对数据有个初步的了解。一般对各个变量的含义会进行解释说明，这是非常重要的，在后续的分析中，如果忘记了变量的含义，分析将会陷入困境。
另外，如果拿到的问题是自己不熟悉的领域，还应该去搜集相关信息，充分了解背景知识能帮助我们更好地解决问题。