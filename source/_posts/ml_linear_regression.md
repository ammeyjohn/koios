---
title: 一元线性回归
date: 2018-09-29 21:29:07
tags: [Machine Learning,回归,线性回归]
categories: 机器学习系列
mathjax: true
---

# 一元线性回归方程

$$E(y)=E(\beta_0+\beta_1x+\epsilon)=>y=\beta_0+\beta_1x$$

> 回归方程从平均意义上表达了变量y与x的统计规律性。

# 最小二乘估计(Least Square Estimation, OLE)

![OLE](/images/ole.png)

# 最大似然估计(Maximum Likelihood Estimation, MLE)

> 解决“模型已定，参数未知”的问题。即用已知样本的结果，去反推既定模型中的参数最可能的。

![MLE](/images/mle.png)

# 回归模型的显著性检验

## 回归系数是否显著：t检验

![tcheck1](/images/t_check_1.png)
![tcheck2](/images/t_check_2.png)
![tcheckp](/images/t_check_p.png)

## 回归方程是否显著：F检验

### 平方和的分解式

![平方和的分解式](/images/f_check_ss.png)

> F检验是根据平方和分解式，直接从回归效果检验回归方程的显著性。由平方和分解可得到SSR越大，回归效果越好。

![F检验](/images/f_check.png)

## 相关系数显著性检验：t检验

### 相关系数(Correlation Coefficient)

![相关系数](/images/correlation_coef.png)

### 决定系数(Coefficient of Determination)

![决定系数](/images/determination_coef.png)



