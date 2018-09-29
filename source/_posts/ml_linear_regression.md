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

$$Q(\beta_0,\beta_1)=\sum_{i=1}^n[y_1-E(y_1)]^2=\sum_{i=1}^n(y_1-\beta_0-\beta_1x_i)^2$$

$$
\frac{\partial Q}{\partial\beta_0}|_{\beta_0=\hat{\beta_0}}=-2\sum_{i=1}^n(y_1-\hat{\beta_0}-\hat{\beta_1}x_i)=0
$$
$$
\frac{\partial Q}{\partial\beta_0}|_{\beta_0=\hat{\beta_0}}=-2\sum_{i=1}^n(y_1-\hat{\beta_0}-\hat{\beta_1}x_i)=0
$$

# 最大似然估计(Maximum Likelihood Estimation, MLE)