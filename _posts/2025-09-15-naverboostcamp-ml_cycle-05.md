---
title: "[BoostCamp AI Tech / ML LifeCycle] 기초 신경망 이론 2: Backpropagation"
description: "Backpropagation의 과정을 학습하며, 편미분을 통해 Backpropagation의 결과 구하는 과정에 대한 내용 정리 포스트 입니다."

categories: [NAVER BoostCamp AI Tech, AI Core]
tags: [Linear Model, Neural Network, Backpropagation]

permalink: /naver-boostcamp/ml-life-cycle/05

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-09-16
last_modified_at: 2025-09-16
---

`Backpropagation`은 경사하강법을 사용하여 가중치를 업데이트하는 과정을 말한다.

## Computational Graph
-----------


Forward/backwrad propagation 과정을 그래프로 나타낸 것이다.

Computation graph를 이용해서 컴퓨터가 gradient를 간단하게 계산할 수 있도록 변환할 수 있다.

## Forward Pass
---------

$f(x, y, z) = (x + y)z$ 이며 $x = -2$, $y = 5$, $z = -4$ 라고 가정해보자.

위의 값을 통해 최종값을 구하는 과정을 `Forward Pass`라고 말한다.

## Backpropagation
--------

`Backpropagation`은 input이 입력되면 이에 따라 loss 값이 얼마나 변화하는지, 즉 loss를 입력된 파라미터에 대해 미분한 값을 계산하는 과정이다.

<img src="../assets/img/post/naver-boostcamp/backpropa_ex1.png">

- input: $x$, loss: $f$ &rarr; $\frac{\partial f}{\partial x}$
- input: $y$, loss: $f$ &rarr; $\frac{\partial f}{\partial y}$


## Chain Rule
-----------

<img src="../assets/img/post/naver-boostcamp/chain_rule.png">

- Upstream Gradient : loss를 output에 대해 미분
- Local Gradient : output을 input에 대해 미분
- Downstream Gradient: loss를 input에 대해 미분한 값으로 Upstream Gradient와 Local Gradient를 곱으로 표현된다.

## Logistic Regression: Backpropagation
--------

<img src="../assets/img/post/naver-boostcamp/backpropa_ex2.png">

## Patterns in Gradient Flow
---------

역전파를 진행하는 과정에서 몇몇 gate는 계산 과정에서 특정한 패턴을 가지고 있다.

<img src="../assets/img/post/naver-boostcamp/gradient_flow.png">