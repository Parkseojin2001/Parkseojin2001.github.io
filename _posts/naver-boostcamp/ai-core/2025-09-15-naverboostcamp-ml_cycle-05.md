---
title: "[BoostCamp AI Tech / ML LifeCycle] 기초 신경망 이론 2: Backpropagation"
description: "Backpropagation의 과정을 학습하며, 편미분을 통해 Backpropagation의 결과 구하는 과정에 대한 내용 정리 포스트 입니다."

categories: [NAVER BoostCamp AI Tech, AI Core]
tags: [linear model, neural network, backpropagation]

permalink: /naver-boostcamp/ml-life-cycle/05

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-09-16
last_modified_at: 2026-11-11
---

`Backpropagation`은 경사하강법을 사용하여 가중치를 업데이트하는 과정을 말한다.

## 역전파 알고리즘의 원리
-----------

`순전파(forward propagation)`는 입력값 $\mathbf{X}$를 받아서 선형모델과 활성함수를 반복적으로 적용하여 출력하는 연산이었다.

이 때, 가중치 $\mathbf{W}$를 학습시키려면 각각의 가중치에 대한 gradient 벡터를 계산해야한다.
- [선형회귀에서의 경사하강법](/naver-boostcamp/gradient-descent/02)에서 $\beta$를 구하던 것과 같은 개념이다.

이 과정을 `역전파(backpropagation)` 알고리즘으로 수행한다.
- 각 충에 존재하는 파라미터들에 대한 미분을 계산해서, 그 미분 값을 가지고 파라미터를 업데이트한다.

가중치를 업데이트할 때, **행렬의 모든 원소개수만큼, 또 $y$절편의 모든 원소 개수만큼 경사하강법이 적용**된다. 따라서 기존의 선형모델보다 훨씬 더 많은 파라미터들에 대해 경사하강법을 적용하게 된다.

또, 선형모델은 한 층에 대해서만 계산하는 모델이므로 gradient 벡터를 한번에 계산할 수 있었지만, 딥러닝은 여러 층에 걸쳐 순차적으로 계산하기 때문에 gradient 벡터를 한번에 계산할 수 없다.

따라서 `역전파` 알고리즘은 `순전파`와 비슷하게 **역순차적으로 층마다 미분을 계산하여 적용**시킨다.

### 위층에서 아래층으로 전달되는 그레디언트

- 딥러닝은 `역전파(backpropagation)`을 이용하여 각 층에 사용된 파라미터 $\mathbf{W}^{(l)}$, $\mathbf{b}^{(l)}$ 를 학습한다.
- 각 층 파라미터의 그레이디언트 벡터는 윗층부터 역순으로 계산하게 된다.

<img src="../assets/img/naver-boostcamp/pre-course/backpropagation.png">

- 각 층에서 계산된 gradient 벡터들은 **밑의 층으로 전달**되는 flow 형태이다.
    - 저층에 있는 gradient 벡터를 계산할 때는 위층에 있는 gradient 벡터가 필요하다.
- **위층에서 아래층으로 내려오면서** 업데이트하는 방식이다.

이 원리를 사용할 때 **합성함수의 미분법**인 `연쇄법칙`을 사용하여 gradient 벡터를 전달한다.

`연쇄법칙(chain-rule)`은 합성함수를 미분하는 방식이다. 오늘날의 딥러닝 프레임워크들은 이러한 연쇄법칙을 기반으로 한 `자동 미분(auto-differentiation)`을 수행하여 신경망을 학습시킨다.

> 역전파 알고리즘은 각 뉴런의 값(텐서 값)이 컴퓨터 메모리에 저장되어야 역전파 알고리즘에 사용할 수 있다. 즉, x에 대해 미분하고싶다면 x와 y값을 알고있어야만 미분 가능하다. 따라서 역전파는 순전파보다 다소 메모리를 많이 사용하게 되는 방법이다.

<img src="../assets/img/naver-boostcamp/pre-course/backpropagation_calculation.png">

- 위의 그림에서, **파란 회살표**는 `순전파`, **빨간 화살표**는 `역전파`를 의미한다.

각각의 연쇄법칙을 순서대로 적용할 때, 미분값이 각 층마다 계산되므로, 경사하강법에서 실제 가중치 행렬 $W^(1)$에 적용시킬 gradient 벡터를 찾아낼 수 있다.

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

<img src="../assets/img/naver-boostcamp/ai-core/backpropa_ex1.png">

- input: $x$, loss: $f$ &rarr; $\frac{\partial f}{\partial x}$
- input: $y$, loss: $f$ &rarr; $\frac{\partial f}{\partial y}$


## Chain Rule
-----------

<img src="../assets/img/naver-boostcamp/ai-core/chain_rule.png">

- Upstream Gradient : loss를 output에 대해 미분
- Local Gradient : output을 input에 대해 미분
- Downstream Gradient: loss를 input에 대해 미분한 값으로 Upstream Gradient와 Local Gradient를 곱으로 표현된다.

## Logistic Regression: Backpropagation
--------

<img src="../assets/img/naver-boostcamp/ai-core/backpropa_ex2.png">

## Patterns in Gradient Flow
---------

역전파를 진행하는 과정에서 몇몇 gate는 계산 과정에서 특정한 패턴을 가지고 있다.

<img src="../assets/img/naver-boostcamp/ai-core/gradient_flow.png">