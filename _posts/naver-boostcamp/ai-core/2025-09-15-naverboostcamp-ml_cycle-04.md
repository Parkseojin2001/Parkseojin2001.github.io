---
title: "[BoostCamp AI Tech / ML LifeCycle] 기초 신경망 이론 1: Neural Network"
description: "Feature와 Image Feature 그리고 Neural Network의 원리가 어떻게 되는지 학습하며, MLP 그리고 Activation Function 대한 내용 정리 포스트 입니다."

categories: [NAVER BoostCamp AI Tech, AI Core]
tags: [linear model, neural network, deep learning]

permalink: /naver-boostcamp/ml-life-cycle/04

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-09-16
last_modified_at: 2026-11-11
---

## Linear Model
-------

- model function

$$
f(x, W) = Wx + b
$$

- loss function: softmax loss(분류)

$$
p(y = c_i | x) = \frac{e^{x_i}}{\sum_{j} e^{s_j}}
$$

- 가중치 업데이트: Stochastic Gradient Descent

1. Linear Classifiers의 단점은 각 클래스 당 하나의 template(카테고리)만 학습할 수 있다는 점이다.

    - ex. 자동차의 앞모습, 옆모습, 뒷모습의 이미지가 있지만 이 중 하나의 이미지만 학습이 가능하다.

2. 기하학적으로 직선 형태의 decision boundary들만 그릴 수 있다.

    - 복잡한 관계로 이루어진 두 개의 클래스들은 완벽하게 분리할 수 없다.

### Features

위의 단점을 해결하기 위한 방법이 `featurization` 이다. 

이 방법은 입출력 관계를 linear classifier로 직접 연결(mapping)하는 대신 입력을 표현할기 위해 몇 가지 특징들을 추출할 수 있다.

만약, feature space에서 입력이 선형 분류가 가능하다면 linear classifier이 잘 작동할 수 있다.

<img src="../assets/img/naver-boostcamp/ai-core/featurization.png">

즉, featurization은 데이터셋을 이용하려는 모델이 처리하기 쉬운 형태로 바꾸는 과정 말하고 이렇게 바꾼 데이터를 `feature` 라고 말한다.

이 feature는 원본 데이터는 유지한 채 다른 embedding space로 이동시킨 데이터다.

### Image Features

고전적인 image featurization은 직접 입출력(pixel-class) 관계를 mapping하는 대신 몇 가지 특징을 추출하여 입력(image)를 표현한다.

- Color histogram
- Histogram of oriented gradients (HoG)
- Bag of words (BoW) with a pre-defined dictionay (codebook)

이러한 방법의 단점은 데이터마다 각각의 다른 featurization 방법을 적용해야하는 수고가 필요하다.

### image Classifier with pre-extracted Features

위에서 말한 rule-base 방법을 요약하면 input 데이터에서 특징을 추출한 다음에 추출한 feature를 모델의 입력값으로 이용하여 ouput을 예측한다.

위의 방법에서 특징 추출하는 과정과 추출한 데이터를 모델에 입력값으로 사용하여 예측하는 과정을 하나의 과정으로 합치면 어떻게 될까?

하나의 과정으로 합치는 작업이 `Neural Network` 에 해당한다.

즉, `Neural Network`는 모델을 만들고 input에서 output을 출력하는 일련의 과정을 하나의 단일의 모델을 이용하는 방식을 의미한다. 

## Neural Networks
--------

Neural Network의 시작은 인간의 뉴런의 작동 방식을 모방하였다.

<img src="https://wikidocs.net/images/page/24958/%EB%89%B4%EB%9F%B0.PNG">

> "Neural networks are computing systems vaguely inspired by the biological neural networks that constitute animal brains." -- wekipedia

위키피디아의 정의는 인공신경망이 동물의 뇌신경망을 모방했다고 말한다. 그러나 최초에는 그렇게 시작했을지 몰라도, 요새 딥러닝에 사용되는 인공신경망은 다른 형태를 가진다.

인공신경망을 수학적으로 정의해보면 다음과 같다.

> Neural Networks are function approximators that stack affine transformations followed by nonlinear transformations.

"목적하는 결과를 도출해주는 어떤 함수 $f$에 근사하기 위해, 행렬-비선형 연산이 반복적으로 일어나는 함수"

### Perceptron

<img src="../assets/img/naver-boostcamp/ai-core/perceptrion.png">

1. 여러 input을 받는다.
2. 각각 input에 해당 가중치 파라미터를 곱한 결과값을 모두 더한다. 
3. 이 값을 activation function의 입력값을 넣어 최종 결과값을 도출한다.

### Single Layer

<img src="https://mblogthumb-phinf.pstatic.net/MjAxNzA2MTZfNDYg/MDAxNDk3NTc4Njk4MTM1.uEvkg1FYCUhTar5RcZSMZY5-U6swcECrpJALz5KwAA8g.JUB9h9ctN5KK-9zy7JxHpEBgRz_iB7At18obvWX03Skg.PNG.samsjang/%EC%BA%A1%EC%B2%98.PNG?type=w420">

위의 그림은 입력층과 출력층만 존재하는 1층짜리 신경망으로 `Logistic regression` 이다.
$$
f(\mathbf{x}) = \mathbf{W}\mathbf{x}
$$

- $\mathbf{x} \in \mathbb{R}^d$
- $\mathbf{W} \in \mathbb{R}^{c \times d}$
- $\mathbf{s} \in \mathbb{R}^c$



### Multi Layer


<img src="https://smartstuartkim.wordpress.com/wp-content/uploads/2019/02/mlp.png">

$$
f(\mathbf{x}) = \mathbf{W}_2(\mathbf{W}_1\mathbf{x})
$$

- $\mathbf{x} \in \mathbb{R}^d$
- $\mathbf{W_1} \in \mathbb{R}^{h \times d}$
- $\mathbf{W_2} \in \mathbb{R}^{c \times h}$
- $\mathbf{s} \in \mathbb{R}^c$

위의 MLP의 수식은 여전히 linear하다고 말할 수 있다.

$$
W = W_2W_1
$$


그러면 어떻게 해야 non-linear하게 만들 수 있을까?

만드는 방법은 `Activation functions` 를 이용하는 것이다.

$$
f(\mathbf{x}) = a_2(\mathbf{W}_2 a_1(\mathbf{W}_1\mathbf{x}))
$$

Activation functions의 종류는 아래와 같다.

<img src="https://machine-learning.paperspace.com/~gitbook/image?url=https%3A%2F%2F2327526407-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-legacy-files%2Fo%2Fassets%252F-LvBP1svpACTB1R1x_U4%252F-LvNWUoWieQqaGmU_gl9%252F-LvO3qs2RImYjpBE8vln%252Factivation-functions3.jpg%3Falt%3Dmedia%26token%3Df96a3007-5888-43c3-a256-2dafadd5df7c&width=768&dpr=4&quality=100&sign=27b81237&sv=2">


위의 내용을 통하여 Neural Network를 모델을 시각화하면 아래 그림과 같다.

<img src="https://courses.grainger.illinois.edu/cs440/fa2023/lectures/nn-figs/gormley-2layer-net.jpg">

## 신경망의 수식 표현
--------

### 행렬로 표현한 선형 모델

<img src="../assets/img/naver-boostcamp/pre-course/linear_model.png">

- **전체 데이터**가 모인 행렬 $\mathbf{X}$
    - $\mathbf{X}$의 한 **행벡터**는 하나의 점으로 표현되는 **데이터 포인트이다.
- $\mathbf{X}$의 데이터를 출력 $\mathbf{O}$로 보내주는 가중치 행렬 $\mathbf{W}$
    - $\mathbf{X}$의 데이터를 **다른 공간으로 보내주는** 역할을 한다.
- $y$ 절편에 해당하는 행벡터를 모든 행에 복제하여 만든 절편 행렬 $b$
    - 각 행들은 전부 같은 값 $[b_1, b_2, \cdots, b_p]$를 가진다.

이 때, 출력벡터 차원(열)은 기존의 $d$에서 $p$로 바뀌게 된다.

<img src="https://blogik.netlify.app/static/5551d2650565a6cdf3aaaa12e86d7a37/d52e5/d-to-p.png" width="400" height="200">

$x_1, x_2, \cdots x_d$는 행벡터에 들어있는 **$d$개의 변수이며 $p$개의 선형 모델을 만들어서 $p$개의 잠재변수를 설명하는 모델**과 동일한 의미이다.(행벡터 $\mathbf{x}$와 행벡터$\mathbf{o}$의 선형 결합)
- 위의 그림에서 **화살표**는 가중치 $\mathbb{W}$를 의미한다. 즉, 화살표의 개수는 $d \times p$가 된다.

### 소프트맥스(softmax) 연산

`소프트맥스` 함수는 **모델의 출력을 확률로 해석**할 수 있게 변환해주는 연산이다.

**분류 문제**를 풀 때 선형모델과 소프트맥스 함수를 결합하여 예측할 수 있다.

$$
\text{softmax}(o) = \left( \frac{\exp(o_1)}{\sum_{k=1}^{p} \exp(o_k)}, \cdots, \frac{\exp(o_p)}{\sum_{k=1}^{p} \exp(o_k)} \right)
$$

- 선형 모델의 출력 $\mathbf{O}$의 행벡터를 softmax함수의 $o$로 집어넣는다.
- softmax 함수는 입력을 확률벡터로 변환시켜 출력한다.
    - 주어진 데이터가 특정 클래스에 속할 확률이 얼마인지 계산한다.

Python으로 구현한 코드는 다음과 같다.

```python
def softmax(vec):
    # 수식과 달리, np.max값을 취해주는 과정이 추가되어 있다.
    # softmax는 지수함수를 취하므로, 너무 큰 값이 들어왔을 경우 overflow가 생길 위험이 있다.
    # 이를 방지하기 위해 np.max 값을 vector에다 빼 준 뒤 해당 값을 input으로 적용한다.
    denumerator = np.exp(vec - np.max(vec, axis=-1, keepdims=True))
    numerator = np.sum(denumerator, axis=-1, keepdims=True)
    val = denumerator / numerator
    return val

vec = np.array([[1,2,0], [-1,0,1], [-10,0,10]])
softmax(vec)
'''
array([[2.44728471e-01, 6.65240956e-01, 9.00305732e-02],
       [9.00305732e-02, 2.44728471e-01, 6.65240956e-01],
       [2.06106005e-09, 4.53978686e-05, 9.99954600e-01]])
'''
```

선형 모델 결과값은 대부분 확률벡터가 아닌 경우가 많기 때문에 softmax 함수를 이용해 확률벡터로 변환하여 해석을 한다.

단, 소프트맥스 함수는 학습 시에만 사용하고, 추론을 할 때에는 사용하지 않는다.

추론을 할 때에는 **출력값에서 최댓값을 가진 주소만 1로 출력**하는 `원-핫(one-hot)` 벡터를 사용하여, **주어진 출력 중 최댓값 주소만 가져가는 형태로 구현**하기 때문이다.

선형 모델로 나온 출력값에 소프트맥스 함수를 적용시켜 원하는 분류문제로 바꾸었듯이 **다른 활성함수를 합성하여 출력값을 조정하여, 원하고자 하는 의도로 바꿔서 해석**할 수 있다.

이러한 방법을 이용하여 비선형 함수를 모델링할 때 사용할 수 있다.

### 활성함수(activation function)

`활성함수`는 실수값을 입력으로 받아서 비선형(nonlinear)으로 변환시킬 수 있다.
- 

<img src="../assets/img/naver-boostcamp/pre-course/activation_function.png">

- 선형모델을 입력으로 받아서 각각의 원소에 대하여 적용된다.
- 활성함수 $\sigma$는 비선형 함수로 , 입력 `잠재벡터` $\mathbf{z} = (z_1, \cdots, z_q)$의 각 노드에 개별적으로 적용하여 새로운 `잠재 벡터` $\mathbf{H} = (\sigma(z_1), \cdots, \sigma(z_n))$를 만든다.
    - 이 잠재벡터 $\mathbf{H}$를 `히든(Hidden) 벡터` 또는 `뉴런(Neuron)`이라 부르기도 한다.
    - 이런 뉴런의 집합체를 `신경망`, `Neural Network`라고 부른다.

활성함수는 $\mathbb{R}$위에 정의된 **비선형(nonlinear)함수**로서 딥러닝에서 매우 중요하다. 

**활성함수를 쓰지 않으면 딥러닝은 선형모형과 차이가 없다.**

<img src="https://studymachinelearning.com/wp-content/uploads/2019/10/summary_activation_fn.png">


과거에는 활성함수로 `시그모이드(sigmoid)` 함수나 `하이퍼볼릭탄젠트(tanh)` 함수를 많이 사용되었지만, 오늘날 딥러닝에 가장 많이 쓰이는 함수는 `ReLU` 함수로, 선형함수처럼 보이지만 전형적인 비선형함수이며 비선형함수로서의 좋은 성질들을 많이 가지고 있다.

### 다층 퍼셉트론(MLP)

`신경망`은 `선형모델`과 `활성함수`를 합성한 함수이며, 이러한 `신경망`이 **여러층 합성된 함수**를 `다층(multi-layer) 퍼셉트론, MLP`라고 한다.

<img src="https://cdn-images-1.medium.com/max/800/0*eaw1POHESc--l5yR.png">

$$
\begin{align*}
\vdots\\
\mathbf{H}^{(l)}&= \sigma(\mathbf{Z}^{(l)}) \\
\mathbf{Z}^{(l)} &= \mathbf{H^{(l-1)}}\mathbf{W}^{(l)} + \mathbf{b}^{(l)}\\
\vdots \\
\mathbf{H}^{(1)}&= \sigma(\mathbf{Z}^{(1)}) \\
\mathbf{Z}^{(1)} &= \mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}
\end{align*}
$$

`다층(multi-layer) 퍼셉트론, MLP`는 이처럼 **[선형변환 - 활성함수 합성] 사이클을 n회 반복하여 신경망이 여러층 합성된 함수**이다.
- MLP의 파라미터는 $L$개의 가중치 행렬 $W^{(L)},...,W^{(1)}$과 $y$절편 $b^{(L)},...,b^{(1)}$로 이루어져 있다.
- 이 때 $l = 1,...,L$까지의 순차적인 신경망 계산을 `순전파(forward propagation)`이라고 부른다.

#### 층을 여러 개 쌓는 이유

이론적으로는 2층 신경망으로도 임의의 연속함수를 근사할 수 있다.
- 이를 `universal approximation theorem`이라고 한다.

그러나 층이 깊을수록 **목적함수를 근사하는데 필요한 뉴런(노드)의 숫자가 훨씬 빨리 줄어들어 좀 더 효율적으로 학습이 가능**하다. 즉, 바꾸어 말하면 **적은 수의 뉴런으로도 층을 더 깊게 쌓으면 훨씬 더 복잡한 함수를 표현할 수 있다.**
- 층이 얇으면 필요한 뉴런의 숫자가 기하급수적으로 늘어나, 넓은(wide) 신경망이 되어야한다.

> 주의할 점은, 층이 깊을수록 최적화가 더 어려워져 학습하기가 어려워진다는 것이다.

## 손실 함수는 어떻게 고를까
--------


`Regression Task`에는 MSE(Mean Squared Error)를 사용한다는 것을 언급했었다. 오차제곱($L_2−loss$)을 이용해서 추정치와 정답의 차이를 도출해낼 수 있다.

$$
MSE = \frac{1}{N} \sum_{i=1}^{N} \sum_{d=1}{D} (y_i^{(d)} - \hat{y}_i^{(d)})^2
$$

굳이 오차제곱을 사용하는 이유가 있을까? 오차제곱이 아니라 절댓값($L_1 - loss$)을 사용할 수 없을까?

실제로는 절댓값을 사용하는 것과 오차제곱을 사용하는 경우에 차이가 있다. 절댓값을 사용하는 방식을 `MAE(Mean Absolute Error)`라고 한다.

- `MAE`를 사용하는 경우
    - 데이터에는 에러, outlier가 있기 마련이다. 그런데 에러가 있는 데이터가 너무 많다면, 그 데이터에 맞추려다가 모델 전반의 학습이 망가지게 된다.
    - MAE는 이 경우에 outlier에 대해 큰 페널티를 주지 않으므로, **튀는 데이터값을 무시하고 강건한 모델링**을 할 수 있다.
    - 다만, **outlier가 너무 많다면 학습이 망가지게 된다.**
- `MSE`를 사용하는 경우
    - MAE에 비하여 MSE는 **튀어나온 값(평균과 차이가 있는 값)에 더 큰 페널티**를 주므로, outlier들을 수용하는 모델링을 하기에 좋다.
    - 절댓값보다 **컴퓨터 연산이 더 쉽다.**

이처럼 **상황에 따라 Loss Function을 다르게 사용**한다. 

이번엔, `Classification Task`를 살펴보자.

분류문제에서는 주로 **`Cross Entropy`**가 Loss Function으로 사용된다. 굳이 크로스엔트로피를 사용하는 이유가 있을까?

$$
CE = -\frac{1}{N} \sum_{i=1}^{N} \sum_{d=1}^D y_i^{(d)} log \ \hat{y}_i^{(d)}
$$

분류 문제에서 정답 레이블은 일반적으로 **원-핫 벡터**로 표기된다. 즉 크로스 엔트로피 식에서 $D$개의 $y_i^{(d)}$ 중 유효한 값은 하나 밖에 없다. 그러므로 **내 모델의 출력값 중에서 정답과 일치하는 클래스의 값만 높여주겠다**는 의미이다. 얼마나 높일지가 중요한 게 아니라, 해당 차원의 값만 키워주는것이 목적이 된다.

마지막으로, `Probability Task`를 살펴보자.

확률문제의 Loss Function으로는 **`MLE(Maximum Likelihood Estimation)`**가 많이 쓰인다.

$$
MLE = \frac{1}{N} \sum_{i=1}^N \sum_{d=1}^D log \ \mathcal{N} (y_i^{(d)}; \hat{y}_i^{(d)}, 1) \ (= MSE)
$$

예를 들어 회귀 문제를 풀었을 때, output을 단순히 출력값이 아닌 확률로 내보내는 모델을 만들고 싶다면 MLE를 사용한다.

가장 높은 가능성을 가진 예측 모델을 찾아야하기 때문이다.

만약 단순히 최고가능성만 따지는게 아니라, 해당 판단이 얼마나 확률적으로 높고, 얼마나 신뢰성이 있는지 등의 정보가 필요하다면 다른 손실함수를 사용하기도 한다.

## Computing Gradients
--------

### 1차원 선형 모델로 먼저 보기

<img src="https://datacorner.fr/wp-content/uploads/2018/05/RG1.png" width="400" height="300">

- Data: $\mathcal{D} = \{ (x_i, y_i) \} _{i=1}^N$
- Model : $\hat{y} = wx + b$
- Loss : $loss = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$

우리의 목적은 Loss를 최소화하는 것이므로, `Loss Function`을 $w$에 대해 편미분하여, Loss가 작아지는 방향으로 $w$를 업데이트해준다. 

$$
\begin{align*}
\frac{\partial loss}{\partial w} &= \frac{\partial}{\partial w} \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 \\ 
&= \frac{\partial}{\partial w} \frac{1}{N} \sum_{i=1}^{N} (y_i - wx_i - b)^2 \\
&= - \frac{1}{N} \sum_{i=1}^{N} 2(y_i - wx_i - b)x_i \\ 
\\
w &\leftarrow w - \eta \frac{\partial loss}{\partial w}
\end{align*}
$$

$b$에 대해서도 동일한 과정을 거친다.

$$
\begin{align*}
\frac{\partial loss}{\partial b} &= \frac{\partial}{\partial b} \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 \\ 
&= \frac{\partial}{\partial b} \frac{1}{N} \sum_{i=1}^{N} (y_i - wx_i - b)^2 \\
&= - \frac{1}{N} \sum_{i=1}^{N} 2(y_i - wx_i - b)x_i \\
\\
 b &\leftarrow b - \eta \frac{\partial loss}{\partial b}
\end{align*}
$$

이 방식을 `Gradient Descent`라고 하고, 만약 Loss Function이 아니라 `Reward Function`이어서 어떤 값을 키우고싶다면 `Gradient Ascent`를 사용한다.

그런데, Layer는 한 층만 있는게 아니라 여러 층이 있다.

**마지막 층에서 나왔던 Loss Function 값을 전체 파라미터로 다 미분을 하는 것**이 **`역전파`**이고, **역전파로 나오는 각 파라미터의 편미분을 업데이트 시키는 것**이 실제 딥러닝에서의 **`경사 하강법`**이 된다.

이 때, 업데이트의 `Step size(learning rate, $\eta$)`가 너무 크면, 학습 중에 점프하는 크기가 너무 커져 minimum 값에 접근할 수 없고, 결국 학습이 아예 되지 않는다. 따라서 적절한 step size를 잡는 것이 중요한데, 이 때문에 `adaptive learning rate` 같은 방법이 생겨났다.

### 신경망에서의 그레디언트


(Stochastic) GD에서는 **classification loss에 대한 그레디언트**를 필요로 한다.

- 업데이트해야할 파라미터

    $$
    \frac{\partial L}{\partial W_1}, \frac{\partial L}{\partial W_2}
    $$

<img src="../assets/img/naver-boostcamp/ai-core/computing_gradients.png">


이 과정을 코드로 구현하면 아래와 같이 구현할 수 있다.

```python
import numpy as np
from numpy.random import randn

n, d, h, c = 64, 1000, 100, 10

x, y = randn(n, d), randn(n, c)
w1, w2 = randn(d, h), randn(h, c)
learning_rate = 1e-4

for t in range(1000):
    y_0 = x.dot(w1)
    h_0 = 1 / (1 + np.exp(-y_0))
    y_pred = h_0.dot(w2)
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h.T.dot(grad_y_pred)
    grad_h = grad_y_pred.dot(w2.T)
    grad_w1 = x.T.dot(grad_h * h * (1 - h))

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
```