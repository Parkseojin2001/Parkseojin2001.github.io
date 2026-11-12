---
title: "[BoostCamp AI Tech / Pre-Course 2] 딥러닝 기본 - Convolutional Neural Networks"
description: "합성곱 연산의 원리와 수학적 의미, Stride·Padding·채널·팽창, 풀링과 완전 연결 계층, 파라미터 수와 역전파, 텍스트에 쓰는 1차원 합성곱까지 CNN의 기초를 정리한 포스트입니다."

categories: [NAVER BoostCamp AI Tech, AI Core]
tags: [naver-boostcourse, pytorch, cnn, nlp, cv]

permalink: /naver-boostcamp/dl-basic/03/

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-08-04
last_modified_at: 2026-11-11
---

**합성곱 신경망(Convolutional Neural Network, CNN)**은 주로 이미지 인식과 같은 컴퓨터 비전 분야의 데이터를 분석하기 위해 사용되는 인공 신경망의 한 종류다. 합성곱 신경망은 입력 데이터의 지역적인 특징을 추출하는 데 특화된 구조를 갖고 있으며 이를 위해 **합성곱(Convolution)** 연산을 사용한다.

합성곱 연산은 이미지 특정 영역에서 입력값의 분포 또는 변화량을 계산해 출력 노드를 생성한다. 특정 영역 안에서 연산을 수행하므로 **지역 특징(Local Features)**을 효과적으로 추출할 수 있다.

이미지 데이터는 고정된 프레임 내에 객체들의 위치와 형태가 자유분방하므로 여러 영역의 지역 특징을 조합해 입력 데이터의 전반적인 **전역 특징(Global Features)**을 파악할 수 있다.

합성곱 신경망은 원래 컴퓨터비전을 위해 고안되었지만, 자연어 처리에서도 사용되기 시작했다. 이를 통해 입력 데이터의 길이에 상관없이 병렬 처리가 가능하고, 학습에 필요한 가중치 수를 줄여 깊은 신경망을 구성할 수 있게 됐다.

## MLP에서 Convolution으로
-------

기존의 `다층신경망(MLP)`는 각 뉴런들이 선형모델과 활성함수로 **모두 연결된(fully connected)**구조였다.

<img src="../assets/img/naver-boostcamp/pre-course/fully_connected.png">

이런 다층신경망의 경우는 **각 성분 $h_i = \sigma(\sum_{j=1}^{p} W_{ij} x_j)$ 에 대응하는 가중치 행 $W_i$ 가 필요**했다. 

만약에 $i$가 바뀌면 사용되는 가중치도 바뀌게 되므로 가중치 행렬의 크기가 굉장히 크며 이에 따라 학습해야할 파라미터도 많다.

### 커널을 움직이며 계산한다

`Convolution 연산`은 기존의 MLP 방식과는 달리, `커널(kernel, 필터)`을 **입력벡터 상에서 움직여가면서** 선형 모델과 합성함수가 적용되는 구조이다.

<img src="../assets/img/naver-boostcamp/pre-course/kernel_image.png">

- `커널(kernel, 필터)` : 고정된 가중치 행렬
- `입력벡터` : 커널 사이즈에 해당되는 부분만 사용한다.

이 방식또한 활성화 함수를 제외한 Convolution 연산도 선형변환에 속한다.

$$
h_i = \sigma \bigg( \sum_{j=1}^k V_j x_{i+j-1} \bigg)
$$

- $h_i$는 $i$번째 히든레이어를 의미한다.
- $k$는 커널(필터) 사이즈를 의미한다.
- $V$는 가중치 행렬, 즉 커널을 의미한다. $V$는 모든 $j$에 대해서 같다.
    - 이것이 기존 방식과의 가장 큰 차이점이다.

$i$의 수와 상관없이 커널 크기 그대로 공통적으로 사용하기 때문에 **파라미터 사이즈를 많이 줄일 수 있다.**

### 수학적 의미

Convolution 연산의 수학적 의미는 **커널을 이용해 신호(signal)을 국소적으로 증폭 혹은 감소시켜 정보를 추출 또는 필터링** 하는 것이다. 엄밀히 말하면 CNN에서 사용하는 연산은 convolution이 아니고 `cross-correlation`이라고 부르지만, 관습적으로 Convolution으로 부르곤 한다.

$$
\begin{align*}
\text{continuous} \ \ \ [f * g](x) &= \int_{\mathbb{R}^d} f(z)g(x+z) \, dz = \int_{\mathbb{R}^d} f(x+z)g(z) \, dz = [g * f](x) \\
\text{discrete} \ \ \ [f * g](i) &= \sum_{a \in \mathbb{Z}^d} f(a)g(i+a) = \sum_{a \in \mathbb{Z}^d} f(i+a)g(a) = [g * f](i)
\end{align*}
$$

- $x + z$ 또는 $i + a$가 들어간 함수값은 신호를 의미한다.
- $z$ 또는 $a$만 들어간 함수 값은 커널을 의미한다.

연속변수이든 이산변수이든, 합과 적분의 차이를 제외하고는 연산이 거의 동일하다.

CNN을 수식만으로 이해하기는 어려우므로, graphical하게 이해해보자.

<img src="../assets/img/naver-boostcamp/pre-course/convolution_graph.png">

커널은 **정의역 내에서 움직여도 변하지 않고(translation invariant), 주어진 신호에 국소적(local)으로 적용**된다. 이를 `locality`가 있다고 말한다.

### 영상처리

Convolution Network의 커널을 적용할 때 어느 위치를 찍느냐에 따라 하나의 이미지를 여러 형태(blur, emboss, outline 등)로 해석할 수 있다.

## Convolution 연산
---------

- Continuous convolution

$$
(f * g)(t) = \int f(\tau)g(t - \tau) \ d\tau = \int f(t - \tau)g(t) \ d \tau
$$

- Discrete convolution

$$
(f * g)(t) = \sum_{i = -\infty}^{\infty} f(i)g(t-i) = \sum_{i = -\infty}^{\infty} f(t - i)g(i)
$$

<img src="../assets/img/naver-boostcamp/pre-course/convolution-cal.png" width = "400" height="300">

- 2D image convolution
    - 적용하고자 하는 필터에 따라 같은 이미지에 대해 결과값이 Blur, Emboss, Outline 등 다른 결과의 이미지가 출력된다.

$$
(I * K)(i, j) = \sum_{m}\sum_{n} I(m, n)K(i - m, j - n) = \sum_{m}\sum_{n}I(i - m, j - n)K(m, n)
$$

<img src="../assets/img/naver-boostcamp/pre-course/2d-image-convolution.png" width="400" height="300">

### 다차원에서의 Convolution 연산

**데이터의 성격에 따라 사용하는 커널을 다르게** 하며, 다차원으로 Convolution 연산을 할 수 있다.

$$
\begin{align*}
\text{1D-conv} \ \ [f * q](i) &= \sum_{p=1}^{d} f(p)g(i + p) \\
\text{2D-conv} \ \ [f * q](i, j) &= \sum_{p, q} f(p, q)g(i+p, j+q) \\
\text{3D-conv} \ \ [f * q](i, j, k) &= \sum_{p, q, r} f(p, q, r)g(i + p, j + q, k + r)
\end{align*}
$$

- 여기서 $f$는 커널, $g$는 입력을 뜻한다.
- 데이터의 종류
    - 1차원 데이터 - 음성, 텍스트 등
    - 2차원 데이터 - 흑백 영상 등
    - 3차원 데이터 - 컬러 영상 등

다만, 중요한 것은 $i$,$j$,$k$ **위치가 바뀌더라도 커널 $f$의 값은 바뀌지 않는다**는 점이다.

### 2차원 Convolution 연산 이해하기

2차원에서는 1차원과 달리 입력데이터의 형태가 **행렬형태**이다. 따라서 1차원에서처럼 한쪽 방향으로 한칸씩 움직이면서 커널을 이동시켰던것과 달리, **커널을 x축과 y축 방향으로 한칸씩 움직여가며 적용**시킨다.

<img src="../assets/img/naver-boostcamp/pre-course/2d-convolution.png">

입력 크기를 $(H,W)$, 커널 크기를 $(K_H, K_W)$, 출력 크기를 $(O_H, O_W)$라 하면, 출력 크기는 다음과 같이 계산한다.

$$
\begin{align*}
O_H &= H - K_H + 1 \\
O_W &= W - K_W + 1
\end{align*}
$$

가령, 28x28 입력을 3x3 커널로 2차원 Convolution 연산을 하면 26x26이 된다.

그러나 실제로 이미지 분석을 하는 경우에는, 2차원이지만 채널이 여러개라 마치 3차원 같은 입력데이터(ex-RGB)가 들어오게 된다. 이런 경우는 2차원 Convolution을 만들어 적용하는데, **입력값의 채널 개수만큼 커널 채널을 만들어주는 것**이 포인트이다. 즉, 입력값의 채널 개수와 커널의 채널 개수가 같아야한다.

<img src="../assets/img/naver-boostcamp/pre-course/3d-convolution_1.png">

입력 채널 개수와 커널 채널 개수를 같게하여 합성곱연산을 수행하면, 각 채널의 연산들이 모두 더해지기 때문에 채널이 1이 되어 출력 값의 크기는 $(O_H, O_W, 1)$이 된다.

<img src="../assets/img/naver-boostcamp/pre-course/3d-convolution_2.png">

만약 출력 채널의 개수를 여러개로 만들고 싶다면 어떻게 하면 될까? **커널 텐서 자체의 개수를 여러개로 하면된다.**

<img src="../assets/img/naver-boostcamp/pre-course/2d-conv.png">

위와 같이 3차원 커널의 개수를 $O_c$개 사용하면, 출력도 3차원 텐서로 얻을 수 있다. 이것이 오늘날의 CNN 연산에서 사용하는 Convolution 연산의 기본적인 형태이다.

### CNN의 구성

`CNN`은 `합성곱 계층(convolution layer)`과 `풀링 계층(pooling layer)`, 그리고 `전연결 계층(fully-connected layer)`으로 이루어져 있는 신경망으로, 이미지 등을 처리하는데에 자주 사용되는 모델이다.

- 합성곱 계층, 풀링 계층 : feature를 추출한다.
- 전연결 계층 : 분류/회귀 문제에 대해 decision making한다(출력값을 만든다)

<img src="../assets/img/naver-boostcamp/pre-course/basic-convolution.png">

그러나 모델의 파라미터가 너무 많아지면 Generalization 이슈가 생기기 때문에, **최근에는 전연결계층을 없애려고 하는 추세**이다.

> 네트워크의 레이어 별로 몇 개의 파라미터로 구성되었는지 전체 네트워크의 총 파라미터가 몇 개인지를 알고있는 것이 중요하다.

## Stride, Padding, 채널, 팽창
---------

### Stride(간격)

`Stride`는 '넓게 걷는다'라는 뜻으로, **매번 커널을 찍을때 얼마나 이동할것인가**를 의미한다. stride가 클수록 output의 사이즈는 작아지게 된다.

<img src="https://miro.medium.com/v2/resize:fit:1400/1*zUDC6dNc3rIOwTfRg3sEBw.png" width="500" height="200">

- 간격을 작게 설정하면 입력 데이터의 공간적인 정보를 보존
- 간격을 크게 설정하면 입력 데이터의 공간적인 정보를 감소

간격을 조절해 합성곱 신경망이 학습해야 하는 모델 매개변수의 수를 감소시킬 수 있으며, 이를 통해 모델의 복잡도를 낮추고 과대적합을 방지할 수 있다.

### Padding(패딩)

`패딩(Padding)`은 입력 값의 가장자리를 커널로 찍기 위해 덧대는 공간이다. 일반적으로 해당 패딩 위치에 0을 넣는 제로패딩을 사용한다.

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRxTZKpsJRKk7bYZ_Jeo9oKs5j75XT4U5PVWQ&s">

커널의 크기를 어떻게 하느냐에 따라 패딩의 크기가 달라지기도 한다. 예를 들어, 3x3 커널에서는 패딩을 1칸만 추가해도 가장자리를 찍을 수 있겠지만, 5x5 커널에서는 모서리를 커널의 중심으로 두고 가장자리에 커널을 찍기위해 패딩을 2칸은 추가해야할 것이다.

Stride와 Padding을 모두 적용하면 아래 그림과 같이 학습이 진행된다.

<img src="../assets/img/naver-boostcamp/pre-course/cnn-stride-padding.png">

패딩이 없으면 특징 맵이 계속 작아진다. 이때 생기는 문제는 다음과 같다.

**작은 특징 맵의 문제점**<br>
- 합성곱 신경망을 더 깊게 쌓는 데 제약사항이 될 수 있으며, 합성곱 신경망 성능에 악영향을 끼칠 수 있다.
- 가장자리에 있는 정보는 다른 위치에 있는 정보에 비해 학습하기가 더 어렵다. 
- 가장자리와 필터의 합성곱 연산 횟수가 중심과 필터의 합성곱 연산 횟수보다 적어 정보가 학습되는 데 제한이 있을 수 있다.

### 채널

입력 데이터와 필터 간의 연산은 **채널(Channel)**에서 수행된다. 채널은 입력 데이터와 필터가 3차원으로 구성되어 있을 때 같은 위치의 값끼리 연산되게 한다. 이를 통해 입력 데이터의 공간 정보를 유지하면서 추출되는 특징을 확장할 수 있다.

채널 개수는 일반적으로 합성곱 계층에서 설정되며, 이는 모델의 구조나 목적에 따라 달라진다. 채널의 개수가 많아질수록 학습할 수 있는 특징의 다양성이 증가해 모델의 **표현력(Representational Power)**이 높아지는 효과를 가져온다.

<img src="https://wikidocs.net/images/page/64066/conv15.png">


출력 채널이 많은 경우, 각 체널은 입력 데이터에서 서로 다른 특징을 학습할 수 있다. 따라서 모델은 더 많은 종류의 특징을 학습하게 되며, 그로 인해 더 복잡한 문제를 해결할 수 있는 능력을 갖추게 된다.

그러나 출력 채널이 많을수록 모델의 매개변수가 많아지므로 학습 시간과 메모리 사용량이 증가하는 단점이 있다.


### 팽창

**팽창(Dilation)**이란 합성곱 연산을 수행할 때 입력 데이터에 더 넓은 범위의 영역을 고려할 수 있게 하는 기법이다. 팽창은 필터와 입력 데이터 사이에 간격을 두는 방법이다.

<img src="../assets/img/deep-learning/computer-vision/dilated_convolution.png">

팽창은 입력 데이터의 각 픽셀이 출력에 미치는 영향이 강화되는 효과를 갖는다. 팽창은 필터의 크기를 키우지 않고 입력 데이터에 더 넓은 영역을 고려할 수 있게 해 더 깊고 복잡한 모델을 구성할 수 있게 한다. 

하지만 필터가 바라봐야 하는 입력 데이터의 범위가 커지므로 오히려 연산량이 늘어날 수도 있다. 또한 팽창 크기가 너무 크다면 인접한 픽셀값을 고려하지 않게 되므로 공간적인 정보가 보존되지 않아 특징 추출의 효과가 떨어질 수 있다.

### PyTorch: `nn.Conv2d`

```python
# 2차원 합성곱 계층 클래스
conv = torch.nn.Conv2d(
  in_channels,
  out_channels,
  kernel_size,
  stride=1,
  padding=0,
  dilation=1,
  groups=1,
  bias=True,
  padding_mode="zeros"
)
```

합성곱 계층의 출력 크기 계산 방법은 아래와 같다.

$$
L_{out} = \Bigg[\frac{L_{in} + 2 \times {padding} - {dilation} \times (kernel\_size - 1) - 1}{stride} + 1 \Bigg]
$$

## 활성화 맵
---------

**활성화 맵(Activation Map)**은 합성곱 계층의 특징 맵에 활성화 함수를 적용해 얻어진 출력값을 의미한다. 
- 합성곱 계층에서 입력 이미지와 필터의 합성곱 연산을 통해 특징 맵을 추출하면 이 값에 비선형성을 추가하기 위해 활성화 함수를 적용한다.
  - ReLU 함수가 적용되며, 이를 통해 특징 맵의 값이 0보다 크면 그 값을 그대로 출력하고, 0 이하일 경우에는 0을 출력한다.

이 과정을 거친 활성화 맵은 다음 계층의 입력값으로 사용된다. 다음 계층이 합성곱 계층이라면 동일한 방식을 거친다. 합성곱 연산과 활성화 함수를 여러 번 반복하여 신경망을 구성하면, 입력 이미지에서 추출된 추상적인 특징을 학습할 수 있게 된다.

*만약 합성곱 계층의 출력값에 활성화 함수를 적용하지 않으면?*<br>
모델이 선형적인 결합만 수행하게 되어 복잡한 패턴이나 추상적인 특징을 학습하는 것이 어려워진다.

따라서 활성화 함수를 적용함으로써 모델이 비선형성을 가지게 되어 입력 데이터에서 다양한 추상적인 특징을 학습할 수 있게 된다.

## 풀링
---------

**풀링(Pooling)**은 특징 맵의 크기를 줄이는 연산으로 합성곱 계층 다음에 적용된다. 
- 특징 맵의 크기를 줄여 연산량을 감소시키고 입력 데이터의 정보를 압축하는 효과를 가진다.
- 필터와 간격을 이용하며 일정한 크기의 필터 내 특정 값을 선택한다.
  - 최댓값 풀링(Max Pooling): 특정 크기의 필터 내 원소값 중 가장 큰 값을 선택해 특징 맵의 크기를 감소시키는 방법
  - 평균값 풀링(Average Pooling): 필터 내 원소값들의 평균값으로 특징 맵의 크기를 감소시키는 방법

<img src="https://www.researchgate.net/publication/333593451/figure/fig2/AS:765890261966848@1559613876098/llustration-of-Max-Pooling-and-Average-Pooling-Figure-2-above-shows-an-example-of-max.png">

**장점**<br>
- 입력 데이터의 공간적 크기를 줄이기 때문에 계산 비용을 감소시킬 수 있다.
- 입력 데이터의 특징 위치가 변경되더라도 인근 영역에 대한 연산을 적용하기 때문에 공간적 정보를 유지할 수 있다.

**단점**<br>
- 입력 데이터의 위치 정보를 일부 손실
- 세밀한 위치 정보가 필요한 작업에서는 성능 저하를 초래할 수 있다. 

최근에는 풀링을 이용해 공간적 크기를 감소하는 방법보다 연산량이 많더라도 합성곱 계층의 간격을 설정해 입력 데이터의 공간적인 크기를 줄이는 방법을 사용한다.

### 풀링 클래스

```python
# 2차원 최댓값 풀링 클래스
max_pool = torch.nn.MaxPool2d(
  kernel_size,
  stride=None,
  padding=0,
  dilation=1
)

# 2차원 평균값 풀링 클래스
avg_pool = torch.nn.AvgPool2d(
  kernel_size,
  stride=None,
  padding=0,
  count_include_pad=True
)
```

평균값 풀링은 팽창을 지원하지 않는다. 팽창의 경우 원소 간의 거리가 멀어지므로 주변 영역의 특징 계산이 어려워진다.

- `count_include_pad`: 패딩 영역의 값을 평균 계산에 포함할지 여부를 설정

평균값 풀링 클래스의 출력 크기 계산 방법은 아래와 같다.

$$
L_{out} = \Bigg[\frac{L_{in} + 2 \times {padding} - kernel\_size}{stride} + 1 \Bigg]
$$

## 완전 연결 계층
---------


**완전 연결 계층(Fully Connected Layer, FC)**은 각 입력 노드가 모든 출력 노드와 연결된 상태를 의미한다. 이를 통해 완전 연결 계층은 입력과 출력 간의 모든 가능한 관계를 학습할 수 있다. 이 계층은 출력 노드의 수를 조절할 수 있으므로 모델의 복잡성과 용량을 조절하는 데 사용된다.

합성곱 신경망에서는 합성곱 계층과 풀링 계층을 거친 결과물인 특징 맵을 입력으로 받는다. 이때 특징 맵은 3차원이다. 이 특징 맵을 평탄화하여 1차원 벡터로 변경하고 완전 연결 계층의 가중치와 내적 계산을 수행해 출력값을 계산한다.

이 출력값은 특성 맵의 공간 정보가 무시되고 모든 입력을 독립적으로 처리해 계산한다.

완전 연결 계층은 이전 계층에서 추출된 특징을 활용하여 최종적인 분류 작업을 수행하며 최종값은 소프트맥스나 시그모이드와 같은 활성화 함수를 적용해 분류 모델로 구성할 수 있다.

<img src="https://github.com/user-attachments/assets/30fbfba7-4eba-4180-8d01-aa9e2456a3b7">

## Convolution 파라미터 개수 구하기
-------------

### 두 계층 간의 파라미터 개수

만약 (H,W,C)가 (40,50,128)인 입력값을 convolution연산하여 (40,50,64)의 출력값을 만들었다면, 그 사이에 들어가있는 파라미터는 몇개나 될까?

- 패딩은 1, 스트라이드는 1, 커널의 크기는 3x3이라고 한다.

<img src="../assets/img/naver-boostcamp/pre-course/cnn-parameter.png">

커널의 채널 크기는 입력값의 채널수와 같으므로, 커널은 (3,3,128)이 될 것이다. 커널의 갯수는 출력값의 채널 수와 같아야하므로 64개가 될 것이다.

**패딩, 스트라이드 등은 파라미터 개수와 무관하다.** 파라미터 개수는 커널 (내부) 값들의 개수이므로, 3x3x128x64 = 73,728개가 된다.

### Fully Connected Layer가 있다면

위에서 Fully Connected Layer가 있다면 파라미터 개수가 너무 많아져 Generalization 이슈가 생길 수 있다고 했는데, 일반적인 Convolution에 비해 왜 파라미터 개수가 많아질까?

<img src="../assets/img/naver-boostcamp/pre-course/alex-net_parameter.png">

위의 그림은 AlexNet의 파라미터 개수를 나타낸다. 잘 보면 Fully Connected Layer인 Dense Layer에 진입하는 순간 파라미터의 개수가 엄청나게 커지는 것을 볼 수 있다. 이는 **전연결시에 모든 입력 노드(값)들을 다음 계층의 노드들과 연결하여 파라미터를 생성하기 때문**이다.

이와 다르게 Covolution 연산에서 사용하는 파라미터, 커널은 모든 채널과 input에 대해 동일하게 동작하는 `공유 파라미터(shared parameter)`이다. 따라서 전연결 계층에 비해 파라미터 개수가 훨씬 줄어들게 되어 결과적으로 Generalization이 잘 되는 특징을 가진다. 

이 때문에 **최근 CNN의 트렌드는 앞단의 Convolution Layer를 깊게 쌓고, 뒤쪽의 Fully Connected Layer를 최대한 줄이는 것**이다. 이를 통해 네트워크의 깊이는 깊어지지만, 파라미터 개수는 오히려 줄어들어, 적은 연산으로도 더 뛰어난 성능을 낼 수 있게 된다.

## 1x1 Convolution
--------

가끔 Convolution 연산을 보다보면, 1x1의 커널로 연산하는 경우가 있다. 1x1로 연산한다면 H와 W의 값을 조정할수도 없을텐데 왜 하는것일까?

<img src="https://miro.medium.com/v2/resize:fit:1400/1*DWleFx9-wWeF3uFagvo4YA.png" width="500" height="300">

`차원축소(Demension Reduction)` 때문이다. 기존의 H,W라는 공간차원은 그대로 유지한 채, **채널만 줄이기 위해 사용**한다. 이를 통해 **깊이(depth)를 늘리는 도중에 파라미터의 개수를 감소**시킬수 있게 된다.

아주 자주 사용하는 테크닉으로, 예시로 bottleneck architecture 등이 있다.

## Convolution 연산의 역전파
-------------

Convolution 연산도 선형변환인 것은 마찬가지이다. 따라서 기존의 역전파 방식과 동일한 방식으로 진행한다.

다만, Convolution 연산은 **커널이 모든 입력 데이터에 공통으로 적용**되므로, **역전파를 계산할 때(Convolution 연산에 미분할 때)에도 Convolution 연산이 나오게 된다.**

$$
\begin{align*}
\frac{\partial}{\partial x}[f*g](x) &= \frac{\partial}{\partial x} \int_{\mathbb{R}^d} f(y)g(x - y) \, dy \\
&= \int_{\mathbb{R}^d} f(y) \frac{\partial g}{\partial x}(x - y) \, dy \\
&= [f * g'](x)
\end{align*}
$$

$\partial x$는 시그널 $g(x - y)$에 적용되어, $g'$에 대한 합성곱연산이 된다.

이를 그림으로 이해해보자.

<img src="../assets/img/naver-boostcamp/pre-course/cnn_forward.png" width="600" height="400">

순전파에서, 입력값 $X_3$는 출력값 $O_1$으로 갈 때에는 커널의 $W_3$, $O_2$로 갈 때에는 $W_2$, $O_3$로 갈 때에는 $W_1$파트를 사용했다. 

<img src="../assets/img/naver-boostcamp/pre-course/conv-back.png">

반대로, **역전파 과정에서 들어온 미분값 $\delta_1$는 순전파 시 적용된 가중치, 즉 커널의 $W_3$에 곱해져서 Gradient로 전달된다.**

이와 같이 역전파 단계에서는, 입력 단계에서 곱해졌던 커널(파트)을 통해 입력값에 Gradient를 전달한다. 그러면 입력값에서 각각의 커널들에는 어떻게 전달될까?

<img src="../assets/img/naver-boostcamp/pre-course/conv-back2.png">

$O_1$에서 $X_3$로 전달된 $\delta_1$값은, 기존에 적용되었던 커널 $W_3$으로 다시 전달되므로 커널 $W_3$은 $\delta_1x_3$으로 업데이트된다. 마찬가지로 $W_2$와 $W_1$도 각각 $\delta_2x_3$, $\delta_3x_3$으로 업데이트된다.

그런데, 각각의 커널들은 $X_3$뿐만 아니라 다른 입력값에 대해서도 동일하게 적용되었으므로, 각 커널로 들어오는 Gradient를 모두 합해줘야한다.

<img src="../assets/img/naver-boostcamp/pre-course/conv-back3.png">

$$
\frac{\partial \mathcal{L}}{\partial w_1} = \delta_1x_1 + \delta_2x_2 + \delta_3x_3
$$

이렇게 각 커널로 들어오는 모든 gradient들을 다 더하면 결국 convolution 연산과 동일한 연산이 이루어진다.

$$
\frac{\partial \mathcal{L}}{\partial w_i} = \sum_{j} \delta_jx_{i+j-1}
$$

## 합성곱 신경망을 이용한 자연어 처리
---------

이미지에서 2차원 합성곱 신경망 학습을 진행할 때, 필터를 수직/수평 방향으로 이동한다. 그러나 텍스트 데이터의 임베딩 값은 입력 순서를 제외하면 입력값의 위치가 의미를 가지지 않는다.

<img src="https://github.com/user-attachments/assets/9642d8fa-745b-4b06-b595-100b7f010635">

즉, 텍스트 데이터는 **1차원 합성곱(1-Dimensional Convolution)**을 적용해야 한다.

텍스트 데이터는 문장을 단어 단위로 분리하여 각 단어를 임베딩하여 나온 1차원 벡터 데이터를 입력값으로 사용한다. 1차원 합성곱은 이 벡터 입력값을 수직 방향으로만 이동하는 필터를 적용하여 해당 문장의 특징을 추출한다.

1차원 필터의 크기는 필터의 높이에만 영향을 미치며, 필터의 너비는 입력 임베딩의 크기가 된다.

<img src="https://github.com/user-attachments/assets/f73b7e1c-fdb2-4a93-8168-979b7f143c59">

- 필터 크기가 3인 합성곱 임베딩을 수행하면 인접한 3개의 토큰에 대한 연산 수행

1차원 합성곱을 이용한 신경망에서는 다양한 크기(높이 변경)의 합성곱 필터를 사용하여 여러 종류의 정보를 추출할 수 있으다. 또한 1차원 합성곱 연산을 하면 1차원 벡터를 얻으며 이 출력값으로 풀링을 적용하면 하나의 스칼라값이 도출된다.

<img src="https://github.com/user-attachments/assets/596722d4-4129-4fff-b8a4-d64ee5e8725f">

크기가 다른 여러 개의 합성곱 필터를 사용하면 여러 개의 스칼라값을 얻을 수 있고 이를 모으면 하나의 특징 벡터로 연결할 수 있다. 이 특징 벡터에 완전 연결 계층을 추가하여 분류 모델을 구성한다.

합성곱 신경망을 이용한 문장 분류는 사전 학습된 임베딩 벡터를 사용하는 것이 일반적이다.
