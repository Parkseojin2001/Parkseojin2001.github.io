---
title: "[BoostCamp AI Tech / NLP 이론] Exploding and Vanishing Gradient of Recurrent Neural Network"
description: "RNN의 문제점과 Exploding / Vanishing Gradient 문제가 발생하는 원인을 정리한 포스트입니다."

categories: [NAVER BoostCamp AI Tech, AI Core]
tags: [nlp, rnn]

permalink: /naver-boostcamp/nlp/04

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-10-01
last_modified_at: 2026-11-11
---

Language model이나 Question answering 태스크로 해당 문제를 풀 때

- 각 단어는 하나의 Timestep에서 주어지는 Token에 대응되고
- 단어 하나씩 One-hot vector로 표현되어 모델에 입력
- 다음 단어를 맞추기 위하여 이전에 **입력/생성된 단어 정보를 저장하고 있어야 함**

만약 정답과 관련된 단어가 빈칸에서 너무 멀다면?

## RNN의 문제점
---------

위의 상황에서 RNN을 통해 Task를 수행하면 아래와 같은 문제가 발생한다.

- 정답과 관련된 정보 외의 많은 정보가 여러 Time-step을 거치면서 고정된 크기의 Hidden state에 저장
    - 고정된 크기 = 정보 저장 공간
- **Forward propagation 과정 중에 $W_{hh}$ 를 계속 곱함**
    - 반복적인 계산으로 인해 해당 정보가 변질될 수 있음



## RNN의 학습 과정
-----------

RNN 아키텍쳐는 아래처럼 표현할 수 있다.

<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/0*pfDZQ6GIiZ0MGHjG.jpeg">

- $V$, $W$, $U$ : 가중치 행렬
- $b$ : 편향 벡터

학습 과정은 다음과 같다. 

1. 입력이 도착하면 은닉층 레이어를 거쳐 레이블을 예측한다.
2. 실제 레이블과 추정된 레이블의 차이를 평가하기 위해 손실함수를 계산한다.
3. 총 손실 함수 $L$이 계산되고, 이를 통해 순방향 패스가 완류된다.
4. 역방향 패스로 다양한 미분을 계산한다.

<img src="https://miro.medium.com/v2/resize:fit:720/format:webp/0*SKflG0ykj6EWPnm4.jpeg">

기울기를 `layer`와 `Time-step`에 걸쳐 역전파하는 과정이 필요하여 이에 따라 각 Time-step 마다 다음 식과 같이 현재 시간 단위까지 이전 기여도를 모두 합산해야 한다.

$$
\frac{\partial{\mathbf{L}}}{\partial{W}} = \sum_{i=1}^{T}\frac{\partial \mathcal{L}_i}{\partial W} \propto \sum_{i=0}^T \bigg( \prod_{i=k+1}^y {\color{red}{\frac{\partial h_i}{\partial h_{i-1}}}} \bigg) \frac{\partial h_k}{\partial W}
$$

이 식에서는 **시간 단계 $t=T$ 에서 전체 손실 함수 $\mathbf{L}$** 의 기울기에 대한 시간 단계 $k$ 에서의 상태의 기여도를 계산하며 이를 통해 학습 과정에서의 문제는 은닉 상태의 비율에서 발생할 수 있다.

좀 더 심플한 RNN 형태의 예시를 통해 알아보자.

### BPTT(Backpropagation Through Time)

RNN의 역전파는 **잠재변수의 연결그래프에 따라 순차적으로 계산**한다. 이를 `Backpropagation Through Time(BPTT)`이라고 한다.

<img src="../assets/img/naver-boostcamp/pre-course/rnn-back.png">

RNN의 모든 시점에서의 예측이 전부 이루어진 다음, 마지막 시점의 Gradient가 타고 올라와서 과거까지 흐른다. **RNN은 모든 시간 스텝에서 파라미터 $W$를 공유하므로, 한 시점에서 오류가 역전파되면 이것이 모든 이전시점으로 시간을 거슬러 퍼지게 된다.** 그래서 BPTT라는 이름이 붙었다.

위의 그림에서 빨간 선이 역전파과정인데, $H_t$의 역전파에 들어오는 Gradient는 ​$H_{t+1}$과 $O_t$에서 각각 하나씩 총 두개가 있다.

#### BPTT의 Product Term

BPTT를 통해 RNN의 가중치행렬 미분을 계산해보면, 아래와 같은 미분의 곱으로 이루어진 항이 계산된다.

<img src="../assets/img/naver-boostcamp/pre-course/bptt.png">

최종적으로 나오게 되는 Product Term(빨간색 박스가 나타내는 항)은 $i+1$부터 $t$ 시점까지의 모든 잠재변수에 대한 미분값이 곱해지는데, 이 때 **시퀀스의 길이(즉, 시점간의 거리)가 길어질수록 해당 항이 불안정해지기 쉬워진다.**

- 매 시점마다 1보다 큰 수가 곱해진다면 미분값이 아주 커진다.
- 매 시점마다 1보다 작은 수가 곱해진다면 미분값이 아주 작아진다.

가장 주의해야 하는 것은 작은 수의 미분값이 곱해져 **Gradient가 0으로 소실되는 현상(`Vanishing Gradient`)**이다. 이 경우 과거시점으로 갈 수록 Gradient가 점점 작아지기 때문에 과거 시점의 정보가 제대로 반영이 되지 않아 유실된다. 이러면 긴 시퀀스를 분석해야하거나 문맥적인 부분이 중요한 모델의 경우 좋은 결과를 얻을 수 없다.

### RNN 학습 예시

먼저 **Forward propagation** 과정은 다음과 같다.

<img src="../assets/img/naver-boostcamp/ai-core/simple_rnn_0.png">

위의 그림에서 input 벡터($x$)와 hidden state 벡터($h$) 그리고 output 모두 스칼라 값으로 가정하자.

- $h_1 = tanh(ah_0 + bx_1 + c)$
- $h_2 = tanh(ah_1 + bx_2 + c)$
- $h_3 = tanh(ah_2 + bx_3 + c)$
- $y = dh_3 + e$

<img src="../assets/img/naver-boostcamp/ai-core/simple_rnn_1.png">

그리고 Ground truth 값과 예측값을 비교하여 loss 계산을 진행한다. 계산한 loss 값을 이용하여 **Backward propagation** 과정을 진행한다.


이때 최종 output을 $f(=y_3)$ 로 표현하자.

$x_1$ 입력으로부터 $y_3$를 잘 예측할 수 있는 정보가 $h_1$ 에 잘 담기도록 해야하며 이를 위해 $x_1$ 에서 제공된 정보를 $h_1$의 값이 잘 담을 수 있도록 하는 학습에 필요한 gradient signal은 $\frac{df}{dh_1}$ 으로 표현할 수 있다.

$\frac{df}{dh_1}$  을 구하기 위해서는 `backpropagation` 이 필요하다.

$$
\begin{align*}
\frac{df}{dh_1} = \frac{df}{dh_1} = \frac{dh_2}{dh_1} \times \frac{dh_3}{dh_2} \times \frac{df}{dh_3}
\end{align*}
$$

$\frac{df}{dh_3}$의 값은 $d$ 이며 $\frac{dh_{i}}{dh_{i-1}}$ 는 $\frac{1}{2} a$ 이므로 이를 적용하면 아래와 같다.

$$
\frac{df}{dh_1} \le \frac{1}{2} a \times \frac{1}{2} a \times d
$$

이때, $a$ 값에 따라 두 가지의 결과가 나올 수 있다.

- $\| a \| > 1 \Rightarrow \lim_{n \to \infty} a_n = \infty$ &rarr; Exploding Gradient
- $\| a \| < 1 \Rightarrow \lim_{n \to \infty} a_n = 0$ &rarr; Vanishing Gradient

> $\underset{x}{\text{max}} \bigg(\frac{d}{dx} tanh(x) \bigg) = \frac{1}{2}$ 이다.

이렇게 Time-step 간의 Gap이 커지면 커질수록 첫번째 Time-step에 도달하는 gradient가 굉장히 크거나 작은 값이 될 것이며 이는 학습이 불안정해지거나 학습이 잘 안될 수 있는 원인이 된다.

## Exploding / Vanishing Gradient

$$
\mathbf{h}_t = \text{tanh}(W_{hh} \mathbf{h}_{t-1} + W_{xh}\mathbf{x}_t + \mathbf{b})
$$

위 수식은 첫 Hidden-state vector에 $W_{hh}$ 를 반복적으로 곱하는 것과 유사하다.

$$
\mathbf{h}_t \propto W_{hh}^{t-1} \mathbf{h}_1
$$

이 때, $W_{hh}$ 가 `Eigendecomposition`을 통해 $VDV^{-1}$ 로 분해될 수 있다고 가정하자.
- $V$는 정사각 행렬로 $W_{hh}^T$ 의 eigenvector들을 column으로 가진다.
- $D$는 diagonal matrix(대각 행렬)로써 eigenvalue들을 diagonal entry의 값으로 가진다.

$$
W_{hh}^{t-1} \mathbf{h}_1 = VD^{t-1}V^{-1} \mathbf{h}_1
$$

행렬 $D$는 Eigenvalue의 대각 행렬이므로, Eigenvalue의 지수승을 곱한다.

$$
\begin{bmatrix}
2 & 0 \\
0 & 0.3
\end{bmatrix}^{t-1}
\begin{bmatrix}
a \\
b
\end{bmatrix} = 
\begin{bmatrix}
2^{t-1} & 0 \\
0 & 0.3^{t-1}
\end{bmatrix}^{t-1}
\begin{bmatrix}
a \\
b
\end{bmatrix}
$$

따라서 가중치 행렬 $W_{hh}$ 를 `Eigendecomposition` 했을 때, 
- Eigenvalue의 절대값이 1보다 크다 &rarr; Exploding Gradient
- Eigenvalue의 절대값이 1보다 작다 &rarr; Vanishing Gradient

### 한 셀 단위로 보기: tanh'와 $W_{hh}$

RNN cell을 더 자세히 살펴보자.

<img src="https://wikidocs.net/images/page/227990/20240120_222842.png">

수식은 $h_t = tanh(W_{hh}h_{t-1} + W_{xh}x_t)$ 이다.

$\mathbf{h} _{t}$ 에서 $\mathbf{h} _{t-1}$ 로 backprop를 진행하기 위해 편미분을 하면 아래와 같다.

$$
\frac{\partial{\mathbf{h}_t}}{\partial{\mathbf{h}_{t-1}}} = tanh'(\mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{W}_{xh}\mathbf{x}_{t})\mathbf{W}_{hh}
$$

위의 과정을 $\mathscr{L}$ 에서 $\mathbf{W}_{hh}$ 까지 backprop를 진행하면 아래와 같은 결과가 나온다.

<img src="../assets/img/naver-boostcamp/ai-core/vanilla_rnn_prop.png">


- $\prod_{k=2} ^t \ tanh'(\mathbf{W} _{hh}\mathbf{h} _{k-1} + \mathbf{W} _{xh}\mathbf{x} _{k})$ 
    - `$tanh'` 가 거의 항상 1 미만이면 `기울기 손실(Vanishing gradients)`

- $\mathbf{W}_{hh}^{t-1}$
    - 1보다 큰 값이면 `기울기 폭발(exploding gradients)`
        - `클리핑(clip)`을 통해 thresholde값 이상이면 gradient를 조절하여 해결하는 경우도 있다.
    - 1보다 작은 값이면 `기울기 손실(vanishing gradients)`
        - `클리핑`으로 폭발을 해결해도 기울기 손실은 해결하지 못함

만약, `Xavier Initialization`을 통해 적절한 가중치로 초기화를 한다면?

- 어느 정도 deep model에서는 비교적 잘 작동하지만, 더 깊은 모델을 만들수록 0으로 수렴하는 결과를 얻게된다.

이 때, RNN 모델의 깊이를 결정하는 것은 `input sequence length` 이다. 입력 시퀀스가 길면 깊어가 깊어지고 짧으면 깊이 또한 얕다.

그러므로, 시퀀스 데이터의 길이가 길면 gradient의 값이 모델의 초기 레이어까지 잘 전달되지 않아 RNN 모델은 학습이 진행되지 않는다.

### 활성화 함수에 따라 달라지는 문제

<img src="../assets/img/naver-boostcamp/pre-course/recurrent_neural_network.png">

$$
\begin{align*}
h_1 &= \textcolor{red}{\phi(W^T h_0} + U^T x_1) \\
h_2 &= \textcolor{red}{\phi(W^T \phi(W^T h_0} + U^Tx_1) + U^Tx_2) \\
h_3 &= \textcolor{red}{\phi(W^T\phi(W^T\phi(W^T\phi(W^Th_0} + U^Tx_1) + U^Tx_2) + U^Tx_1)+ U^Tx_3) \\
\ldots
\end{align*}
$$

RNN은 이런식으로 과거의 $h$ 들을 고려하는 중첩된 구조이다. $\phi$는 활성화함수(activation function)이다. 시퀀스 길이가 늘어남에 따라, 이처럼 **중첩되는 가중치와 activation function이 굉장히 많아진다.**

위의 식에서 만약 활성화함수가 `sigmoid`라고 하자. sigmoid의 성질은 값을 계속 0과 1사이로 바꿈으로써 축소시키는 것이므로, 함수가 중첩될수록 점점 `vanishing gradient`의 문제가 생긴다.

만약, sigmoid가 아닌 ReLU함수라면 어떨까? ReLU함수는 $x>0$ 일때 해당 input을 그대로 가져가므로, $W$와 input의 곱이 계속 쌓이는 구조가 될 것이다. 따라서 자칫하면 Gradient가 아주 커져 네트워크가 터지는 `exploding gradient`의 문제가 생긴다.
