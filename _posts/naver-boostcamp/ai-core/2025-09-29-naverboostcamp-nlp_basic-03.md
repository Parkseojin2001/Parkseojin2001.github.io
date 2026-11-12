---
title: "[BoostCamp AI Tech / NLP 이론] RNN과 Language Modeling"
description: "자연어 처리 분야에서 Recurrent Neural Network(RNN)를 활용하는 방법과 Language Modeling에 대한 설명을 정리한 포스트입니다."

categories: [NAVER BoostCamp AI Tech, AI Core]
tags: [nlp, rnn, lm]

permalink: /naver-boostcamp/nlp/03

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-09-30
last_modified_at: 2026-11-11
---

## 시퀀스 데이터와 자기회귀 모델
---------

### 시퀀스 데이터

`시퀀스 데이터`란 **순차적으로 들어오는 데이터**를 일컫는다. 소리, 문자열, 주가 등이 있다.

시퀀스 데이터에서는 정보들이 시간 순서대로 종합되어서 시점별로 나타난다. 따라서 **이벤트의 발생 순서가 데이터의 중요한 요소로서 동작**한다.

기존의 데이터에 비해 시퀀스 데이터에서 유의할 점은, **독립동등분포(independant identically distributed, i.i.d)를 위배하기 쉽다**는 것이다. 맥락과 순서가 중요하기 때문에, 순서를 바꾸거나 과거 정보에 손실이 발생하면, 데이터의 확률 분포가 바뀌거나 미래 예측의 정확도가 떨어지게 된다.

### 시퀀스 데이터를 다루는 방법

이전 시퀀스의 정보를 가지고 앞으로 발생할 데이터의 확률분포를 다루기 위하여, `조건부확률`을 이용할 수 있다.

$$
\begin{align*}
P(X_1, \ldots, X_t) &= P(X_t | X_1, \ldots, X_{t-1})P(X_1, \ldots, X_{t-1}) \\
&= P(X_t | X_1, \ldots, X_{t-1})P(X_{t-1} | X_1, \ldots, X_{t-2})P(X_1, \ldots, X_{t-2}) \\
&= \prod_{s=1}^t P(X_s | X_{s-1}, \ldots, X_1)
\end{align*}
$$

현재 시점인 $t$ 까지의 확률분포를 알기 위해, $t−1$까지의 확률분포를 구한다. $t−1$의 확률분포를 구하기 위해 $t−2$까지의 확률분포를 구한다. 이를 반복하여 최초까지 내려가면, 최초의 시점부터 순차적으로 조건부확률을 곱하는 식으로 일반화할 수 있다.

$$
X_t \sim P(X_t|X_{t-1}, \ldots, X_1)
$$

다만, 위의 조건부확률과는 달리 시퀀스 데이터를 분석할 때 일반적으로는 과거의 **모든 정보(즉, 최초시점부터의 모든 정보)를 사용하지는 않는다.** 최근 어느 시점까지의 정보를 이용하거나, 몇 개의 과거 정보들을 truncation하기도 한다. 필요성에 따라서 어느 시점까지 데이터를 활용할 지 모델링이 달라질 수 있다.

$$
X_{t+1} \sim P(X_{t+1}| \textcolor{red}{X_t, X_{t-1}, \ldots, X_1})
$$

따라서, $X_{t+1}$을 예측하기 위한 **조건부확률에 들어가는 데이터 길이 $X_t, X_{t-1}, \cdots, X_1$는 가변적**이므로, 시퀀스 데이터를 다루기 위해서는 길이가 가변적인 데이터를 다룰 수 있는 모델이 필요하다.

가변적인 길이나 아니라 고정된 길이 $\tau$만큼의 시퀀스만 사용하는 경우 `AR 모델`, 즉 `자기회귀모델(Autoregressive Model)`이라고 부른다. 다만, $\tau$의 길이는 직접 정해주어야하는 하이퍼파라미터가 되므로, $\tau$를 결정할 때에는 어느정도의 설계자의 사전지식이 필요하게 된다.

또, 문제에 따라 먼 과거의 정보들을 고려해야 하기도하고, $\tau$가 바뀌는 경우가 있다. 이런 경우 사용되는 방법이 RNN의 기본 모형인 `잠재 AR 모델(Latent Autoregressive Models)`, 즉 `잠재자기회귀모델`이다.

$$
\begin{align*}
X_t \sim P(X_t|X_{t-1}, \textcolor{red}{\ldots, X_1}) \textcolor{red}{\to H_t} \\
X_{t+1} \sim P(X_{t+1}|X_t, \textcolor{red}{X_{t-1}, \ldots, X_1}) \textcolor{red}{\to H_{t+1}}
\end{align*}
$$

이 경우 $X_t$를 예측하기 위해 직전정보인 $t−1$ 시점의 정보 $X_{t-1}$를 제외한 나머지 $X_{t-2}, \cdots$ 의 정보들을 $H_t$라는 `잠재변수`로 인코딩하여 이용한다.

따라서, 직전 시점의 정보와 잠재변수 $H$ 두 가지만 가지고 예측할 수 있기 때문에 고정된 길이의 데이터를 가지고 모델링 할 수 있게 된다.

$$
H_t = Net_\theta (H_{t-1}, X_{t-1})
$$

이 때 잠재변수 $H_t$를 신경망을 통해 반복 사용하여 시퀀스 데이터의 패턴을 학습하는 모델을 `RNN`이라고 한다.

### Naive sequence model과 Markov model

<img src="https://velog.velcdn.com/images%2Fhanlyang0522%2Fpost%2F51f2f295-967b-4b00-af41-631bc464269d%2Fimage.png">

가장 기본적인 sequence model은 입력이 여러 개 들어왔을 때 다음 번 입력을 예측하는 것이다. 시간이 지날수록 봐야할 과거 정보가 많아져 계산량이 많아진다.
- ex. Language Model


### Autoregressive Model

<img src="https://velog.velcdn.com/images%2Fhanlyang0522%2Fpost%2F32ea3a44-e6db-4222-bdc7-48c50efe5d51%2Fimage.png">

이를 가장 간단히 해결하는 방법은, **고정된 길이($\tau$)의 과거 정보만을 확인**하는 것이다. 정해진 길이의 과거 정보만을 확인하기 때문에 계산이 쉬워진다.

#### Markov model(first-order autoregressive model)

<img src="https://velog.velcdn.com/images%2Fhanlyang0522%2Fpost%2Fba28002b-7a14-481e-a439-eb65853a7715%2Fimage.png">

Autoregressive Mdoel 중 가장 쉬운 방법은 `Markov Model`이다. 이 모델을 극단적으로 간단히 만든 것이 바로 직전 시점 정보만을 고려하는 `AR(1) 모델`이다. 하지만 너무 많은 정보를 버릴 수밖에 없다.

## RNN(Recurrent Neural Network)
---------

`RNN(Recurrent Neural Network)`은 자연어 처리 및 시계열 데이터를 다루는데 특화된 모델로 어떤 **가변적인 길이의 Sequence 데이터를 입력으로 받아서** 어떤 벡터 형태의 output을 매 Time step에서 반복적으로 출력한다. 또한, 이름에서 알 수 있듯이 무언가 반복적, 재귀적으로 호출되는 함수의 형태를 가지고 있다.

<img src="https://velog.velcdn.com/images/beaver_zip/post/0170a235-c8d7-4e93-9d56-9c451f6bd6e9/image.png">

* $W_{hh}$, $W_{xh}$ : $h_t$로 션형 변환하기 위한 가중치 행렬

- Activation function으로 주로 Zero-centered인 Tanh를 사용
- 추가로, Target을 예측하기 위하여 $h_t$ 를 사용해 $y_t$ 출력할 수 있음
    - e.g. $y_t = W_{hy}h_t$

> 모든 Time step에서 같은 함수 $f_{\theta}$ 와 같은 Parameter $\theta$ 적용

### Hidden-State Vector 계산

<img src="https://velog.velcdn.com/images/beaver_zip/post/5d299625-5029-4edd-9dc0-40a19238c3c1/image.png">

### Rolled / Unrolled RNN

<img src="https://velog.velcdn.com/images/beaver_zip/post/15178823-4336-46c7-a78d-0472260a6cf6/image.png">

- `Rolled RNN` : RNN의 기본 입출력 세팅을 재귀적인 형태
    -  RNN의 출력이 $h_t$로 나왔을 때 다음 Time-step에서는 이 $h_t$를 $h_{t-1}$ 자격으로 다시 RNN 모델에 입력되는 방식으로 나타낸다.
- `Unrolled RNN` : 여러 Time-step을 펼쳐두고 RNN의 입출력 구조를 이전 Time-step에서 다음 Time-step으로 넘어가는 형태
    - 각 Time-step에서 나온 hidden state vector가 그 다음 Time-step의 입력으로 들어가며 그 중 맨 처음에 입력되는 $h_0$는 zero vector로 입력으로 주는 것이 일반적이다.


### RNN의 응용

주어진 입력 sequence를 RNN으로 다룰 때 여러 layer에 걸쳐서 RNN을 쌓아 나갈 수도 있다.

<img src="https://velog.velcdn.com/images/beaver_zip/post/3eabf6ce-0f17-4cd4-a79d-c2ead0f4c4fb/image.png">

> 이렇게 모델의 출력을 다시 입력으로 사용해 순차적으로 예측하는 모델을 Auto-regressive 모델이라고 함.

- `Multi-layer RNN`
    - 첫 번째 layer는 입력을 통해 hidden state를 만들고 그 다음 layer부터는 이전에 만들어낸 hidden state vector을 입력으로 받음
    - 입력 시퀀스의 길이를 유지하는 것을 동시에 이 정보들을 계속 축적해나가는 방식으로 조금 더 유의미한 정보가 담긴 벡터들로 변환할 수 있음
    - 장점: 데이터의 다양한 특징을 추출할 수 있고, 층이 깊을수록 더 복잡한 패턴을 학습할 수 있음
    - 단점: 학습 시간이 길어지고, 기울기 소실 가능성이 높아지며, 먼 과거 정보를 잘 기억하지 못함

- `Bidirectional RNN`
    - 왼쪽 &rarr; 오른쪽, 오른쪽 &rarr; 왼쪽 방향으로 각각 정보를 축적한 뒤 concat 
        - 기본적인 RNN은 왼쪽에서 오른쪽 방향으로 정보를 축적함.
    - 오른쪽에서 나타나는 정보를 반영해야 하는 경우에 유리 (e.g. I read books **that he gave**.)

### RNN의 형태

입력뿐만 아니라 출력도 sequence 형태로 받을 수 있다.

입력과 출력의 형태가 단일/시퀀스 인지에 따라 RNN을 분류할 수 있다.

<img src="https://velog.velcdn.com/images/beaver_zip/post/ba923beb-536e-4e31-bfba-aff4bc416e86/image.png">

- `one-to-one` : 평범한 Neural Network (e.g. 자동차 사진 &rarr; "자동차")
- `one-to-many` : Image captioning 등에 사용 (e.g. 자동차 사진 &rarr; "전시장", "의", "자동차")
    - 입력이 없는 경우는 zero-vector를 사용함.
- `many-to-one` : 감정 분류 등에 사용 (e.g. "재미있는", "영화", "였어요" &rarr; "긍정")
- `many-to-many(1)` : 기계 번역 등에 사용 (e.g. "I", "love", "you" &rarr; "당신을", "사랑", "합니다")
    - 입력 시퀀스를 끝까지 읽은 후 출력 시퀀스를 Time-step 마다 하나씩 출력
- `many-to-many(2)` : 지연이 없어야 하는 실시간 처리에 사용
    - 입력 시퀀스를 읽으면서 바로 출력 시퀀스를 바로 출력
    - Frame 단위 Video classification, Language Modeling 등

### RNN의 장단점

**장점**<br>
- 가변적인 길이의 input sequence를 처리할 수 있음.
- 입력이 많아져도 **모델의 크기는 증가하지 않음.**
    - 동일한 weight 파라미터를 사용하기 때문에 크기가 증가 X
- (이론적으로) t 시점에서 수행된 계산은 여러 단계 이전의 **정보를 사용할 수 있음.**
- 모든 단계에서 동일한 가중치가 적용됨

**단점**<br>
- Recurrent computation이 느림
- Sequence output inference는 **병렬화(parallelization)가 어려움**
- Vanilla RNN은 훈련 중 **기울기 소실(vanishing gradient)** 문제를 겪음
- Vanilla RNN은 종종 시퀀스 내 **장거리 의존성(long-range dependence)**를 모델링하는데 실패함
- 실제로는 여러 단계 이전의 정보에 접근하기 어려움
    - 뒤로 갈수록 앞의 정보에 대한 기억이 떨어짐

## Language Modeling
------------

Language Model은 딜레이없이 매 Time-step마다 예측을 수행하는 구조로 다음에 어떤 단어(혹은 토큰)이 올 지 예측한다.

### 문자열 단위 Language Model

- Vocabulary: [h, e, l, o]
- 학습 문자열

<img src='https://velog.velcdn.com/images/beaver_zip/post/acf62073-80f1-4d3d-9d3e-1112eff5bd50/image.png'>

1. 각 문자를 `One-hot encoding`으로 입력

> **각 문자를 One-hot encoding으로 표현하는 이유**<br>
> - 'h': 1, 'e': 2, 'l': 3, ... 으로 표현하면 각 문자에 Continuous한 순서가 생김.
> - 실제론 순서가 존재하지 않으며, 2.5같은 index는 존재하지 않음.
>
> One-hot encoding으로 표현하면 각 문자는 독립성을 유지하며, 일종의 Categorical variable로 생각할 수 있음.

2. 입력 $x_t$으로 Hidden state $h_t$ 계산
- $h_t = tanh(W_{hh}h_{t-1} + W_{xh}x_t)$

3. Hidden state $h_{t}$ 에서 출력 $y_{t}$ 를 계산
- $y_t = W_{hy}h_{t}$


4. Softmax를 통해 **각 문자의 확률** 계산
5. `Cross-entropy loss` 계산 및 학습
- 학습은 softmax에 출력된 확률값들 중 정답 문자에 대한 확률값이 높게 예측되도록 학습이 진행한다.

이렇게 학습을 마친 모델은 추론할 때 한 글자씩 예측값을 생성한다.

> 현재 예측된 출력값을 다음 Time-step의 입력값으로 사용하는 방식을 `auto-regressive` 라고 한다.

### 문자열 단위 Language Model 학습

1. Forward 과정에서 예측값과 Ground-truth 값의 Softmax Loss를 계산한다.
 
- 사전이 시작 토큰 `<sos>`와 마침 토큰 `<eos>` 를 미리 정의하여 문장의 시작과 끝을 알져준다.

<img src="https://velog.velcdn.com/images/beaver_zip/post/26e06572-f13f-4807-9381-56b58e417474/image.png">

2. Backward 과정에서 가중치($W_{xh}$, $W_{hy}$, $W_{hh}$)를 Update함.

- Output과 정답 (Ground-truth) 간의 차이가 발생할 시에 Loss를 이용
- 가중치는 모든 Timestep에서 공유함
- 빨간색 Backward 경로의 가중치들이 Gradient flow를 통해 Update

<img src="https://velog.velcdn.com/images/beaver_zip/post/62906a09-de68-4ed1-a36d-312b5549729f/image.png">

Backpropagation을 위해 전체 Sequence에 대한 Gradient 계산하는 방식을 `BPTT(Backpropagation Through Time)`이라고 한다. 이 방식의 문제점은 계산 비용이 많이 들고 메모리 요구량이 크다는 점이다. 이를

이를 해결하기 위한 방법이 바로 `Truncated Backpropagation Through Time` 이다.

#### Truncated Backpropagation Through Time

Sequence를 Chunk 단위로 나눠 Forward와 Backward를 진행한다.

<img src="https://velog.velcdn.com/images/beaver_zip/post/f07d0785-f749-4269-9de8-ddbb4a4b99b9/image.png">

Backpropagation을 Chunk 안에서만 진행해 계산량을 줄인다.
- 파라미터를 Update를 한 후에 이 값들을 GPU에서 모두 지우고 새로운 다음 Chuck를 GPU에 할당한다.

<img src="https://velog.velcdn.com/images/beaver_zip/post/fdf98f6c-47c9-48a2-bf39-cc9cb11cb93e/image.png">