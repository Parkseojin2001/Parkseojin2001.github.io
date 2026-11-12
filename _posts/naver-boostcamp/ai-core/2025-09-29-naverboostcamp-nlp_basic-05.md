---
title: "[BoostCamp AI Tech / NLP 이론] LSTM과 GRU"
description: "Vanishing Gradient 문제를 완화한 LSTM과 GRU의 구조 및 동작 과정을 정리한 포스트입니다."

categories: [NAVER BoostCamp AI Tech, AI Core]
tags: [nlp, rnn, lstm, gru]

permalink: /naver-boostcamp/nlp/05

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-10-02
last_modified_at: 2026-11-11
---


## LSTM(Long Short-Term Memory)
--------

`LSTM(Long Short-Term Memory)` 모델은 RNN과 같이 매 Time step 마다 반복적으로 동일 모듈을 적용한다. 모델 안에는 4개의 Vector를 계산하는 신경망이 존재한다.

<img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png">

LSTM과 Vanilla RNN과의 차이점은 **hidden state를 계산하는 방식**이다. 

RNN은 전 단계의 hidden state와 현재 input을 통해 현재 hidden state를 구했다. 반면에, LSTM은 hidden state를 계산하는 방법이 훨씬 복잡해졌다. 

LSTM에서는 RNN에 없던 `cell state`가 있는데 전 Time step의 cell state 정보로부터 일부의 정보만을 담도록 변환된 벡터를 해당 Time step의 hidden state가 된다. 즉, cell state가 RNN의 hidden state의 역할을 한다고 말할 수 있다.

cell state의 업데이트 과정이 동일한 $W_{hh}$ 행렬을 반복적으로 곱하는 것이 아니라 좀 더 복잡한 계산을 거쳐서 업데이트를 한다.

이러한 방법은 `Gradient Vanishing`과 `Exploding` 문제를 방지한다.

### Highway network

LSTM에서는 ResNet 모델에서 사용한 기법인 `skip connection`을 변형한 `Highway network` 기법을 사용한다.

<img src="../assets/img/naver-boostcamp/ai-core/highway_network.png">


- 첫번째 구조 : skip connection은 입력값과 함수를 거친 출력값을 그대로 더하는 방식이다.
- 두번쨰 구조 : 이를 발전시켜 입력과 함수의 출력값의 비율($\alpha$)를 조정하는 방식으로 계산할 수 있다.
    - 여기서 $\alpha$는 고정된 값으로 적용되는 상수라고 볼 수 있다.
- 세번째 구조 : 입력과 함수의 출력값에 곱해지는 가중치 또한 동적으로 만든 방식이다.
    - $g(x)$ : 가중치를 계산하는 function
    - $\alpha(x)$ 는 $g(x)$를 입력으로 한 sigmoid 함수의 출력값으로 사용하여 0과 1 사이의 값으로 범위를 변경해주는 과정을 거친다.

입력, 출력, 가중치 모두 동일한 차원이다.

LSTM의 구조는 세번째 구조에서 더 발전된 구조를 갖는다.

<img src="../assets/img/naver-boostcamp/ai-core/highway_lstm.png">

첫번째 구조에서는 입력 $x$와 함수의 출력값 $f(x)$ 모두 $\alpha(x)$ 라는 가중치에 영향을 받는다. 하지만 두번째 구조는 입력과 출력이 동적인 서로 다른 가중치에 영향을 받는다.

최종 결과값은 동적인 서로 다른 가중치의 영향을 받는 값들의 합으로 결정된다.

### LSTM의 Notation

<img src="../assets/img/naver-boostcamp/ai-core/lstm-gates.png">

현재 Time step의 입력 벡터 $x$와 이전 Time step의 hidden state를 concat 한 벡터 형태로 입력을 받은 후 $W$ 와 행렬곱하여 선형 변환을 한다. 이 결과값은 hidden state의 차원과 동일한 차원을 갖는다.

이 벡터들 중 3개의 벡터는 sigmoid를 나머지 하나는 tanh 연산을 통해 4개의 gate 만든다.

- $f$ : Forget gate, Cell의 내용을 지울지 결정
- $i$ : Input gate, Cell의 내용을 적을지 결정
- $g$ : Function output, Cell에 적을 내용
- $o$ : Output gate, Cell의 내용을 출력할지 결정

위의 gate를 이용해서 현재 Time step의 cell state와 hidden state를 계산한다. 이때, 벡터의 곱($\odot$)은 element-wise multiplication을 수행한다.

### LSTM의 구조

LSTM를 좀 더 자세히 살펴보자.

LSTM의 구조를 보면 위에서 설명한 `Highway network` 구조를 갖는다. 


<img src="../assets/img/naver-boostcamp/ai-core/lstm_highway.png">

- 이전 Time step에서 넘어온 $c_{t-1}$는 forget gate와 곱해지는데 이 gate는 sigmoid를 지나기 때문에 원소들이 0~1 사이의 값을 가진다.
- $c_t$를 계산하는 식에서 두번쨰 항의 $i(x_t, h_{t-1})$는 input gate로 이 gate는 sigmoid 함수를 통과한 벡터이므로 각각의 원소들이 0~1 사이의 값을 가진다.

즉, `input gate`가 현재 Time step에서 계산한 정보의 출력을 조절하고, `forget gate`가 $c_{t-1}$ 을 조절하는 방식이다.

LSTM에서 핵심 아이디어는 Cell state 정보를 어떠한 왜곡 없이 전달할 수 있는 능력을 가지기 때문에 장기 기억 문제(Long-term dependency)를 해결할 수 있다.

LSTM의 **cell state는 덧셈 형태로 이전 상태가 전달**되기 때문에, gradient가 곱셈으로 계속 축소되지 않고 그대로 전달되어 장기 의존성을 학습할 수 있게 된다.

$$
\frac{\partial L}{\partial C_{t-k}} = \frac{\partial L}{\partial C_t} \odot \prod_{i=t-k+1}^t f_i
$$

역전파 과정을 수식으로 표현하면 위와 같이 표현할 수 있으며 여기서 $f_i$는 0.8~1.0 사이를 유지하기 때문에 RNN의 장기 의존성 문제를 부분적으로 해결이 가능하다.

<img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-o.png">

위에서 계산한 Cell state로부터 Hidden state를 생성한다. 이렇게 생성된 Hidden staet를 그대로 사용하지 않고 `Output gate`를 통해 조절한다.

이러한 과정을 거친 Hidden state를 다음 Time step으로 전달, 그리고 출력/다음 층으로 전달한다.

### 게이트별로 자세히 보기

LSTM의 cell state의 연산 과정은 다음과 같다.

<img src="https://wikidocs.net/images/page/152773/3.JPG">

메모리 셀은 순환 신경망의 은닉 상태와 유사하게 현재 시점의 입력과 이전 시점의 은닉 상태를 기반으로 정보를 계산하고 저장하는 역할을 한다. 

하지만 순환 신경망에서 은닉 상태는 출력값을 계산하는 데 사용되지만, LSTM의 메모리 셀은 출력값 계산에 직접 사용하지 않는다.

대신 forget gate, input gate, output gate를 통해 어떤 정보를 버릴지, 어떤 정보를 기억할지, 어떤 정보를 출력할지를 결정하는 추가적인 연산을 수행한다.

세 가지 게이트 모두 시그모이드를 활성화 함수로 사용하므로 게이트의 출력값은 각각 0에서 1 사이의 값으로 결정된다. 이 값이 해당 게이트가 입력값에 대해 얼마나 많은 정보를 통과시킬지 결정한다.

**Forget gate**

<img src="https://wikidocs.net/images/page/152773/4.JPG">

- $h_{t-1}$: 이전 시점의 은닉상태
- $x_t$: 현재 시점의 입력값
- $f_t$: 시그모이드 활성화 함수를 사용해 계산
- $W_f$: 입력값과 은닉 상태를 위한 가중치
- $b_f$: forget gate를 위한 편향

forget gate는 두 가중치를 통해 forget gate 출력값을 최적화하며 이 출력값은 메모리 셀을 계산하기 위한 가중치로 사용된다.

- forget gate = 1: 이전 시점의 기억 상태가 현재 시점의 기억 상태에 완전히 유지
- forget gate = 0: 이전 시점의 기억 상태는 현재 시점의 기억 상태에 전혀 반영되지 않음

**Input gate**

<img src="https://wikidocs.net/images/page/152773/5.JPG">

- $\tilde{C_t}$: -1에서 1 사이의 값을 가지므로 이전 시점의 은닉 상태와 현재 시점의 입력값은 모두 [-1, 1] 범위 안에 존재한다. 하지만 이것만으로는 새로운 은닉 상태를 계산하기 위해 이 은닉 상태를 얼마나 기억할지 제어하기가 어렵다.

- $i_t$: 새로운 은닉 상태의 기억을 제어한다. [0, 1]의 값을 가지므로 현재 시점에서 얼마나 많은 정보를 기억할 것인지를 결정하는 가중치 역할을 한다.
  - 1에 가까울수록 기억할 정보가 많아지고, 0에 가까울수록 정보를 망각하게 된다.

현재 시점의 입력과 이전 시점의 은닉 상태를 입력으로 받아 시그모이드 함수를 거친 값과 하이퍼볼릭 탄젠트 함수를 거친 값의 곱으로 새로운 기억 값을 계산한다.

**Update**

<img src="https://wikidocs.net/images/page/152773/6.JPG">

- $f_t$: forget gate의 출력값
- $c_{t-1}$: 이전 시점의 메모리 셀 값

forget gate와 input gate 값을 각각 원소별 곱셈 연선을 의미하는 **아다마르 곱**한 후 합산한다. 
- forget gate는 이전 시점의 메모리 셀을 얼마나 유지할지 결정
- input gate는 현재 시점의 새로운 정보를 얼마나 받아들일지를 결정

**Output gate**

<img src="https://wikidocs.net/images/page/152773/7.JPG">


- $o_t$: 현재 시점의 출력값
  - 출력값의 범위는 [0, 1]이며 은닉 상태의 값을 얼마나 사용할지 결정한다.
- $h_t$: 현재 시점의 은닉 상태
  - output gate와 하이퍼볼릭 탄젠트가 적용된 메모리 셀 값으로 계산되며 범위는 [-1, 1]이다.

정리하면 다음과 같다.

- LSTM은 Vanilla RNN 보다 **long-range information을 더 잘 보존하다.**
    - forget gate=1이고 input gate=0 이면, 이전 정보는 계속 유지하고 새로운 정보에 대한 반영은 전혀 하지 않게 작동하므로 계속 이전의 값을 유지한다.
- LSTM은 기울기 소실/폭발이 없다는 것을 보장하지는 않지만, 모델이 long-distance dependencies(장거리 의존성)을 학습하기에 더 쉬운 방법이다.

#### Cell staet 계산 예시

<img src="../assets/img/naver-boostcamp/ai-core/cell_staet_lstm.png">

참고: [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)


## GRU(Gated Recurrent Unit)
--------

`GRU(Gated Recurrent Unit)`는 LSTM을 경량화한 모델로 cell state 벡터와 hidden state 벡터를 하나의 벡터로 통합하여 LSTM의 cell state 벡터의 역할을 한다.

<img src="../assets/img/naver-boostcamp/ai-core/gru_2.png">

GRU는 `Gated Recurrent Unit`의 약자로, **게이트가 3개 있던 LSTM과는 달리 2개의 게이트(reset, update)만을 가진다. 또, cell state가 없고, hidden state만 가진다.**

<img src="https://blogik.netlify.app/static/f2716bc289734d8b545926b38a224692/2bef9/gru.png">

`Reset Gate`가 기존의 `Forget Gate`역할을 하고, `Input&Output Gate`가 합쳐져 `Update Gate` 역할을 한다고 볼 수 있다.

파라미터 개수가 LSTM보다 적음에도 불구하고 비슷한 작용을 하므로, **대체로 일반화 성능이 좋은 편**이다.

그러나 최근에는 LSTM과 GRU 모두 Transformer가 나오면서 대체되고 있는 추세다.

LSTM과 유사한 또 다른 아이디어로 RNN에서 장거리 의존성(long-range dependency)을 제공한다.

- LSTM처럼 **추가적인 cell states가 필요하지 않음**
- LSTM보다 **적은 파라미터를 사용함**
- LSTM과 유사한 **graident highway**를 제공하며, 이전 hidden state와 input에서 계산된 새로운 **convex combination을 사용함**

<img src="../assets/img/naver-boostcamp/ai-core/gru.png">



> convex combination : 한 벡트를 다른 벡터의 `가중 평균(weighted average)`으로 표현한 것으로 이 때, 가중치는 0~1 범위이고 합이 1이어야 한다.<br>
>
> $$c = \alpha a + (1 - \alpha)b \ , \ \ \ \text{단} \ \ 0 \le \alpha \le 1 $$


위의 내용을 정리하면 다음과 같다.

- `LSTM`은 RNN에 대한 좋은 기본 선택이다.
    - RNN 이라고 말하는 경우, 대부분이 LSTM을 가르키는 경우가 많다.
- 더 빠른 게산과 적은 파라미터를 원한다면 GRU를 사용할 수 있다.
- 무거운 자연어 처리(NLP) 문제에 대해서는 `Transformer`가 쓰인다.
