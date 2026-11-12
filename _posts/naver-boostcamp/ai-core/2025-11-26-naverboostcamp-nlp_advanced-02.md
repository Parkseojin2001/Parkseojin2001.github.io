---
title: "[BoostCamp AI Tech / NLP 이론] Transformer 1"
description: "Transformer의 기본 원리인 Self-Attention와 Transformer의 장점에 대한 내용을 정리한 포스트입니다."

categories: [NAVER BoostCamp AI Tech, AI Core]
tags: [nlp, self-attention, transformer]

permalink: /naver-boostcamp/nlp/07

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-11-26
last_modified_at: 2026-11-11
---

## RNN 기반 모델의 한계
-----

`RNN`에서 다루었던 Sequential Model들은 완벽한 구성성분을 가진 특정 데이터가 아니면 학습하기가 어려웠다.

- 문장을 학습시킨다고 할 때
    - Original Sequence
        - 나는 오늘 학교에 가서 학식을 먹었다.
    - Trimmed Sequence - 문장마다 길이가 다르다.
        - 나는 오늘 학교에 갔다.
    - Ommited Sequence - 문장성분이 누락되어 있을 수 있다.
        - 오늘 학식 먹었다.
    - Permuted Sequence - 성분의 순서가 permute 될 수 있다.
        - 오늘 학교 가서 학식 먹었지, 나는.

또한 RNN은 단어를 하나씩 차례로 처리해야 해서 병렬화가 어렵다. 트랜스포머는 재귀 없이 **N개의 단어를 한 번에** 처리한다(생성할 때만 단어마다 auto-regressive하게 수행한다).

RNN 계열의 모델에서 시퀀스를 인코딩 과정을 살펴보자. 

<img src="../assets/img/naver-boostcamp/ai-core/transformer_need.png">

위의 그림을 통해 알 수 있듯이 원하는 단어에 접근하기 위해서는 여러 Timestep을 거쳐야 하는데 이 과정이 많으면 정보가 손실되거나 변질될 수 있다. 

이러한 문제가 일어나지 않기 위해서는 직접적으로 접근하는 방식을 통해 해결할 수 있는데 이 방법이 바로 Attention 방법이다.

seq2seq 모델에서 입력 문장의 각각의 단어가 Encoder를 거칠 때 마다 hidden state vector가 출력되며 최종 layer에서 인코딩된 hidden state vector가 output으로 나온다. 

이 vector에 Attention을 적용하여 필요한 정보들을 각각의 Decoder에 반영한다. 마지막 Decoder layer에서는 전 Decoder의 output과 Encoder의 마지막 hidden state vector를 concat하여 출력 layer의 input으로 사용한다.

<img src="https://static.wikidocs.net/images/page/31379/transformer4_final_final_final.PNG">

이때 디코더는 마치 기존의 seq2seq 구조처럼 시작 심볼 \<sos \>를 입력으로 받아 종료 심볼 \<eos \>가 나올 때까지 연산을 진행한다.


## Self-attention의 개념
-----

Self-Attention을 설명하기 위해 동물과 해당 동물의 다리 개수가 매핑된 Hash Table $T$가 있다고 가정하자.

|Key(동물)|Value(다리 개수)|
|--------|--------------|
|강아지|4|
|닭|2|
|문어|8|
|오징어|10|
|고양이|4| 

- `Query` : 어떤 동물을 입력
- `Query`와 정확하게 일치하는 Key의 Value를 출력
    - $T(\text{"강아지"}) = 4$
    - $T(\text{"문어"}) = 8$
- 만약 정확히 매칭되지 않는다면 문제가 발생
    - $T(\text{"사자"}) = \text{"Error: Key is not found"}$

다만, 딥러닝 모델에서는 벡터 형태를 입력으로 받기 때문에 **Key와 Value가 모두 특정한 차원으로 이루어진 Vector** 로 이루어져있다.

- Key와 Value의 벡터 차원은 같을 필요가 없다.
- 찾고자하는 Query를 Key 벡터와 매칭하기 때문에 Query는 Key와 차원이 같은 벡터로 구성된다.

**Attention은 Soft matching을 통한 Hash table** 이라고 말할 수 있다.

- `Query`와 정확히 일치하는 Key가 없어도 유사한 Key를 찾아 활용
    - Query, Key, Value가 모두 벡터로 표현
    - Query-Key 간의 Soft한 유사도(softmax 계산)를 가중치로 사용하여 Value의 가중 평균을 계산

<img src="../assets/img/naver-boostcamp/ai-core/self-attention-2.png">

위의 그림은 Self-Attention 과정을 표현한 것으로 이 과정을 거치면 입력 벡터들이 서로 다른 가중 평균을 취한 결과 벡터가 나타난다.  

이때, 입력 벡터를 그대로 사용하면 정보의 표현력이 제한되므로 입력 벡터에 가중치 벡터($W^Q$ , $W^K$ , $W^V$ )를 곱하는 선형변환을 적용한다.

위의 과정을 수식을 표현하면 다음과 같다.

- 입력 : Query $q$ 하나와 여러 Key-Value($k$, $v$) 쌍
    - Query, Key, Value는 모두 벡터 형태
- 출력: Value 벡터의 가중 평균(Weighted average)
    - 각 Value의 가중치는 대응되는 Key와 Query의 Inner product로 계산
    - Query와 Key는 같은 크기의 차원 $d_k$를 가지고, Value는 $d_v$ 차원을 가짐

$$
\text{Attention}(q, K, V) = \sum_{i} \frac{\text{exp}(q \cdot k_i)}{\sum_{j} \text{exp}(q \cdot k_j)} v_i
$$

<img src="../assets/img/naver-boostcamp/ai-core/self-attention-3.png">

만약 Query가 여러 개 있다면 이를 행렬 Q의 형태로 쌓을 수 있다.

$$
\text{Attention}(Q, K, V) = \text{softmax}(QK^T)V
$$

<img src="../assets/img/naver-boostcamp/ai-core/query-batch.png">

### 트랜스포머의 핵심 아이디어

- 기본 가정 : input $x$는 서로 유기적으로 관련(`organically related`)된 여러 요소(`multiple elements`)로 분할될 수 있다.
    - 사회 속의 사람들
    - 문장 속의 단어
    - 비디오의 프레임
- `Self-attention` : 각 element는 해당 `context`(input의 다른 요소)에 참여하여 own representation을 개선하는 방법을 학습한다.
    - ex. 문장 안의 단어들은 다른 단어들과 유기적으로 연결되어 그 단어가 문장 속에서 가지는 의미(대표성)이 정의된다.

위에서 설명한 Attention과 Transformer의 차이점은 Attention에서 사용하지 않는 `Weight paramters`를 사용한다는 점이다.

Attention에서는 Query, Key, Value를 통해 Attention 값을 구했지만 Transformers에선 **$W_Q$, $W_K$, $W_V$를 사용**한다. 

또한 attention value를 **원래 공간에 다시 매핑하기 위해 $W_O$를 추가적으로 사용**한다.

- Input token {$x_1, x_2, \cdots, x_N$} 가 주어지면, 각 token $x_i$ 는 `linear transformation` 에 의해 각각의 Query $Q_i$, Key $K_i$, Value $V_i$ 벡터로 매핑된다.
- Linear weights($W_Q, W_K, W_V$) 는 `learned parameter`로 `모든 input`에서 공유된다.
- $W_Q \ (W_K, W_V)$ 는 일반적으로 **Query (Key, Value) 역할을 할 벡터를 표현하는 방법을 학습**한다.
- `attention value`를 원래 공간에 다시 매핑하는 또 다른 학습 가능한 매개변수 $W_O$ 가 필요하다.

> `linear transformation` : 벡터를 다른 차원의 벡터로 바꾸는 선형 연산으로 transformers에서는 $(\text{입력 벡터}) \times (\text{학습 가능한 행렬})$ 연산을 수행한다.

Transformers에서 Attention을 수행하는 방법은 다음과 같다.

<img src="../assets/img/naver-boostcamp/ai-core/attention_for_transformer.png">

- $i$에 대해 학습할 때 각 token $x_i$ 는 Query가 된다.
    - input sequence : {$x_1, \cdots, x_N$}
- Key와 Value는 $x_1, x_2, \ldots, x_N$ 에 해당한다.
- 이 떄, Query (Key, Value)를 벡터 형태로 바꾸기 위해 각각 $W_Q$ ($W_K$, $W_V$)를 곱한다.
- 곱해서 만들어진 Query ($Q_i$) 와 Key ($K_1, \ldots , K_N$) 를 dot-product 계산을 하여 각각의 유사도를 계산한다.
- 이 각각의 유사도에 Value ($V_1, \dots, V_N$) 를 곱한 후 모두 더한다.
- 이를 원래 embedding space로 매핑하기 위해 $W_O$를 곱하면 최종 결과값 $z_i$가 된다.

결과값 $z_i$의 의미는 $x_i$과 다른 input들과 연관성(유사도)를 반영하여 다시 표현한 값이다.

이러한 과정을 `contextualize`라고 말하며 이 단계를 반복해 여러 개의 Layer로 구성한다.

### 예시: it은 무엇을 가리킬까

"The animal didn't cross the street because it was too tired."라는 문장이 있다고 했을 때, it이 무엇을 의미하는 지 알아내는 것은 문장 해석의 중요한 포인트이다.

<img src="https://blogik.netlify.app/static/a103df16bceed84e7dd0dac59042db48/5a428/self-attention2.png">

이 때 Self Attention을 사용하면 **문장 내에서 it에 대응하는 단어들을 모두 고려**하여 가장 가능성이 높은 'animal'로 학습하게 된다. 따라서 기계가 문장 내부에서 단어의 의미를 비교적 더 잘 이해할 수 있다고 표현할 수 있다.

### Query · Key · Value의 차원

이 때 유의할 사항들이 몇 가지 있다.

- Score 벡터를 구하기 위해 Query 벡터와 Key 벡터를 내적해야 하므로, **Query와 Key 벡터의 차원은 같아야한다.**
- **Value 벡터는 차원이 달라도 된다.** Value 벡터는 attention rate와 가중합을 계산하기만 하면 되기 때문이다.
- 최종적으로 산출되는 (특정 단어의) **인코딩된 벡터는 Value 벡터의 차원과 동일**하다.

- softmax는 행(row) 단위로 적용한다.

### 어텐션은 왜 잘 작동할까

그렇다면 도대체 왜 이러한 `어텐션 메커니즘(Attention Mechanism)`이 잘 작동하는 것일까?

어떤 이미지가 주어졌다고 생각하자. 이 이미지를 MLP나 CNN에 집어넣으면, 입력이 Fix되어 있으므로 출력이 고정된다.

그러나 Transformer는 네트워크가 Fix되어있다고 하더라도, 인코딩하려는 단어와 **그 옆의 단어들에 따라** 인코딩 결과값이 달라진다. 따라서 결과적으로 MLP보다 좀 더 유연한 모델이 된다. 단순히 **입력된 단어에 따라 출력이 고정되는 것이 아니라, 옆에 주어진 다른 입력(단어)들을 모두 고려하므로 출력이 변화할 여지가 있기 때문**이다.

물론, 더 많은 것을 고려하므로 더 많은 연산($n^2$)이 필요하게 된다. 예를 들어 1000개의 단어가 주어지면 한번에 1000$^2$의 입력을 처리해야한다. 반면, RNN은 1000개의 단어가 주어지면 그냥 연산을 1000번 수행하면 된다. **너무 긴 입력이 주어지면 메모리 부족으로 한번에 계산할 수 없다는 한계를 가진다.**

정리하자면,

- 기존의 RNN에 비해 훨씬 더 많은 문맥정보들을 파악할 수 있다.
- 연산자원이 비교적 많이($n^2$) 든다.

## Scaled dot-product attention
--------

위의 방식은 문제점이 있다.

- $d_k$가 커지면 Dot-product $q \cdot k$ 의 분산이 증가
    - Softmax 내 특정 값이 유난히 커질 가능성이 증가
    - Softmax 출력이 한 값에 편중되는 분포를 가짐
    - **Gradient가 매우 작아짐**

해결 방법은 **Query / Key 벡터의 길이**만큼 출력 값의 분산값을 줄이는 것이다.

$$
\text{Attention}(Q, K, V) = \text{softmax} \bigg( \frac{QK^T}{\sqrt{d_k}}\bigg) V
$$

<img src="../assets/img/naver-boostcamp/ai-core/scaled_dot-product_attention.png">


## Multi-head attention
------

Scaled dot-product attention을 거친 결과들을 취합하는 형태로 동작하는 방식이 `Multi-head attention` 이다.

Query와 가장 유사한 Key의 Value를 활용하면 하나의 Attention은 하나의 Softmax에 기반한다.

이러한 계산 방식때문에 주어진 단어가 가장 연관성이 높은 단어에 치중하여 인코딩되므로 상대적으로 연관성이 낮은 단어가 주어진 단어의 인코딩 결과에 반영되지 못하는 문제점이 있다.

- ex. I came home late
    - 주어진 단어 : I
    - 계산 결과 : came(0.7), home(0.15), late(0.05)
    - 하나의 계산 결과만 사용하면 came에만 집중하여 인코딩하므로 나머지 단어인 home과 late에 대한 정보가 인코딩에 반영이 되지 못함 

위의 문제를 해결하기 위해서 `Multi-head attention`으로 여러 Softmax를 활용한다.

- Q, K, V를 $W_i$ 행렬들을 통해 $h$ 개의 저차원 공간으로 매핑
- 각각의 차원에 Attention을 적용
- 이후 Attention 출력들을 Concatenation 한 후 선형 변환

<img src="https://velog.velcdn.com/images/shparksue/post/379ed996-1384-48f0-b3c4-aa8c7aadc92a/image.png">


이 과정을 수학적으로 표현하면 아래와 같이 표현할 수 있다.

$$
\begin{align*}
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^{o} \\
\text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\end{align*}
$$

<img src="../assets/img/naver-boostcamp/ai-core/multi-head-attention1.png">

<img src="../assets/img/naver-boostcamp/ai-core/multi-head-attention2.png">

위의 과정을 요약해서 하나의 그림으로 표현하면 아래와 같다.

<img src="../assets/img/naver-boostcamp/ai-core/multi-head-attention3.png">

### 실제 구현: 차원을 나눠서 계산한다

그러나 **실제 구현은 위와 같은 방식으로 이루어지지 않는다.** 원래 주어진 임베딩 단어의 차원이 100차원이고 10개의 head를 사용한다면, 이를 10개로 나눈다. 즉, 10차원짜리 입력만을 가지고 Query, Key, Value 벡터를 만들게 된다. 이는 구현된 코드를 보고 직접 확인해보자.

텐서 모양으로 보면, 입력이 `[N, S, d]`(문장 수, 토큰 수, 임베딩 차원)일 때 다음과 같다.

**멀티 헤드(Multi-Head)**는 이러한 셀프 어텐션 벡터를 생성하고자 한다면 헤드에 대한 차원 축을 생성해 [N, k, S, d/k] 텐서 형태를 구성한다. 이 텐서는 k개의 셀프 어텐션된 [N, S, d/k] 텐서를 의미한다.

<img src="https://wikidocs.net/images/page/31379/transformer17.PNG">

이렇게 생성된 k개의 셀프 어텐션 벡터는 임베딩 차원 축으로 다시 **병합(concatenation)**되어 [N, S, d]의 형태로 출력된다.

<img src="https://wikidocs.net/images/page/31379/transformer18_final.PNG">
<img src="https://wikidocs.net/images/page/31379/transformer19.PNG">

### Attention 시각화

Attention을 시각화하면 각 단어가 어느 다른 단어에 집중하는지 확인할 수 있다.

<img src="https://velog.velcdn.com/images%2Ftobigs-nlp%2Fpost%2F6e1d8934-3251-4970-b96d-8f7f302c1df7%2Fimage.png">

2개의 attetion을 사용한 경우와 8개의 attention을 사용한 경우를 시각화 한 결과이며, 같은 it이라는 단어에 대해서도 attention score가 조금씩 다르게 scoring이 되는 것을 확인할 수 있습니다.

### Self-Attention 비교

<img src="https://windmissing.github.io/NLP-important-papers/AIAYN/assets/8.png">

- Self-attention은 입력 길이가 비교적 짧을 때 효율적으로 계산됨
- Attention은 병렬 연산이 가능
    - Sequential operation이 $O(1)$
- 모든 단어까지의 Gradient flow 거리가 모두 동일
    - Maximum Path Length가 $O(1)$