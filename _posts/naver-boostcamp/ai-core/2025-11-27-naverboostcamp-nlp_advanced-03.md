---
title: "[BoostCamp AI Tech / NLP 이론] Transformer 2"
description: "Transformer의 주요 구성요소와 Masked Self-Attention에 관한 내용을 정리한 포스트입니다."

categories: [NAVER BoostCamp AI Tech, AI Core]
tags: [nlp, self-attention, transformer, positional encoding, masked self-attention]

permalink: /naver-boostcamp/nlp/08

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-11-27
last_modified_at: 2026-11-11
---

Transformer의 구조는 아래 그림과 같이 구성되어 있다.

<img src="../assets/img/naver-boostcamp/ai-core/transformer_seq2seq.png" width="500">

위의 그림에서 보면 Nx 라고 쓰여있다. 이는 Transformer 구조가 N개 쌓여있다는 의미로 RNN으로 구성된 Seq2seq에서 Encoder와 Decoder가 여러 레이어로 구성되어있는 것처럼 생각할 수 있다.

## Transformer Block
----------

먼저 Encoder 부분을 살펴보자

<img src="https://miro.medium.com/v2/resize:fit:876/1*7sjcgd_nyODdLbZSxyxz_g.png" width="400" height="700">

각 Block은 크게 두 개의 layer로 구분할 수 있다.

- `Multi-head attention`
    - Sequence dimension 축으로 Attention 적용
- `Feed Forward(Two-layer perceptron)`
    - Hidden dimension 축으로 MLP 적용
    - ReLU activation을 사용

위에서 구분한 layer에는 추가적으로 하나의 layer(Add & Norm)가 있다.

- `Add & Norm`
    - Residual connection : 입력 벡터가 Multi-Head Attention과 Add & Norm layer과 연결된 부분
    - Layer normalization

    수식으로 표현하면 다음과 같다.

    $$
    \text{LayerNorm}(x + \text{sublayer}(x))
    $$

### Layer Normalization

Layer Normalization에 대해서 좀 더 구체적으로 살펴보자.

- Normalization을 통해 입력을 평균 0, 분산 1로 변환
- Batch Normalization과 달리, 입력 예제 혹은 입력 길이 단위로 수행

$$
\mu^{l} = \frac{1}{H} \sum_{i=1}^{H} a_{i}^{l}, \ \ \ \sigma^{l} = \sqrt{\frac{1}{H} \sum_{i}^{H} (a_{i}^{l} - \mu^{l})^2}, \ \ \ h_{i} = f(\frac{g_i}{\sigma_i}(a_{i} - \mu_{i}) + b_{i})
$$

<img src="https://miro.medium.com/0*rjGwjSS2k7zonUhB.png" width="500" height="300">

입력이 {학교에, 갑니다} 인 예시를 통해 Layer Normalization 계산 과정을 살펴보자.

<img src="../assets/img/naver-boostcamp/ai-core/layer_normalization.png">

1. Multi-Head Attention 거친 각 차원 벡터를 Normalization하여 평균과 분산을 0과 1로 변환
2. 각 차원 별로 학습 가능한 Parameter로 Affine transformation
    - 이 때 $y = \gamma x + \beta$ 식에서 $\gamma$ 와 $\beta 파라미터들은 학습과정 중에 최적화의 대상이다.

> Layer Normalization : 학습과정 중에 어떤 특정 노드에서 발견되는 activation 값들의 분포를 잘 조정
> - Non-linear unit을 통과할 때 가지는 입력값들의 분포를 적절한 범위 내에 있도록 조절함으로써 학습을 용이
> - overfitting을 방지하고 regularization 효과를 줄 수 있음

참고 : [Batch Normalization vs Layer Normalization](https://kwonkai.tistory.com/144)

### Feed Forward 
-----

- two layer로 구성되어 있으며 fully connected network이며 중간에 ReLU를 사용한다.
- sequence 길이 개수 만큼으로 이루어진 각각의 벡터들을 한번에 하나씩 Feed Forward Network를 통과한다.
- output 또한 sequence 길이에 해당하는 개수만큼으로 나오게 되고 입력벡터와 동일한 차원을 가지도록 한다.
- Residual connection을 통해 Feed Forward output과 input을 각 Time step 별로 더해주는 과정을 수행한다.
- Layer Normalization을 수행한다.

## Positional Encoding
-----

Positional Encoding을 설명하기에 앞서 이를 사용하는 주된 이유는 다음과 같다.

Transformer 블록을 사용했을 때 어떤 동일한 시퀀스 내에서 입력 단어의 순서가 바뀐다하더라도 각각의 해당 단어에 인코딩된 hidden state 벡터는 동일하게 나오는 문제가 발생한다.

이를 `Permutation-invariant` 라고 한다. 즉, 입력 순서가 뒤바뀌어도 똑같은 출력이 나온다.

따라서, Transformer 모델이 같은 단어들로 이루어진 입력 Sequence에 대해서도, 서로 다른 어순을 구별하도록 하기 위해, 각 단어에 현재 위치 정보를 넣을 필요가 있다.

<img src="https://builtin.com/sites/www.builtin.com/files/styles/ckeditor_optimize/public/inline-images/Transformer-neural-network-13.png" height="500" width="300">

- 원 Transformer 논문에선 `Sinusoidal function`을 활용
- 학습 가능한 Positional embedding을 사용하기도 함

### Sinusoidal Positional Encoding

- 서로 다른 주파수를 가진 Sinusoidal function을 활용

$$
\begin{align*}
PE_{(pos, 2i)} = sin(pos / 10000^{2i/{d_\text{model}}}) \\
PE_{(pos, 2i + 1)} = cos(pos / 10000^{2i/{d_\text{model}}})
\end{align*}
$$

<img src="https://nlp.seas.harvard.edu/images/the-annotated-transformer_49_0.png">

ex. 각 time-step 또는 각 토큰 별로 주어지는 입력 벡터가 100차원이라고 생각해보자.

1. 50개의 서로 다른 주파수를 가지는 sin 및 cos 함수를 사용해서 해당 함수에 입력값을 대입하여 총 100개의 함수값을 이용하여 100차원 벡터를 구성한다.

2. 첫번째 time-step에서 나타난 토큰의 입력 벡터에 100개의 함수에 1을 대입하여 얻은 총 100개의 서로 다른 함수값들을 하나의 100차원 벡터로 구성하여 더해주는 방식으로 해당 토큰을 변형해준다. 

3. 이 과정을 매 time-step 또는 토큰에 적용하여 입력값들을 변형시킨다.

### PyTorch로 구현하기

```python
# 위치 인코딩
import math
import torch
from torch import nn
from matplotlib import pyplot as plt

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

encoding = PositionalEncoding(d_model=128, max_len=50)

plt.pcolormesh(encoding.pe.numpy().squeeze(), cmap="RdBu")
plt.xlabel("Embedding Dimension")
plt.xlim((0, 128))
plt.ylabel("Position")
plt.colorbar()
plt.show()
```

출력 결과는 아래와 같다.

<img src="../assets/img/deep-learning/nlp/postional_embedding.png" height="320" width="350">

- `PositionalEncoding` 클래스는 입력 **임베딩 차원(`d_model`)**과 **최대 시퀀스(`max_len`)**를 입력받는다. 입력 시퀀스의 위치마다 $sin$과 $cos$ 함수로 위치 인코딩을 계산한다.
- `pe` 변수의 텐서 차원은 [50, 1, 128]이 되며, [최대 시퀀스, 1, 입력 임베딩 차원]을 의미하며 산출된 위치 인코딩은 입력 임베딩과 더해진 후 드롭아웃이 적용된다.
- `register_buffer` 메서드는 모델이 매개변수를 갱신하지 않도록 설정한다.

## 특수 토큰과 입력 임베딩
-----

트랜스포머는 단어 토큰 이외의 특수 토큰을 활용하여 문장을 표현한다. 이 특수 토큰은 입력 시퀀스의 시작과 끝을 나타내거나 **마스킹(Masking)** 영역으로 사용된다.

특수 토큰으로 모델이 입력 시퀀스의 시작과 끝을 인삭할 수 있게 하며, 마스킹을 통해 일부 입력을 무시할 수 있다.

<img src="../assets/img/deep-learning/nlp/token.png" width = "700" height="350">

**BOS(Beginning of Sentence)**, **EOS(End of Sentence)**, **UNK(Unknown) 및 PAD(Padding)** 토큰은 모두 특수 토큰으로 자연어 처리에서 일반적으로 사용된다.

- BOS 토큰 & EOS 토큰: 문장의 시작(BOS)과 문장의 종료(EOS) 나타내며 이는 문장의 시작과 끝을 명확히 하기 위해 사용된다.
- UNK 토큰: 어휘 사전에 없는 단어로, 모르는 단어를 의미하며 이전에 본 적이 없는 단어를 처리할 때 사용된다.
- PAD 토큰: 모든 문장을 일정한 길이로 맞추기 위한 토큰이며 주로 짧은 문장의 빈 공간을 채울 때 사용된다.

이렇게 생성된 문장 토큰 배열을 어휘 사전에 등장하는 위치에 원-핫 인코딩으로 표현한다.

<img src="../assets/img/deep-learning/nlp/input_embedding.png">

입력 임베딩으로 변환되는 과정은 Word2Vec 방법과 동일하다.
- 어휘 사전의 크기를 V, 입력 임베딩 차원을 d라고 했을 때, '트랜스포머'의 원-핫 벡터는 [1,V] 크기를 갖는다.
- 원-핫 벡터는 임베딩 행렬 [V, d]에 의해 [1, d] 크기의 벡터로 변환된다.

이를 일반화하면, N개의 문장이 최대 S개의 토큰 길이를 가질 때 [N, S, V] 크기의 원-핫 벡터 텐스는 [N, S, d] 크기의 임베딩 텐서로 변환된다.

이 임베딩 텐서는 입력으로 사용되며, 트랜스포머의 모든 계층에서 공유되는 텐서로 사용된다.

## Decoder
----

Transformer의 Decoder는 Encoder와 비슷하게 Multi-Head Attention 모듈을 중심으로 하는 어떤 특정 구조로 이루어진 하나의 블럭을 단위로 해서 이 블럭이 총 N개 쌓여있는 형태로 이루어져있다.

이 떄, Decoder의 입력에서 아직 추론이 시작히지 않은 즉, 아직 예측을 하나도 하지 않는 단계에서 입력으로 `<SOS> (start of sentence)` 를 넣어준다. 

<img src="https://imghub.insilicogen.com/media/photos/encoder_decoder.png" width="500" height="300">

Decoder가 예측한 단어는 현재 입력과 함께 다음 스텝의 입력으로 사용한다.

- 입력: \<SOS \> &rarr; 출력: 나는
- 입력: \<SOS \> 나는 &rarr; 출력: 학교에

이렇게 Decoder는 다음에 나타나는 단어를 순차적으로 예측하는 `Auto-Regressive` 한 방식으로 추론한다.

이러한 방식으로 학습을 진행하면 문제가 발생한다. 학습 초기에는 다음에 올 단어의 추론 정확도가 굉장히 떨어져 잘못된 단어를 예측할 수 있다. 하지만, 잘못 예측한 단어를 입력으로 사용하면 이 학습은 무의미한 과정이다. 

따라서, 학습 과정에서 나타나는 문제점을 해결하기 위해서는 실제 값과 동떨어진 값을 예측했다고 하더라도 그 다음 Time-step의 입력으로 주는 값은 정답을 입력으로 줌으로써 다음 단어를 예측할 때 올바른 정보를 참조할 수 있도록 한다.

이를 구현하기 위해 Decoder의 입력으로 전체 문장에서 오르쪽으로 shift한 문장(마지막 단어가 잘린 형태)을 사용한다.

하지만, 이를 사용하면 발생하는 문제점이 있다. 이 방식을 사용하면 현재 디코더가 추론을 아직 하지 않는 단어에 접근할 수 있기 때문이다.

- ex. \<SOS \> 다음 단어를 예측하지만 "\<SOS \> 나는 학교에" 가 입력으로 주어지기 때문에 다음 예측 단어에 접근 가능

이러한 문제를 해결하기 위해서 Decoder에서는 `Masked Multi-Head Attention` 기법을 사용한다.

### Masked Attention

`Masked Attention`은 `Causal attention`이라고도 부르며 Self-attention이 과거 생성 정보만을 참조한다.

- 아직 생성되지 않은 단어는 추론 과정 중엔 접근할 수 없음
- Softmax 출력을 수정하여 아직 생성되지 않은 단어를 참조하지 못하게 함

<img src="../assets/img/naver-boostcamp/ai-core/masked_attention_1.png">

위의 그림에서 초록색 행렬에서 가중치가 저장되어있다. 이때, 빨간색 글씨가 가르키는 행렬값들은 \<SOS \>의 다음 단어를 예측하는데 사용되는 입력 벡터에 대한 정보를 담고있다.

이러한 값을 참조하지 못하게 하기 위해서 해당 행렬값을 $-infty$로 대체함으로써 이 값에 softamx를 씌운 값이 0이 되도록 한다. 이렇게 Mask를 씌우는 방식을 통해 다음 단어 예측에 사용되는 정보들에 접근하지 못하도록 만든다.

<img src="../assets/img/naver-boostcamp/ai-core/masked_attention_2.png">

위의 방식으로 변형된 attention weight들을 가지고 해당 행렬과 Value Matrix를 곱한다.

### Cross Attention

`Cross Attention`은 Encoder-decoder 간의 attention 하는 방법이다.

<img src="../assets/img/naver-boostcamp/ai-core/cross-attention.png" height="300" width="500">

- Query: Decoder의 Causal-attention 결과
- Key & Value: Encoder의 최종 layer에서 계산한 vector

Self-Attention 과 Cross-Attention 은 Input 을 제외하면 차이가 나지 않는다.(동일한 계산 방법)

하지만 Input, 목적에서 차이가 있으며 `Self-Attention` 은 번역작업 기준 Source 언어의 토큰별 관계를 파악하기 위해 진행한다면, `Cross-Attention` 은 **Target 언어와 Source 언어의 Token 간 관계를 파악**하기 위해 사용된다.

아래 그림은 디코더가 인코더의 Key·Value를 참조하며 한 단어씩 생성하는 과정이다.

<img src="http://jalammar.github.io/images/t/transformer_decoding_1.gif">

Input에 있는 값들을 디코더에 있는, 즉 출력하고자 하는 단어들에 매핑하는 `Attention Map`을 만드려면, Key 벡터와 Value 벡터가 필요하다. 마지막 인코더 층(가장 상위 인코더)의 출력값은 여러 단어들의 [Key 벡터, Value 벡터] 시리즈로 제공되고, 이 값들을 각 Decoder 내부의 `Encoder-Decoder Attention` 계층에서 해석하여 입력 시퀀스의 적절한 위치를 잡아주게 된다.

- Decoder에서 `Encoder-Decoder Attention Layer`는 MHA와 동일한 방식으로 동작하지만, Query 벡터를 이전 Decoder에서 받아오고 Key, Value 벡터를 인코더 스택에서 받아온다는 차이가 있다.

<img src="http://jalammar.github.io/images/t/transformer_decoding_2.gif">

모든 출력이 끝나면, `EOS(End Of Sentence) 토큰`이 나와 종료를 알린다.

<img src="https://jalammar.github.io/images/t/transformer_decoder_output_softmax.png">

디코더의 마지막 층에서는, **디코더 스택의 출력물을 단어들의 분포로 만들어 내보낸다.** 이 단어들을 매번 샘플링하는 형식으로 모델의 출력이 만들어진다.

## Token Aggregation
-------------

분류(classification) 또는 회귀(regression)와 같은 task를 수행하는 방법으로는 여러 방법이 있다.

### Average Pooling

수행하려는 task가 분류(classification) 또는 회귀(regression)와 같은 작업을 할 때 가장 간단한 작업은 바로 **평균을 구하는 것**이다.

- transformers에서 나온 revised token embeddings $\{ z_1^{(L)}, \ldots, z_N^{(L)}\}$ 전체에 대해서 평균을 취하면, 전체 Input에 대한 single embedding $z$를 얻게 된다.
- 위에서 얻은 값을 가지고 classifier 나 regressor 모델을 훈련시킬 수 있다.

만약, 시퀀스가 매우 길지 않고 relatively homogeneous한 경우 평균을 구하는 것이 효과적이지 않은 경우는 `MLP`를 사용할 수도 있다.

### Classification Token

또다른 방법으로는 `attention mechanism` 에 의존하는 것이다.

- 집계된 embedding은 시퀀스 내에서 **정보를 제공하는 부분만 선택적으로 attend하도록 훈련**된다.
- 그러나 $z_i$는 token $i$와의 관련성을 기반으로 계산되므로, 각 $z_i$는 전체 sequence가 아닌 특정 $x_i$를 나타낸다.

전체 시퀀스의 특징을 나타내기 위해 고안한 방법이  바로 `더미 토큰 classification token; [CLS]` 를 이용하는 것이다.

<img src="https://www.mccormickml.com/assets/BERT/CLS_token_500x606.png">

- `CLS` 를 입력 시퀀스에 추가하여 이를 embedding으로 사용한다.
- 더미 입력은 어떤 의미도 전달하지 않기 때문에 특정한 토큰에만 치우치지 않는다.
- `CLS`에서 나온 출력을 classifier/regressor의 입력으로 사용한다.

실제로 `CLS token`을 사용하는 방법을 훨씬 많이 사용한다.

마지막 레이어의 출력(또는 `[CLS]` 출력)에 classifier나 regressor를 붙이고, 여기서 정답과 비교해 loss를 구한다.

## 트랜스포머 계열 모델과 활용
-----

트랜스포머 기반 모델들은 **오토 인코딩(Auto-Encoding)**, **자기 회귀(Auto-Regressive)** 방식 또는 두 개의 조합으로 학습된다.

- 오토인코딩 방식은 랜덤하게 문장의 일부를 빈칸 토큰으로 만들고 해당 빈칸에 어떤 단어가 적절할지 예측하는 작업(Task)을 수행한다. 예측되는 토큰의 양옆에 있는 토큰들을 참조하기 때문에 양방향 구조를 가지며 이를 **인코더(Encoder)**라고 한다.

<img src="../assets/img/deep-learning/nlp/encoder.png">

- 단방향 구조는 예측되는 토큰의 왼쪽에 있는 토큰들만 참조하며 이를 **디코더(Decoder)**라고 한다.

트랜스포머 모델들은 양방향성 구조의 인코더(오토 인코딩) 또는 단방향성 디코더(자기 회귀)를 사용하는 공통적 특징을 가지고 있다.

|모델|학습구조|학습방법|학습 방향성|
|----|-----|-----|---------|
|BERT|인코더|오토 인코딩|양방향|
|GPT|디코더|자기 회귀|단방향|
|BART|인코더+디코더|오토 인코딩+자기 회귀|양방향+단방향|
|ELECTRA|인코더+판별기|오토 인코딩+대체 토큰 탐지|양방향|
|T5|인코더+디코더|오토 인코딩+자기 회귀+다양한 자연어 처리 작업을 학습|양방향|

### Vision Transformer

Transformer 모델에서 인코더를 차용해 이미지 분류 시에 사용한다. 이 때 Encoder만 활용한다. Encoder에서 나온 출력값(Encoding vector)으로 이미지를 분류한다.

<img src="../assets/img/naver-boostcamp/shared/vit_architecture.png">

논문: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929)

### DALL-E

문장이 주어지면 이에 맞는 이미지를 만들어낸다. Transformer의 decoder만 이용하여 만들었다.(GPT-3를 활용했다고 한다)

참고: [DALL·E: Creating Images from Text](https://openai.com/ko-KR/index/dall-e/)
