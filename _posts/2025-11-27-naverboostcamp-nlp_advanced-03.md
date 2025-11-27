---
title: "Transformer 2"
description: "Transformer의 주요 구성요소와 Masked Self-Attention에 관한 내용을 정리한 포스트입니다."

categories: [Naver-Boostcamp, NLP 이론]
tags: [NLP, Self-Attention, Transformer, Positional Encoding, Masked Self-Attention]

permalink: /naver-boostcamp/nlp/08

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-11-27
last_modified_at: 2025-11-27
---

Transformer의 구조는 아래 그림과 같이 구성되어 있다.

<img src="https://images.velog.io/images/tobigs-nlp/post/92305540-bc49-4873-bec2-dec27741d2b5/transformer.PNG">

위의 그림에서 보면 Nx 라고 쓰여있다. 이는 Transformer 구조가 N개 쌓여있다는 의미로 RNN으로 구성된 Seq2seq에서 Encoder와 Decoder가 여러 레이어로 구성되어있는 것처럼 생각할 수 있다.

## Transformer Block
----------

먼저 Encoder 부분을 살펴보자

<img src="https://miro.medium.com/v2/resize:fit:876/1*7sjcgd_nyODdLbZSxyxz_g.png">

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

<img src="https://miro.medium.com/0*rjGwjSS2k7zonUhB.png">


