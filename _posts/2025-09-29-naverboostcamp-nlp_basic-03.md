---
title: "RNN과 Language Modeling"
description: "자연어 처리 분야에서 Recurrent Neural Network(RNN)를 활용하는 방법과 Language Modeling에 대한 설명을 정리한 포스트입니다."

categories: [Deep Learning, NLP]
tags: [NLP, RNN, LM]

permalink: /naver-boostcamp/nlp/03

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-09-30
last_modified_at: 2025-09-30
---

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

## Language Modeling
------------