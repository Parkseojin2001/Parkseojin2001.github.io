---
title: "Exploding and Vanishing Gradient of Recurrent Neural Network"
description: "RNN의 문제점과 Exploding / Vanishing Gradient 문제가 발생하는 원인을 정리한 포스트입니다."

categories: [Deep Learning, NLP]
tags: [NLP, RNN]

permalink: /naver-boostcamp/nlp/04

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-10-01
last_modified_at: 2025-10-01
---

Language model이나 Question answering 태스크로 해당 문제를 풀 때

- 각 단어는 하나의 Timestep에서 주어지는 Token에 대응되고
- 단어 하나씩 One-hot vector로 표현되어 모델에 입력
- 다음 단어를 맞추기 위하여 이전에 **입력/생성된 단어 정보를 저장하고 있어야 함**

만약 정답과 관련된 단어가 빈칸에서 너무 멀다면?

위의 상황에서 RNN을 통해 Task를 수행하면 아래와 같은 문제가 발생한다.

- 정답과 관련된 정보 외의 많은 정보가 여러 Time-step을 거치면서 고정된 크기의 Hidden state에 저장
    - 고정된 크기 = 정보 저장 공간
- **Forward propagation 과정 중에 $W_{hh}$ 를 계속 곱함**
    - 반복적인 계산으로 인해 해당 정보가 변질될 수 있음

    