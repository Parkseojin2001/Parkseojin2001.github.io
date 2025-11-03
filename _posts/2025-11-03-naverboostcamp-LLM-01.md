---
title: "Generative AI Introduction"
description: "생성형 AI 중 생성형 언어/이미지 모델의 전반적인 개요에 대한 내용을 정리한 포스트입니다."

categories: [Naver-Boostcamp, Generative AI]
tags: [Generative AI, LLM]

permalink: /naver-boostcamp/generative-ai/01

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-11-03
last_modified_at: 2025-11-03
---

과거의 인공지능은 주로 이해를 목표로 한 AI였다. 예를 들어 알파고는 바둑의 수를 이해하고 예측하는 데 초점이 맞춰져 있었다. 그러나 오늘날의 인공지능은 단순한 이해를 넘어, **새로운 것을 창조해내는 생성형 AI(Generative AI)**로 발전하고 있다.

## Generative AI: NLP
------------

`Large Language Model(LLM)`은 텍스트를 입력으로 받아 적잘한 출력을 산출하는 언어모델이다.

- 대량의 텍스트 데이터로 사전학습
- Billion Scale의 파라미터 보유

현재 LLM의 발전은 InstructGPT / ChatGPT 출현 이후 활발히 연구와 일상에 적용되고 있다.

- 학습 방법론 관련 연구 : Corpus 정제, Instruction Tuning 등
- 활용 연구 : 의료/법률 도메인 적용, 서비스 적용 디자인
- 최적화 연구 : 추론 속도/메모리 사용량 최적화, 입력 문장 길이 확장 등

LLM의 장점은 다음과 같다.

- 별도의 Finetune 없이 다양한 태스크 수행 가능
- 다양한 분야에 적용되어 광범위한 활용 가능성 보유
- 사용자 입력에 대해 적절한 태스크 수행 능력 보유
- 텍스트 입력을 통해 태스크 및 출력문 제어 가능

하지만 LLM은 높은 비용이 요구되기 때문에 아래와 같이 특정 상황에서 사용된다.

- 인간 행도 모사가 필요한 경우(ChatBot)
- 태스크가 매우 어려운 경우
- 데이터가 매우 제한적인 경우
- 사실 정보를 기반으로 생성해야 하는 경우

## Generative AI: Vision
---------

생성형 이미지 모델 $p_{model}$ 은 특정 데이터의 분포 $p_{data}$ 를 기반으로 새로운 이미지를 생성하는  모델이다. 이 모델의 학습 목표는 특정 데이터를 생성할 확률인 likelihood를 최대화하는 것이다.

대표적인 생성형 이미지 모델의 유형은 다음과 같다.

- GAN
- VAE
- Flow-based models
- Diffusion models

<img src="https://www.researchgate.net/publication/378336281/figure/fig30/AS:11431281242375755@1715478873773/A-comparison-between-VAE-GAN-flow-and-diffusion-generative-models-Ho-et-al-63.png">

### GAN

판별자(Discriminator)와 생성자(Generator)를 적대적으로 학습하는 모델 구조이다.

- 판별자: 입력 이미지가 생성된 이미지인지 진짜 이미지인지 판별
- 생성자: 잠재 변수 $z$ 를 입력으로 받아 학습 데이터의 분포에 가까운 이미지를 생성

### Autoencoder

Autoencoder(AE)는 Encoder와 Decoder로 구성되어 입력 이미지를 다시 복원하도록 학습하는 모델 구조이다.

- Encoder: 입력 이미지를 저차원 잠재 공간(Latent Space)으로 매핑하여 잠재 변수 $z$ 로 변환
- Decoder: 잠재 변수를 입력으로 사용하여 원본 이미지를 복원

Autoencoder의 변종으로는 VAE(Variationl AE)와 VQ-VAE(Vector Quantized-VAE) 가 있다.

Autoencoder는 잠재 변수의 분포를 정의하지 않지만 VAE는 잠재 변수의 분포를 정의하며 VQ-VAE는 잠재 변수의 분포를 이산화하여 정의한다.

### Flow-based models

입력 이미지를 함수 $f$를 통해 잠재 공간으로 변환하고 역함스 $f^{-1}$ 를 통해 이미지를 복원하는 구조이다.

- 함수 $f$ : 연속적, 미분 가능, 역변환이 가능한 함수

Flow가 Encoder 역할을 Inverse가 Decoder 역할을 수행한다.

이 모델은 변수 변환(change of variable)을 기반으로 구성되어 있는 구조이다.

### Diffusion models

입력 이미지를 forward process를 통해 잠재 공간으로 변환하고 reverse process로 복원하는 구조이다.

- Forward process: 점진적으로 가우시안 노이즈를 추가하여 잠재공가능로 매핑하는 과정
- Reverse process: forward process에서 추가된 노이즈를 추정하여 제거하는 과정

### 생성형 이미지 모델 활용 분야

- Stype transfer
    - 이미지의 스타일을 다른 이미지에 적용하는 방법
- Inpainting
    - 이미지의 손상된 부분이나 누락된 부분을 복원하거나 채우는 방법
- Image editing
    - 이미지를 변경하거나 개선하느 방법
- Super-resolution
    - 저해상도 이미지를 고해상도 이미지로 변환하는 방법

### Multi-modal 생성형 이미지 모델

- Text-to-Image
    - 텍스트를 입력으로 사용하여 이미지를 생성
- Text-to-Video
    - 텍스트를 입력으로 사용하여 비디오 생성
- Image-to-Video
    - 이미지와 prompt를 사용하여 비디오 생성

## 생성 모델 활용 사례
---------

- ChatGPT and GPT API
- DALL-E3 and Stable Diffusion