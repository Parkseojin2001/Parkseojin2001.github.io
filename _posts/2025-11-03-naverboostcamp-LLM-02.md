---
title: "Text Generation-LLM Pretrained Models"
description: "생성형 언어 모델이 다양한 태스크에서 놀라운 성능을 발휘할 수 있는 학습 방법론 대한 내용을 정리한 포스트입니다."

categories: [Naver-Boostcamp, Generative AI]
tags: [Generative AI, LLM, Text Generation]

permalink: /naver-boostcamp/generative-ai/02

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-11-03
last_modified_at: 2025-11-03
---

## Large Language Model
----------

Large Language Model 이란?

- 범용적인 태스크 수행이 가능한 Language Model
- 사전학습 데이터 및 파라미터 수가 매우 큰 모델의 종합적 지칭
- 사전학습 데이터 : 온라인 상 수집가능한 최대한의 텍스트 데이터(LLaMA 학습 데이터 : 4TB)
- 파라미터 수 : 하드웨어 상 학습 가능한 최대한의 파라미터(LLaMA 파라미터 수 : 7B ~ 65B)


Language Model은 두 가지 종류로 나눌 수 있다.

- Pretrained LM: GPT-1/2, BERT 등
    - Downstream 태스크 별 Finetune을 통해 목적 별 모델 구축
    - 하나의 모델을 이용하여 하나의 태스크 해결
- LLM: GPT-3/4, ChatGPT, LLaMA, Mistral 등
    - 사전학습 및 Finetune을 통해 범용 목적 모델 구축
    - 하나의 모델을 이용하여 다양한 태스크 해결

LLM의 동작은 `zero/Few-Shot Learning` 으로 설명할 수 있다.

`zero/Few-Shot Learning`은 LLM의 범용 목적 모델 동작 원리고 모델 추가 학습없이 입력 데이터 구성을 통해 다양한 태스크를 수행한다.

- Zero Shot: 모델이 Prompt만으로 태스크를 이해하여 수행
- Few Shot: 모델이 Prompt와 Demonstration(예시)를 통해 태스크를 이해하여 수행

같은 태스크를 수행할 경우, 대체적으로 Few Shot의 성능이 높다. 또한, 모델 능력이 충분한 경우에 Demonstration을 통해 성능 향상이 가능하다.

이떄, Prompt 구성 요소는 아래와 같다.

- Task Description: 수행할 태스크에 대한 묘사
- Demonstration: 수행할 태스크 예시(입력-출력 쌍)
- Input: 실제 태스크를 수행할 입력 데이터

Prompt의 구성 방식에 따라 모델 성능이 변화하기 때문에 어떤 방식으로 작성할 지가 굉장히 중요하다.


### Model Architecture

LLM의 아키텍처는 두가지의 Transformer 변형 구조가 사용된다.

- Encoder - Decoder 구조
    - 입력 이해와 문장 생성 모델 분리
    - 입력 이해: Encoder를 통해 처리
    - 문장 생성: Decoder를 통해 처리
- Decoder Only 구조
    - 단일 모델을 통해 이해 및 생성
    - ex. GPT


### Pretrain Task

Pretrain Task는 모델 구조 별 사전 학습 태스크가 상이하다는 특징을 가지고 있다.

Encoder - Decoder 구조인 경우 아래와 같은 특징이 있다.

- Span Corruption
- T5에서 제안된 Pretrain Task
- 손상된 입력 문장의 일부를 복원하도록 생성

> **Span Corruption**
>
> 1. 입력 문장 중 임의의 Span을 Masking
> 2. Masking 시 각 Masking Id 부여
> 3. Span Corruption된 문장을 Encoder 입력
> 4. Masking Id와 복원 문장을 Decoder 입력
> 5. Decoder는 복원된 문장 생성
>
> 이를 통해 입력 문장 이해 및 문장 생성 능력을 학습한다.

Decoder Only 구조인 경우 아래와 같은 특징을 갖는다.

- Language Modeling
- GPT-1에서 제안된 Pretrain Task
- 입력된 토큰을 기반으로 다음 토큰 예측 수행

> **Language Modeling**
>
> 1. 문장 토큰 단위로 입력
> 2. 매 토큰마다 다음 토큰을 예측하도록 학습
>
> 이전 입력을 바탕으로 다음 토큰 생성 능력 학습

최근 대부분의 LLM에서는 Causal Decoder 구조를 사용하며 내부 구조는 Transformer와 비슷하다. 또한 Pretrain 방식으로 대부분 Next Token Prediction 수행을 하며 이는 구현 방식 및 연산이 효율적이라는 장점이 있다.

### Pretrain Corpus

모델 크기 경향은 2020년 GPT-3 이후 모델 크기가 점차 확장되고 있으며 이러한 대형 모델을 훈련 시키기 위한 사전학습 코퍼스 구축이 중요해지고 있다.

코퍼스는 사전학습을 위한 대량의 텍스트 데이터 집합으로 구축 절차는 다음과 같다.

- 원시 데이터: 온라인 상에서 수집된 최대한 많은 데이터
    - 블로그, 뉴스, 서적, 댓글 등
- 원시 데이터 내 학습 불필요 데이터 존재
    - 욕설 및 혐오 표현, 중복 데이터, 개인정보가 포함된 데이터
    - 대량의 데이터 정제 작업이 필요( ~ 5TB)

이러한 방식으로 학습을 진행하게되면 `Memorization in LLM` 문제가 발생한다.

- Memorization: LLM이 코퍼스 내 존재하는 데이터를 암기하는 현상
- 코퍼스 내 중복하여 등장한 데이터는 쉽게 암기
- 모델 크기가 클수록 암기 능력 향상
- 데이터 정제 수행 X
    - 혐오 표현, 개인 정보 등 부적절한 답변 도출 가능
    - 정제하지 않으면 학습 자원을 소모하고 모델 학습에 도움 X
    - 개인 정보같은 경우는 서비스 배포 시 개인정보 유출 가능

## Instruction Tuning
---------

LLM은 대형 코퍼스로 학습되었기 때문에 굉장한 능력을 보유한다.

- Zero/Few-Shot Learning: 입력된 프롬프트 내 정보만으로 태스크 학습 및 수행
- 다양한 문장 생성 능력 보유
    - 존재하지 않는 어휘에 대해서도 문장 생성이 가능

이러한 문제점을 보완할 필요가 있다.

### Helpfulness & Safety

위에서 제시된 문제와 온라인에 존재하는 혐오/차별/위험 표현이 문제가 된다. 이는 LLM 학습에 반영되어 새로운 혐오/차별/위험 표현을 생성할 수 있다.

- Safety: LLM이 생성한 표현이 사회 통념상 혐오/위험/차별적 표현이 아니어야 함
    - 혐오/차별: 특정 종교, 성별, 정치 성향에 따른 일반화
    - 위험: 신체적, 정신적 위험을 초래할 수 있는 생성문
    - 특정 질병에 관해 잘못된 조언 및 진단을 생성하면 안됨
- Safety를 위해 특정 입력에 대해 답변을 거부하거나 우회할 수 있어야 함

- Helpfulness: LLM이 사용자의 다양한 입력에 적절한 답변을 생성해야함
    - 기존의 Pretrained LM은 입력/출력이 훈련된 태스크로 구성되어있어서 Helpfulness에 대한 이슈가 발상하지는 않음
    - LLM은 사용자가 원하는 광범위한 입력에 적절하게 출력을 생성해야 하기 때문에 Helpfulness문제가 발생

이러한 이슈들은 `Instruction Tuning`을 통해서 해결할 수 있다.

- 사전학습: 이전 단어를 바탕으로 단순히 다음 단어를 예측하도록 학습
- Instruction Tuning
    - Instruction: 사용자의 광범위한 입력에 대해
    - Safety: 안전하면서
    - Helpfulness: 도움이 되는
    - 적절한 답변을 하도록 Fine-Tune하는 과정

이러한 Instruction Tuning은 크게 3단계로 구성된다.

1. SFT(Supervised Fine Tuning)
2. Reward Mdoeling
3. RLHF(Reinforcement Learning with Human Feedback)

### Supervised Fine-Tuning(SFT)

`Supervised Fine-Tuning(SFT)`은 굉범위한 사용자 입력에 대해 정해진 문장을 생성하도록 FineTune하는 것을 말한다.

- 학습 데이터
    - Prompt: 사용자의 매우 다양한 요청으로 도메인, 입력 형태 등이 매우 자유롭다.
    - Demonstrations: 해당 요청에 대한 적절한 답변으로 Safety와 Helpfulness를 만족한다.
- 학습 방법: LLM에게 사용자 입력에 적절히 답변하도록 지도학습

### Reward Modeling

LLM이 SFT를 통해서 만들어낸 답변을 가지고 사람의 선호도를 모델링하는 것이다.

- LLM의 생성문이 Helpfulness와 Safety를 만족도를 점수로 산출

학습 데이터는 아래와 같이 구성된다.

- Prompt: 사용자의 매우 다양한 요청
- Demonstrations: LLM이 생성한 답변 후보
- Rating: 사람이 판단한 Prompt에 대한 Demonstration의 적절성

학습 방법은 Prompt와 Demonstration을 입력으로 Rating을 산출하도록 학습한다.
- Ranking 기반 학습 방법론 사용
- Helpfulness와 Safety에 만족하는 답변에 더 높은 점수를 부여하도록 학습

### Reinforcement Learning by Human Feedback

RLHF는 LLM이 사람의 선호도가 높은 답변을 생성하도록 학습한다.
- 사람의 선호도: Reward Model이 높은 점수를 부여하는 답변

이때 답변 생성은 SFT 모델을 이용하여 답변을 생성하고 이 답변을 Reward Model이 Rating을 부여한다. 이 부여된 Rating을 높이는 방향으로 SFT 모델을 학습한다.
- 활용되는 알고리즘은 PPO 알고리즘이다.

이 방식을 통해 Safety와 Helpfulness를 위반하는 답변을 내지 못하도록 유도하는 쪽으로 학습이 진행된다.

RLHF 방식 학습을 위한 데이터를 구축 시 Safety/Helpfulness 반영으로 막대한 비용이 소모되며 LLM 학습에서는 막대한 GPU 비용이 소모된다.

하지만, SFT와 RLHF를 적용하면 얻는 효과도 있다. 

- 일반적인 사전학습 LLM보다 다양한 지표에서 개선
- 사용자 지시 호응도는 상승하고 거짓 정보(Hallucination) 생성 빈도는 감소
- RLHF 적용 시 작은 모델(1.3B)에서 큰 모델(175B)보다 높은 Instruction Following 능력 기록
- LLM: 모델의 크기가 중요하나, Instruction Tuning 방법론 중요



