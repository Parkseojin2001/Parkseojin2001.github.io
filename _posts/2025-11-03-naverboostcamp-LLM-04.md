---
title: "Text Generation-sLLM Models"
description: "sLLM이 무엇인지 기존의 LLM들과 차이점과 sLLM 활용 방법에 대한 내용을 정리한 포스트입니다."

categories: [Naver-Boostcamp, Generative AI]
tags: [Generative AI, LLM, Text Generation, sLLM]

permalink: /naver-boostcamp/generative-ai/04

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-11-04
last_modified_at: 2025-11-04
---

## Open-Source LLM
-------
Open-Source License는 소프트웨어의 코드 공개를 통해 복제/배포/수정/수정에 대한 명시적 권한을 말한다.

- 소스코드가 공개되어 사용자 및 회사에서 자유롭게 수정 및 배포가 가능
- 다른 소프트웨어를 활용하여 개발을 진행할 경우 해당 소프트웨어 라이센스 고려 필요
- 라이센스를 위반하는 경우 법적/윤리적/경제적 이슈 발생
    - MIT License : 자유로운 복사 및 배포 가능, 유료화 불가능
    - CC-BY-SA 4.0 : 자유로운 복사, 배포 및 유료화 가능
- 개발 목적 및 향후 활용 방안에 따라 활용 소프트웨어 라이센스 검토 필요

머신러닝/딥러닝 분야의 특성을 고려해서 라이센스를 고려해야한다.

- 학습 데이터 + 학습/추론 코드 + 사전학습 모델
    - 각 요소 별 라이센스 및 저작권 검토 필요성 증대
- 학습 데이터 및 코드로 인한 라이센스 변경 가능성 존재
    - 모델의 상용화 불가능

특히 학습 데이터 저작권 문제가 주목받고 있다. 저작권이 존재하는 데이터로 학습된 모델은 상업적 활용이 불가능하다.

Open-Source LLM으로 대표되는 모델은 바로 LLaMA이다. LLaMA의 특징은 다음과 같다.

- 연구 목적 활용이 가능한 Open-Source LLM
    - 학습 코드 및 모델 공개
    - 모델 크기 : 7B ~ 70B
- 모델 공개를 위한 사전학습 데이터 구축
    - 공개된 사전학습 데이터 이용
    - Meta 내부 데이터 활용 X
- 기존 Open/Closed LLM 대비 높은 성능 달성


LLaMA가 좋은 성능을 보이는 이유는 작은 모델을 더 오래 학습시키는 것이 모델 배포에서는 유리하기 때문에 `Chinchilla Scaling Law` 를 위반하여 학습했기 때문이다.

- Chinchilla Scaling Law : 동일 자원에서 모델 성능을 가장 높이는 학습 데이터 수와 모델 크기 관계식로 정해진 사전학습 예산 존재 시 모델 크기와 학습 데이터는 반비례 관계
- 모델 크기: 학습 데이터 = 1: 21.6

## Alpaca and its Friends
------

LLM의 실제 서비스 활용을 위해선 Pretrain ⇒ SFT ⇒ RLHF의 3단계 학습 필요하다.

- LLaMA 2 : 상업적 활용이 가능한 Pretrained LLM
    - 상업적 활용을 위한 공개 SFT, RLHF 학습 데이터 필요
- SFT/RLHF 학습 데이터 구축 비용은 매우 막대함
    - Annotator 고용/데이터 구축 가이드라인 제작/데이터 검수

이 때 필요한 데이터가 Demonstration Data이며 이 데이터의 필수 요건은 아래와 같다.

- 다양성 : Prompt는 사용자들의 다양한 요청 사항을 담고 있어야 함
- 적절성 : 답변은 Prompt에 대응하는 적절한 내용을 포함해야 함
- 안전성 : 답변은 ChatBot으로서 혐오/차별/위험 표현을 담지 않아야 함

데이터 크기는 1만 건 이상이고 양질의 데이터를 충분히 확보하는 것 이 중요하다.

이런 데이터를 확보하기 위해 `Self-Instruct` 방법을 고안하였으며, 이는 고품질의 Demonstration 데이터를 자동으로 구축하기 위한 방법론이다.

- GPT API를 이용하여 데이터 구축
    - Human Annotator 수준의 데이터 구축이 가능
    - Human Annotator 대비 적은 비용 소모

이 방법은 크게 5 step으로 구성된다.

### Prompt Pool

데이터 수집을 위한 초기 Prompt(Instruction) Pool 확보해야한다.
- Human Annotation을 통해 175개 확보
- 다양한 Task에 대한 Prompt-Answer Pair 구축

ex. 

- 요청
    - Instruction : 다음 규칙에 부합하는 단어를 알려줘
- ICL Sample
    - Input : h_ar_
- Input
    - Output : haart, heard, hears, heart, hoard, hoary

### Instruction Generation

추가적인 Prompt 생성하는 단계이다.

- 기존 Pool 내 Prompt(Instruction + Input) 8개를 샘플링하여 In-Context Learning에 활용
- 각각의 태스크에 적합한 Instruction 구조 사용

ex. 

- 요청: 다음 예시를 보고 새로운 작업(task)을 생성해줘

- ICL Sample
    - Task1 : 다음 문장은 긍정문이야?
    ...
    - Task 8: 다음 뉴스를 요약해줘
- Input
    - Task 9 : 여행 예산을 짜줘(LLM이 Task 1 ~ Task 8을 통해 만들어낸 Task)

### Classification Task Identification & Instance Generation

Classification Task Identification는 다음과 같다.

- 생성된 Instruction의 분류문제 여부 판단 단계
    - 향후 단계에서 분류 문제 여부에 따라 다르게 진행
- 고정된 In-Context Learning(Instruction-task label) 이용

ex. 

- 요청: 다음 작업들이 분류문제인지 판단해줘
- ICL Sample
    - 내 직업과 적성을 고려할 때, 적절한지 알려줘
    - Task : Classification
    - 사람의 주민등록번호를 알려줘
- Input
    - Task : Non-Classification
    - 여행 예산을 짜줘

Instance Generation은 다음과 같다.

- 생성된 Instruction에 부합하는 답변(Instance)를 생성하는 단계
- 고정된 In-Context Learning Sample(Instruction-Input-Output) 이용

ex.

- 요청: 다음 예시를 보고 작업에 적절한 답변을 생성해줘
- ICL Sample
    - Instruction : 다음 뉴스를 요약해줘
    -  Input : {뉴스 원문} Output {뉴스 요약문}
- Input
    - Instruction : 여행 예산을 짜줘
    -  Input : {여행 정보} Output :{여행 예산 답변}

### Filtering and Post Processing

- 데이터 다양성 및 품질 확보를 위한 후처리 단계
- 기존 Task Pool 내 데이터와 일정 유사도 이하인 데이터만 Task Pool 추가
- 텍스트로 해결할 수 없는 태스크 제거
    - 이미지 및 그래프 데이터 필요 시

### Supervised Fine-Tuning

- Self-Instruct를 통해 생성한 데이터를 이용한 SFT 학습
- Human Annotation 데이터 없이 LLM에 대한 SFT 학습 진행 가능
- 기존 SFT 학습 방법론 활용

### Alpaca

Alpaca는 2023년 Stanford에서 발표한 LLM SFT 학습 프로젝트이다.

- Alpaca : GPT API를 이용한 SFT 데이터 생성 및 학습 프레임워크
- Self-Instruct 방식으로 생성한 데이터를 이용한 LLaMA SFT 학습
- 초기 175개 데이터를 이용하여 52,000개 SFT 학습 데이터 생성
    - Open-Source LLM을 위한 학습 데이터 생성 효율적 생성 방법론
- LLaMA 7B 훈련 진행

Alpaca Data에 따른 성능을 비교하면 다음과 같다.

- 데이터 생성 API 종류 및 성능에 따라 데이터 품질 결정(GPT-3.5 < GPT-4)
- API가 수행하지 못하는 태스크들은 SFT 학습으로 성능 개선 한계
    - GSM : 대학 수학 문제
    -  TydiQA : 다국어 이해 능력
- API가 잘 이해하는 영역에서 성능 개선 기대 가능
    - Codex-Eval : 코드 생성 문제

## LLM Evaluation Methods
-------

'LLM을 평가한다'의 의미는 목적에 맞게 얼마나 태스크를 잘 수행하는지를 평가한다. 그러므로 LLM 평가는 기존 태스크 수행 능력 평가와 상이히다.

- 기존 태스크 수행 능력
    - 평가 목적 : 모델 해당 태스크 수행능력
    - 평가 데이터 : 해당 태스크 데이터
    - 평가 방법론 : 태스크 평가 파이프라인 및 계산 방법론
- LLM 평가
    - 평가 목적 : LLM의 범용 태스크 수행능력
    - 평가 데이터 : 범용적 능력 평가 데이터
    - 평가 방법론 : 각 태스크 별 상이

LLM을 평가 목적은 다음과 같다.

- LLM의 평가 및 활용 목적
    - 태스크 수행 능력: 다양한 태스크에 대해 적절한 답변을 출력할 수 있는가?
    -  안전성: 답변 내 위험하거나 편향된 내용은 없는가?
- 수행 태스크 범위: 사실상 한정되지 않음
    - 태스크 관점: 코드 작성/ 스토리 생성/ 제안서 작성/ 자기소개서 수정/ 문서 요약/ 사용자 감정 공감
    - 능력 관점: 객관적 사실 판단/ 논리적 추론/ 수학적 추론/ 일반 상식
- 안정성 범위: 정의하기 모호
    - 욕설 사용 여부: 비꼬는 문장은 가능한가?
    - 사회적 편향: 편향의 기준 정의 방법론
    - 유용성 : 유용한 답변 정의 방법론

### LLM Evaluation Datasets

LLM을 평가할 때는 범용적인 태스크 수행 능력을 평가하기 때문에 평가 목적에 따라서 각 데이터를 구축 및 활용한다.

- MMLU(Massive Multitask Language Understanding)
    - LLM의 **범용 태스크 수행 능력** 평가용 데이터셋
    - 다양한 평가 목적에 따른 데이터를 수집 및 통합한 데이터 묶음
    - 총 57개 태스크로 구성(생물, 정치, 수학, 물리학, 역사, 지리, 해부학 등)
- 객관식 형태로 평가 진행 : 정답 보기를 생성하면 맞춘 것으로 간주

- HellaSwag : 사람이 가지고 있는 상식 평가 데이터셋로 **일반 상식 능력**을 평가
    - 사람은 매우 쉽게 해결 가능한 태스크로 구성
    - LLM의 일반 상식 보유 능력 평가
    - 주어진 문장에 이어질 자연스러운 문장 선택
- 객관식 형태로 평가 진행 : 정답 보기를 생성하면 맞춘 것으로 간주

- HumanEval : LLM의 **코드 생성 능력** 평가 데이터셋
    - 함수 명 및 docstring 입력
    - docstring : 해당 함수의 수행 과정 및 의도하는 결과물 명시
    - LLM이 생성한 코드의 실제 실행 결과물을 이용하여 평가 진행
- 실행 결과물이 실제값과 일치 시 맞춘 것으로 간주

### LLM-Evaluation-harness

`llm-valuation-harness`는 자동화된 LLM 평가 프레임워크이다.

- MMLU/HellaSwag/HELM 등 다양한 Benchmark 데이터를 이용한 평가 가능
- 평가 방식
    1. K-Shot Example과 함께 LLM 입력
    2. 각 보기 문장을 생성할 확률 계산
    3. 확률이 가장 높은 문장을 예측값으로 사용 → 정답 여부 확인
- 평가 데이터셋 구성 요소
    1. (optional) 고정된 Few-Shot Example : 강건한 평가를 위해 동일한 Example 사용
    2. Instruction : 해당 태스크에 대한 묘사
    3. Choices : 정답 문장을 포함하는 보기 문장(MMLU/HellaSwag와 동일)
    4. Correct Answer : 정답 문장
- 하나의 모델에 대한 평가 가능
    - 특정 LLM의 특정 Task에 대한 지표 산출
    - 모델 간 비교 : 각 모델의 지표 비교

### G-Eval

창의적인 글쓰기 능력을 평가하며 llm-evaluation-harness는 정답 문장이 존재하여 이를 이용하여 평가하는 반면에 실제 LLM 활용의 상당수는 정답이 존재하지 않는 태스크이다.

- 창의적 글쓰기 태스크 : 자기소개서 수정, 광고 문구 생성, 어투 변경 등
- 실제 생성문을 이용한 정성적 품질 평가

평가를 위해 `Human Evaluation`를 사용할 수 있지만 높은 비용 및 긴 시간 소모한다는 단점이 있다.
- Pilot Test 등 사용 부적합, 다양한 모델에 대한 평가 어려움

이러한 문제를 보완하기 위해서 `G-Eval`를 사용한다.

- `G-Eval` : GPT-4를 이용한 생성문 평가 방법론
    - 창의성 및 다양성이 중요한 태스크에 활용 가능
    - 정답문(Reference)가 존재하지 않아도 평가 가능
- 평가 방법(예시 : 요약 태스크)
    1. 평가 방식에 대한 Instruction 구성
    2. 평가 기준 제시
    3. 평가 단계 생성 : Auto CoT를 통해 모델이 스스로 평가 단계를 정의
        - Auto CoT : 모델 스스로 추론 단계를 구축하는 프롬프트 방식
    4. 1 ~ 3의 문장을 Prompt로 사용하여 각 요약문에 대한 평가 진행
        - GPT-4는 Prompt, 뉴스 원문, 생성된 요약문을 입력으로 점수를 생성

예를 들어 창의적 글쓰기 능력을 평가를 가정하자.

- G-Eval : LLM의 생성문(요약문)에 대한 특정 기준을 이용한 평가 가능
- 평가 기준 설정만 수행하면 창의적 글쓰기 능력 평가 가능
    - GPT4-API 비용으로 정성점수 산출 가능
- 평가 결과가 괜찮은 평가 방식을 측정하기 위해서 Human Evaluation Score와 Correlation 측정할 수 있다.

G-Eval 방식의 주의사항은 다음과 같다.

- 명확한 평가 기준 : 평가 기준에 따라 모델이 평가를 진행하게 됨
- 평가 모델 선택 : 모델의 성능에 따라 평가 결과물의 신뢰도 결정
    - sLLM을 이용한 평가 시 점수 신뢰도 하락
    - GPT-4-turbo 등 안정적 결과물 산출이 가능한 모델 선택 필수
- 검수 필요 : 평가 점수 신뢰도 확보를 위한 일부 데이터 검수 필요
    - 태스크 평가 난이도에 따라 점수 신뢰도 상이