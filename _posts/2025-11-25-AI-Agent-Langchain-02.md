---
title: "RAG"
description: "RAG에 대한 기본 개념에 대한 내용을 정리한 포스트입니다."

categories: [AI Agent, LangChain]
tags: [LLM, LangChain, RAG]

permalink: /ai-agent/inflearn/langchain-02

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-11-25
last_modified_at: 2025-11-25
---

`Retrieval Augmented Generation(RAG)`는 LLM 어플리케이션 개발에서 중요한 개념으로 LLM 등장 전 NLP에서부터 사용되었다.

- Retrieval
    - 데이터를 가져오는 것
    - 구체적으로는 "컴퓨터 시스템에 저장된 자료를 취득하는 것"이라는 뜻도 있음
    - 언어 모델이 가지고 있지 않은 정보를 가져오는 것

- Augmented
    - AR/VR에 사용되는 것과 같은 단어
    - 마치 사실인 것 처럼 Retrieval 된 데이터를 LLM에게 주면서, "마치 이 정보를 아는 것 처럼"

- Generation
    - 생성
    - 가져온 데이터를 제공하면, 이 정보를 알고 있었던 것처럼 "답변을 생성"


개발자가 해야할 일은 데이터를 잘 가져와서 LLM에게 잘 전달해야한다.

- 데이터를 잘 가져오기 위해서는 데이터를 저장하는 과정이 매우 중요하다.
- 데이터를 LLM에 잘 전달하기 위해서는 `프롬프트`를 잘 활용해야하며 이 때, 문맥을 어떻게 제공할 것인가도 매우 중요하다.
    - 데이터를 잘 가져와도 제대로 전달하지 못하면 LLM이 올바른 답변을 주지 못하므로 이러한 문제를 보완하기 위해 LangChain을 사용한다.


## Vector Database
--------

사용자가 원하는 정보는 사용자의 질문과 관련있는 데이터라고 할 수 있다. 그렇다면 관련성이 있다는 것을 어떻게 판단할까

관련성 파악을 위해 `vector`를 활용하여 단어 또는 문장의 유사도를 파악해서 관련성을 측정한다.

Vector을 생성하는 방법은 다음과 같다.

- Embedding 모델을 활용해서 vector를 생성
- 문장에서 비슷한 단어가 자주 붙어있는 것을 학습

그렇다면 Vector Database는 무엇일까?

- Embedding 모델을 활용해 생성된 vector를 저장
    - 단순히 vector만 저장하면 안되고 `metadata`도 같이 저장
        - 문서의 이름, 페이지 번호 등등을 같이 저장 &rarr; LLM이 생성하는 답변의 퀄리티가 상승

- Vector를 대상으로 유사도 검색 실시
    - 사용자의 질문과 가장 비슷한 문서를 가져오는 것 &rarr; Retrieval
        - 문서 전체를 활용하면 속도도 느리고, 토큰 수 초과로 답변 생성이 안될 수 있음
        - 위의 단점을 해결하기 위해 문서를 chunking하고 나눠서 저장해야함
    - 가져온 문서를 prompt를 통해 LLM에 제공 &rarr; Augmented
    - LLM은 prompt를 활용해서 답변 생성 &rarr; Generation