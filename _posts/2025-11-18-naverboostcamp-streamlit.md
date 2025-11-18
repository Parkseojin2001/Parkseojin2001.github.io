---
title: "Streamlit을 활용한 웹 프로토타입 구현하기"
description: "효율적인 서비스 프로토타입 구현을 위해 파이썬 기반의 오픈소스 프레임워크인 Streamlit에 대한 내용을 정리한 포스트입니다."

categories: [Naver-Boostcamp, AI Production]
tags: [Streamlit, User interface]

permalink: /naver-boostcamp/ai-production/02

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-11-18
last_modified_at: 2025-11-18
---

웹 서비스를 배포하기 전에 이 제품이 제대로 작동하는 지 확인하기 위한 샘플 버전이 필요하다. 이를 위해 `프로토타입`을 개발하여 런칭 전에 Test를 진행한다.

주로 웹페이지로 제작을 하며 AI 모델의 경우 input을 넣어 Output을 확인할 수 있도록 설정한다.

일반적인 데이터 분석가 또는 데이터 사이언티스트의 경우 프론트엔드/PM 조직과 협업을 하여 웹 작업을 진행하는 경우 빠른 이터레이션이 어렵다. 

이렇게 **다른 조직의 도움 없이** 빠르게 웹 서비스를 만드는 방법이 필요하다. 

AI 모델을 Test하기 위해서는 아래의 조건을 만족하는 것이 좋다.

- 엔지니어는 노트북을 활용해 쉽게 프로토타입 구현 가능
    - 단, 대시보드처럼 레이아웃을 잡기 어려움 &rarr; 프론트 개발 진행
- 기존 코드 조금만 수정해서 효율적으로 웹 서비스 개발이 가능
- 자바스크립트, React, Vue 등을 사용해 프로토타입을 만들지 않아 시간 효율적
- HTML/자바스크립트 + Flask/Fast API 활용하지 않아 리소스 소모가 적음

## Streamlit
--------

위의 조건을 만족할 수 있는 것이 바로 `Streamlit` 이다.

`Streamlit`는 다음과 같은 장점이 있다.

- 파이썬 스크립트 코드를 조금만 수정하면 웹을 띄울 수 있음
- 백엔드 개발이나 HTTP 요청을 구현하지 않아도 됨
- 다양한 Component 제공해 대시보드 UI 구성 할 수 있음
- Streamlit Cloud도 존재해서 쉽게 배포할 수 있음(단, Community Plan은 Public Repo 만 가능)
- 화면 녹화 기능(Record) 존재

Streamlit는 `pip install streamlit` 를 통해 설치할 수 있으며 실행은 `streamlit run 01-streamlit-hello.py` 로 실행할 수 있다.

Streamlit 개발은 아래와 같은 흐름으로 진행된다.

<img src="../assets/img/post/naver-boostcamp/streamlit-flow.png">

참조 : [Streamlit 공식 문서](https://docs.streamlit.io/)