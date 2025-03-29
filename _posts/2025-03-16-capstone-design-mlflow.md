---
title: "MLflow 소개 및 Tutorial"
excerpt: "MLflow"

categories:
  - Capstone Design
tags:
  - [capstone-design]

permalink: /capstone-design/mlflow/

toc: true
toc_sticky: true

date: 2025-03-16
last_modified_at: 2025-03-16
---

- 머신러닝 프로세스의 관리할 수 있는 오픈소스인 MLflow에 대한 소개 및 간단한 Tutorial에 대한 글입니다.

---


## 🦥 MLflow

- MLflow는 End to End로 머신러닝 라이프 사이클을 관리할 수 있는 오픈소스
 
**주요 기능**

1) MLflow Tracking
  - 모델에 대한 훈련 통계(손실, 정확도 등) 및 하이퍼 매개변수를 기록
  - 나중에 검색할 수 있도록 모델을 기록(저장)한다.
  - MLflow 모델 레지스트리를 사용하여 모델을 등록하여 배포를 활성화한다.
  - 모델을 로드하여 추론에 사용한다.

2) MLflow Projects
  - 머신러닝 코드를 재사용 가능하고 재현 가능한 형태로 포장
  - 포장된 형태를 다른 데이터 사이언티스트가 사용하거나 프러덕션에 반영

3) MLflow Models
  - 다양한 ML 라이브러리에서 모델을 관리하고 배포, Serving, 추론


