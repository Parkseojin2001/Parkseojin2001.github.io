---
title: "Batch Serving과 Airflow"
description: "Batch Serving 개념과 이를 구현하기 위한 핵심 워크플로우 관리 도구 Apache Airflow에 대한 내용을 정리한 포스트입니다."

categories: [Naver-Boostcamp, Model Serving]
tags: [model Serving, batch Serving, Apache Airflow]

permalink: /naver-boostcamp/model-serving/02

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2026-01-09
last_modified_at: 2026-01-09
---

`Batch Serving`은 일정 기간 데이터 수집 후 일괄 학습 및 결과 제공하는 특성이 있으며 대량의 데이터 처리할 때 효율적이며 주로 모델을 **주기적**으로 학습시킬 때 사용하는 방법이다.

- 예측 코드를 주기적으로 실행해서 예측 결과를 제공
- **Job Scheduler는 Apache Airflow를 주로 사용**
- 학습과 예측을 별도 설정을 통해 수행
    - 학습: 1주일에 1번
    - 예측: 10분, 30분, 1시간에 1번씩

## Batch processing
------

`Batch processing` 은 소프트웨어 프로그램을 자동으로 실행하는 방법으로 예약된 시간에 자동으로 실행하는 것을 의미한다.

> ❗️Batch Processing과 Batch Serving의 차이
>
> `Batch Processing` : 일정 기간 동안 일괄적으로 작업을 수행<br>
> `Batch Serving` : 일정 기간 동안 일괄적으로 머신러닝 예측 작업을 수행<br>
>
> Batch Processing이 더 큰 개념이며, Batch로 진행하는 작업에 Airflow를 사용할 수 있다. 

### Crontab

Aifrlow 등장 전에는 대표적으로 Linux의 `Crontab`을 사용하였다.

- (서버에서) crontab -e 입력
- 실행된 에디터에서 0 * * * * predict.py 입력(0 * * * * 은 매 시 0분을 의미)
- OS에 의해 매 시 0분에 predict.py가 실행
- Linux는 일반적인 서버 환경이고, Crontab도 기본적으로 설치되어 있기 때문에 매우 간편
- 간단하게 Batch Processing을 할 때 Crontab도 가능한 선택

이 때 `Cron` 표현식을 활용하며 Batch Processing의 스케줄링을 정의한 표현식이다.

이 표현식은 다른 Batch Processing 도구에서도 자주 사용된다(Airflow에서도 사용).

<img src="https://media.licdn.com/dms/image/v2/D4E12AQEaUoC2X07owQ/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1720803350525?e=2147483647&v=beta&t=1XH-awhdk2nqi6qFUDPbd1rPJ8jVspuspZPx64hb2o4" width="500" height="350">

Cron 표현식이 익숙하지 않을 때 참고할 사이트

- [Cron 표현식을 만들기](http://www.cronmaker.com)
- [Cron 표현식 읽기](https://crontab.guru/)

Linux Crontab의 문제점이 있다.

- **재실행 및 알림**
    - 파일을 실행하다 오류가 발생한 경우, Crontab이 별도의 처리를 하지 않음
    - ex. 매주 일요일 07:00에 predict.py를 실행하다가 에러가 발생한 경우, 알림을 별도로 받지 못함
    - 실패할 경우, 자동으로 몇 번 더 재실행(Retry)하고, 그래도 실패하면 실패했다는 알림을 받아야 대응할 수 있음
- **과거 실행 이력 및 실행 로그**를 보기 어려움
- 여러 파일을 실행하거나, **복잡한 파이프라인**을 만들기 힘듦

Crontab은 간단히 사용할 수는 있지만, 실패 시 재실행, 실행 로그 확인, 알림 등의 기능은 제공하지 않기 때문에 좀 더 정교한 스케줄링 및 워크플로우 도구가 필요하다.

여러 도구가 등장했지만 이 중 가장 많이 사용하는 도구는 `Airflow` 다.

## Apache Airflow
------

Airflow 등장 후, 스케줄링 및 워크플로우 도구의 표준이다.

- 에어비앤비(Airbnb)에서 개발했으며 2점대 버전이 존재. 업데이트가 매우 빠름
- 스케줄링 도구로 무거울 수 있지만, 거의 모든 기능을 제공하고, 확장성이 좋아 일반적으로 스케줄링과 파이프라인 작성 도구로 많이 사용
- 데이터 엔지니어링 팀, MLOps 팀에서 많이 사용

Airflow를 많이 사용하는 이유는 워크플로우 관리 도구로써 코드로 작성된 데이터 파이프라인 흐름을 `스케줄링`하고 `모니터링`하는 목적이다. 또한 데이터 처리 파이프라인을 효율적으로 관리하여 시간과 자원을 절약하도록 한다.

Airflow의 주요 기능은 아래와 같다.

1. 파이썬을 사용해 스커줄링 및 파이프라인 작성
2. 스케줄링 및 파이프라인 목록을 볼 수 있는 웹 UI 제공
    - 성공/실패/진행 중 상태를 확인 가능

    <img src="https://airflow.apache.org/docs/apache-airflow/2.2.0/_images/dags.png">

3. 특정 조건에 따라 작업을 분기할 수도 있음(Branch 사용)

    <img src="https://airflow.apache.org/docs/apache-airflow/2.2.0/_images/edge_label_example.png">

Airflow의 핵심 개념으로 4가지가 있다.

1. `DAGs(Directed Acyclic Graphs)`
    - Airflow에서 작업을 정의하는 방법, 작업의 흐름과 순서 정의
2. `Operator`
    - Airflow의 작업 유형을 나타내는 클래스
    - BashOperator, PythonOperator, SQLOperator 등 다양한 Operator 존재
3. `Scheduler`
    - Airflow의 핵심 구성 요소 중 하나. DAGs를 보며 현재 실행해야 하는지 스케줄을 확인
    - DAGs의 실행을 관리하고 스케줄링
4. `Executor`
    - 작업이 실행되는 환경
    - LocalExecutor, CeleryExecutor 등 다양한 Executor가 존재

Airflow의 기본 아키텍처는 아래 그림과 같이 표현한다.

<img src="https://airflow.apache.org/docs/apache-airflow/2.2.0/_images/arch-diag-basic.png">

Airflow에서 Batch Scheduling을 위한 DAG 생성을 하는 과정은 다음과 같다.

먼저 Airflow에서는 스케줄링할 작업을 DAG이라고 부릅니다. DAG은 Directed Acyclic Graph의 약자로, Airflow에 한정된 개념이 아닌 소프트웨어 자료구조에서 일반적으로 다루는 개념이며 이름 그대로 **'순환하지 않는 방향이 존재하는 그래프'**를 의미합니다.

Airflow는 Crontab처럼 단순히 하나의 파일을 실행하는 것이 아닌, 여러 작업의 조합도 가능합니다.

- DAG 1개 : 1개의 파이프라인

- Task : DAG 내에서 실행할 작업

하나의 DAG은 여러 Task의 조합으로 구성되며, 이러한 DAG 파일들은 DAG Directory를 통해 물리적으로 관리됩니다.

- `DAG Directory`
    - DAG 파일들을 저장
    - 기본 경로는 `$AIRFLOW_HOME/dags`
    - DAG_FOLDER 라고도 부르며, 이 폴더 내부에서 폴더 구조를 어떻게 두어도 상관없음
    - Scheduler에 의해 .py 파일은 모두 탐색되고 DAG이 파싱

### Airflow DAG 작성

DAG 작성을 요약하자면 다음과 같다.

1. AIRFLOW_HOME 으로 지정된 디렉토리에 dags 디렉토리를 생성하고 이 안에 DAG 파일을 작성
2. DAG은 파이썬 파일로 작성. 보통 하나의 .py 파일에 하나의 DAG을 저장
3. DAG 파일을 저장하면, Airflow 웹 UI에서 확인할 수 있음
4. Airflow 웹 UI에서 해당 DAG을 ON으로 변경하면 DAG이 스케줄링되어 실행
5. DAG 세부 페이지에서 실행된 DAG Run의 결과를 볼 수 있음

이때, DAG 파일은 크게 3가지로 구성되어있다.
- DAG 정의
- Task 정의
- Task 순서 정의
