---
title: "[BoostCamp AI Tech / Product Serving] 모델과 코드 배포"
description: " 모델 코드 에셋 관리, 이미지 레지스트리에 등록, 서버에 적용하는 과정을 정리한 포스트입니다."

categories: [NAVER BoostCamp AI Tech, AI Production]
tags: [model serving, CI/CD, Github Actions]

permalink: /naver-boostcamp/model-serving/08

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2026-01-19
last_modified_at: 2026-01-19
---

## 개발 프로세스
-----

현업의 개발 프로세스는 다음과 같다.

- Local
    - 각자의 컴퓨터에서 개발
    - 각자의 환경을 통일시키기 위해 Docker 등을 사용
- Dev
    - Local에서 개발한 기능을 테스트할 수 있는 환경
    - Test 서버
- Staging
    - Production 환경에 배포하기 전에 운영하거나 보안, 성능을 측정하는 환경
    - Staging 서버
- Production
    - 실제 서비스를 운영하는 환경
    - 운영 서버

개발 환경을 이렇게 나누는 이유는 실제 운영 중인 서비스에 장애가 발생하면 안되기 때문이다.

### Git Flow


[main] &rarr; [staging] &rarr; [dev] &rarr; [feature/기능 이름]

Pull Request / Review를 받을 때는 아래와 같은 flow를 갖는다.

[main] &larr; [staging] &larr; [dev] &larr; [feature/기능 이름]

- [main] branch는 Production Server에 연결
- [staging] branch는 Staging Server에 연결
- [dev] branch는 Dev Server에 연결
- [feature/기능 이름]은 local에서 실행

서버에 코드를 보내는 것과 반복적으로 진행할 Test를 어떻게 실행하는 방법은 뭘까?

Dev Branch에 Merge 되면 Local에서 Git Pull & Test 실행 후 코드 배포(FTP로 파일 전송)

하지만 이러한 일을 매번 하는 것은 매우 번거롭다.

### CI/CD - Continuous Intergration(CI)

위의 번거로운 일을 줄이기 위해 `CI/CD`를 구축한다.

- `CI(Continuous Intergration)`, 지속적 통합

    - 새롭게 작성한 코드 변경 사항이 Build, Test 진행한 후 Test Case에 통과했는지 확인
    - 지속적으로 코드 품질 관리
    - 10명의 개발자가 코드를 수정했다면 모두 CI 프로세스 진행

- `CD(Continuous Deploy/Delivery)`, 지속적 배포
    - 작성한 코드가 항상 신뢰 가능한 상태가 되면 자동으로 배포될 수 있도록 하는 과정
    - CI 이후 CD를 진행
    - dev / staging / main 브랜치에 Merge가 될 경우 코드가 자동으로 서버에 배포

위의 내용을 요약하면 다음과 같다.

- CI : 빌드, 테스트 자동화
- CD: 배포 자동화

개인 프로젝트를 할 때에는 서버를 많이 두지 않지 않기 때문에 Local에서 개발을 하고 Main으로 Merge 시 Production Server에 코드를 배포하는 방식으로도 진행한다.

### 모델 배포에서 주의할 점

- Docker 이미지
    - 사이즈가 보통 큰 편이어서 관리 필요
    - 호스트 머신의 디스크 용량 관리 필요(로그를 주기적으로 삭제하거나 클라우드로 보내기)
- 모델 버저닝
    - 모델 코드에 대한 확실한 버저닝
    - 어떤 버전의 모델이 현재 배포 중이고, 과거에는 어떤 버전으로 배포되었는지
    - 롤백이 필요하다면 어떤 버전으로 재배포해야 하는지
    - 모델의 버전 별 특징을 쉽게 볼 수 있어야 함
- 모델 아티팩트
    - 모델 이미지에 저장하는 것보다 S3, 오브젝트 저장소(S3, Cloud Storage)에 저장하는 것을 권장
    - 모델 버전과 아티팩트의 버전이 다른 경우도 있으므로 메타 정보를 확인해야 함
        - ex. 이를 처리하기 위해 알리는 도구: MLflow
    - 적합한 파일 권한 관리(VM 인스턴스, Object Storage 둘 다)
    - VM 인스턴스에서 pth 파일을 읽지 못한다면 정상 실행 불가능

    - Object storage에서 다른 사용자의 버킷을 삭제하면 롤백이 불가능(삭제 권한 제어 필요)


## Github Action
-----

Github에서 출시한 기능으로, 소프트웨어 Workflow 자동화를 도와주는 도구이다.

- Test Code
    - 특정 함수의 return 값이 어떻게되는지 확인하는 Test Code
    - 특정한 유형이 int가 맞나요?
    - Unit Test, End to End Test
- Batch
    - Prod, Staging, Dev 서버에 코드 배포
    - FTP로 전송할 수도 있고, Docker Image를 Push하는 방법 등
    - Node.js 등 다양한 언어 배포도 지원
- 파이썬, 쉘 스크립트 실행
    - Github Repo에 저장된 스크립트를 일정 주기를 가지고 실행
    - crontab의 대용
    - 데이터 수집을 통해 주기적으로 해야할 경우 활용할 수도 있음
- Github Tag, Release 자동으로 설정
    - Main 브랜치에 Merge 될 경우에 특정 작업 실행
    - 기존 버전에서 버전 Up하기
    - 새로운 브랜치 생성시 특정 작업 실행도 가능

그 외에도 다양한 Workflow를 만들 수 있다. 사용자가 만들어서 Workflow 템플릿을 공유하기도한다.

원하는 기능이 있는 경우 <기능> github action 등으로 검색!

- [Action Marketplace](https://github.com/marketplace?type=actions)
- [Awesome Github Action](https://github.com/sdras/awesome-actions)

### 사용 방석

1. 코드 작업
2. 코드 작업 후, Github Action으로 무엇을 할 것인지 생각
3. 사용할 Workflow 정의
4. Workflow 정의 후 정상 작동하는지 확인

핵심 개념: `Workflow`, `Evnet`, `Job`, `Step`, `Action`, `Runner`

- Workflow
    - 여러 Job으로 구성되고 Event로 Trigger(실행)되는 자동화된 Process
    - 최상위 개념
    - Workflow 파일은 YAML으로 작성되고, Github Repository의 `.github/workflows` 폴더에 저장

- Event
    - Workflow를 Trigger하는 특정 활동, 규칙
    - 특정 Branch로 Push하는 경우
    - 특정 Branch로 Pull Request하는 경우
    - 특정 시간대에 반복(Cron)

- Job
    - Runner에서 실행되는 Steps의 조합
    - 여러 Job에 있는 경우 병렬로 실행하며, 순차적으로 실항할 수도 있음
    - 다른 Job에 의존 관계를 가질 수 있음(A Job Sucess 후 B Job 실행)

- Step
    - Job에서 실행되는 개별 작업
    - Action을 실행하거나 쉘 커맨드 실행
    - 하나의 Job에선 데이터를 공유할 수 있음

- Action
    - Workflow의 제일 작은 단위
    - Job을 생성하기 위해 여러 Step을 묶은 개념
    - 재사용이 가능한 Component
    - 개인적인 Action을 만들 수도 있고, Marketplace의 Action을 사용할 수도 있음

- Runner
    - Github Action도 일종의 서버에서 실행되는 개념
    - Workflow가 실행될 서버
    - Github-hosted Runner: Github Action의 서버를 사용하는 방법
        - 성능 : vCPU 2, Memory 7GB, Storage 14GB
    - Self-hosted Runner : 직접 서버를 호스팅해서 사용하는 방법


