---
title: "[BoostCamp AI Tech / Product Serving] Docker"
description: " 가상화의 개념과 Docker그리고 Docker Compose의 기본 개념과 사용법을 정리한 포스트입니다."

categories: [NAVER BoostCamp AI Tech, AI Production]
tags: [model serving, Docker, Docker Compose]

permalink: /naver-boostcamp/model-serving/06

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2026-01-14
last_modified_at: 2026-01-14
---

Docker에 대해 먼저 설명하기 전에 `가상화`라는 개념을 알아야 한다.

개발할 때, 서비스 운영에 사용하는 서버에 직접 들어가서 개발하지 않는다. Local 환경에서 개발하고, 완료되면 Staging 서버, Production 서버에 배포한다.

개발을 진행한 Local 환경과 Production 서버 환경이 다른 경우 설치할 때 다르게 진행해야한다. 만약 같은 OS라고 해도 문제가 생길 수 있다.

이러한 문제를 해결하기 위해 소프트웨어 환경을 만들고, Local, Production 서버에서 그대로 활용하는 방식으로 사용한다.

- 개발(Local)과 운영(Production) 서버의 환경 불일치가 해소
- 어느 환경에서나 동일한 환경으로 프로그램을 실행할 수 있음
- 개발 외에 Research도 동일한 환경을 사용할 수 있음

가상화 기술로 주로 VM(VirtualMachine)을 사용하였다.

VM : 호스트 머신이라고 하는 실제 물리적인 컴퓨터 위에, OS를 포함한 가상화 소프트웨어를 두는 방식

<img src="https://img.vembu.com/wp-content/uploads/2019/02/VM-vs-Containers_02.png">

OS 위에 OS를 하나 더 실행시키는점에서 VM은 굉장히 리소스를 많이 사용하는 단점이 있다(주로 무겁다고 표현).

이러한 문제를 해결하기 위해 등장한 기술이 바로 `Container` 이다. 이 기술의 등장으로 이전보다 빠르고 가볍게 가상화를 구현할 수 있다.

- `Container` : VM의 무거움을 크게 덜어주면서, 가상화를 좀 더 경량화된 프로세스

이 도구가 바로 Docker이다.

<img src="https://img.vembu.com/wp-content/uploads/2019/02/VM-vs-Containers.png">

## Docker

Docker는 2013년에 오픈소스로 등장했으며 컨테이너에 기반한 개발과 운영을 매우 빠르게 확장할 수 있다.

Docker Image로 만들어두고, 재부팅하면 Docker Image의 상태로 실행한다.

- `Docker Image`    
    - 컨테이너를 실행할 때 사용할 수 있는 "템플릿"
    - Read Only
- `Docker Container`
    - Docker Image를 활용해 실행된 인스턴스
    - Write 가능

<img src="https://low-orbit.net/images/docker-container-vs-image2.png">

Docker로 할 수 있는 작업으로 다른 사람이 만든 소프트웨어를 가져와서 바로 사용하는 것이 있다. 이때 다른 사람이 만든 소프트웨어가 `Docker Image`이다.

`Docker Image` 에는 OS 설정을 포함한 실행 환경에 대한 내용이 담겨있으며 Linux, Window, Mac 어디에서나 동일하게 실행할 수 있다.

또한, 자신만의 이미지를 만들어 다른 사람에게 공유할 수 있다. 이때, 원격 저장소에 저장하면 어디에서나 사용할 수 있다.

- 원격 저장소 : Container Registry라고 하며 대표적으로 dockerhub, GCR, ECR이 있다.
    - 회사에서 서비스를 배포할 때는 원격 저장소에 이미지를 업로드하고, 서버에서 받아서 실행하는 방식으로 진행

### Docker 기본 명령어

docker에서 자주 사용하는 명령어로 다음과 같은 것이 있다.

- `docker` : 현재 OS에 docker가 있는지 확인
- `docker pull "이미지 이름: 태그"` : docker 이미지를 다운
- `docker images` : 다운받은 이미지 확인
- `docker run "이미지 이름: 태그"` : 다운맏은 이미지 기반으로 Docker Container를 만들고 실행
    - ex. `docker run --name mysql-tutorial -e MYSQL_ROOT_PASSWORD=1234 -d -p 3306:3306 mysql`
        - `--name mysql-tutorial` : 컨테이너 이름으로 지정하지 않으면 랜덤으로 생성
        - `-e MYSQL_ROOT_PASSWORD=1234` : 환경변수 설정으로 사용하는 이미지에 따라 설정이 다름
        - `-d` : 데몬의 약자로 백그라운드 모드로 실행된다.
        -  `-p 3306:3306` : 포트 지정 
- `docker ps` : 실행한 컨테이너를 확인
- `docker exec -it "컨테이너 이름(or ID)" /bin/bash` : 실앵되고 있는지 확인하기 위해 컨테이너에 진입
    - Compute Engine에서 SSH와 접속하는 것과 유사
- `mysql -u root -p` : MYSQL 프로세스로 들어가면 MySQL 쉘 화면이 보임(먼저 컨테이너 진입이 필요)
- `docker ps -a` : 작동을 멈춘 컨테이너를 확인
- `docker rm "컨테이너 이름(ID)` : 멈춘 컨테이너를 삭제
    - 뒤에 `-f`로 실행중인 컨테이너도 삭제 가능

만약 docker가 실행할 때 파일을 공유하는 방법으로 두 가지 방법이 있다.

1. Docker Container 내부에 파일을 저장 
2. 외부(로컬)에 저장하는 방법이 있다.

첫번 째 방법은 Host와 Container와 파일 공유가 되지 않으며 특별한 설정이 없으면 컨테이너를 삭제할 때 파일이 사라진다.

두번 째 방법은 파일을 유지하고 싶다면 Host(컴퓨터)와 Container의 저장소를 공유하고 로컬에 올리는 방식인데 이 때 `Volume Mount`를 진행하면 Host와 Container의 폴더가 공유된다.

`-v` 옵션을사용하며, `-p`(Port)처럼 사용하면된다.

- `-v Host_Folder:Container_Folder`
- ex. `docker run -it -p 8888:8888 -v /some/host/folder/for/work:/home/jovyan/workspacejupyter/minimal-notebook`

### Docker Image 만들기

이미지를 다운받아 사용하는 방법도 있지만 직접 Docker Image를 만드는 방법도 있다.

`Dockerfile` 파일을 만들어 아래와 같이 작성하여 Docker Image를 만들 수 있다. 이 파일에는 Docker Image를 빌드하기 위한 정보가 저장되어있다.

```Dockerfile
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

COPY . /app
WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONBUFFERED=1

RUN pip install pip==23.0.1 && \
    pip install poetry==1.2.1 && \
    poetry export -o requirements.txt && \
    pip install -r requirements.txt

CMD ["python", "main.py"]
```

- `FROM "이미지 이름:태그"` : 이미지 빌드에 사용할 베이스 이미지를 지정하고 대부분 베이스 이미지는 이미 만들어진 이미지를 사용한다.
- `COPY "로컬 디렉토리(파일)" "컨테이너 내 디렉토리(파일)"` : 컨테이너도 자체적인 파일 시스템과 디렉토리를 가지며 Dockerfile이 존재하는 경로 기준 로컬 디렉토리를 컨테이너 내부의 디렉토리로 복사
- `WORKDIR "컨테이너 내 디렉토리"` : Dockerfile의 RUN, CMD, ENTRYPOINT 등의 명령어를 실행할 컨테이너 경로 지정
- `ENV "환경변수 이름=값"` : 컨테이너 내 환경변수를 지정, 파이썬 애플리케이션의 경우 위의 두 값을 대체로 지정
- `RUN "실행할 리눅스 명령어"` : 컨테이너 내에서 리눅스 명령어를 실행
- `CMD ["실행할 명령어", "인자", ...]` : `docker run`으로 이 이미지를 기반으로 컨테이너를 만들 때, 실행할 명령어
- `EXPOSE` : 컨테이너 외부에 노출할 포트 지정
- `ENTRYPOINT` : 이미지를 컨테이너로 띄울 때 항상 실행하는 커맨드

이렇게 Dockerfile을 만들고 나면 이미지를 빌드(생성) 해야 한다. 빌드 명령어는 `docker build -t <빌드할 이미지 이름: 태그 이름> "Dockerfile이 위치한 경로"` 로 실행하면 된다.

- ex. `docker build -t 02-docker:latest.`
- `.`는 현재 폴더에 Dockerfile이 있음을 의미
- `-t "이미지 이름:태그"` 옵션으로 이미지 이름과 태그 지정할 수 있다.
    - 태크는 미 지정시 "latest"로 채워진다.

이렇게 만든 이미지로 컨테이너를 실행(`docker run`) 한다. 

### Registry에 Docker Image Push

본인이 만든 이미지를 인터넷에 업로드할 수 있는데 이미지 저장소인 Container Registry 에 Docker Image Push를 실행한다.

- Container Registry : Dockerhub, GCP GCR, AWS ECR 등

별도로 지정하지 않으면 기본적으로 Dockerhub를 사용한다.

Dockerhub에 올리는 방법은 다음과 같다.

1. `docker login` 명령어로 계정을 CLI에 연동
2. `docker tag "기존 이미지:태그" "새 이미지 이름:태그"`
    - dockerhub에 올릴 이미지 이름은 내 계정 ID/이미지 이름 형태여야 함
3. `docker push "이미지 이름:태그"` 

이렇게 Push한 이미지는 `docker pull` 명령어로 어디서든 받을 수 있다.

### Docker Image Size

ML 프레임워크 / 모델이 들어간 도커 이미지는 사이즈가 매우 크다. 왜 문제가 될까?

1. 빌드 타임
    - 빌드 속도: 새로운 이미지로 교체하기 위해 기다려야하는 시간이 늘어나서, 신속하게 대응하기 어려워짐
    - 네트워크 전송: 네트워크 전송량 = 비용.
    - 각종 비용: 네트워크 뿐만 아니라 디스크 용량, 빌드 시간에 따라 비용이 늘어남
2. 런타임
    - 컨테이너 시작 시 메모리에 로드되는 용량이 매우 큼
    - Image Pull 시 기다리는 시간이 매우 김
3. 호스트 머신 디스크
    - 이미지 하나의 용량이 매우 크기 때문에 VM 인스턴스 등 환경에서 디스크에 대한 용량 관리가 필요
    - 클라우드 환경에서는 디스크 용량 = 비용 이기 때문에 주의해야 함

이러한 문제를 해결하는 대표적인 방법으로 3가지가 있다.

1. 작은 Base Image 선정해서 사용하기
    - 알맞은 Base Image를 찾는게 중요
    - 사용할 OS package들만 설치 후 사용. 디버깅하는 목적으로 필요한 shell 환경도 중요(bash,zsh등이 안 깔려 있는 이미지도 있음)

    <img src="https://brianchristner.io/content/images/size/w1000/2015/07/Docker_Image_Size.png">
    
    - 예시. python
        - 파이썬 표준 이미지 : python:3.9
        - 파이썬 슬림 이미지 : python:3.9-slim - 슬림한 데비안 이미지 기반으로 Production 환경에 적합
        - 파이썬 알파인 이미지 : python:3.9-alpine - Alpine Linux기반으로 사용. 작은 크기.종속성은 수동 설치해야 할 수도 있음

2. Multi Stage Build 활용하기
    - 도커 이미지를 효율적으로 작성하고 최적화하기 위한 방법
    - 컨테이너 이미지를 만들며 빌드엔 필요하지만 최종 컨테이너 이미지엔 필요없는 내용을제 외하며 이미지를 생성하는 방법
    - 하나의 파일에 여러 이미지를 빌드하고 사용
    - Base 이미지를 바꾸면서 사용하며 2개 이상의 Dockerfile이 있는 것처럼 빌드 수행

    <img src="../assets/img/post/naver-boostcamp/docker-multi_stage.png">

3. Container를 잘 패키징하기
    - .dockerignore로 필요없는 파일들 제거
    - .pt, .pth 파일과 같은 큰 사이즈 asset들은 빌드에서 포함하지 않고, 빌드 타임 혹은 시작하는 스크립트에서 다운
    - Dockerfile 안에서 command들의 순서 최적화를 통해 캐싱을 최대한 이용
        - 변경 가능성이 낮은 명령어를 위로, 변경 가능성이 높은 명령어는 아래에 위치

## Docker Compose

`Docker Compose`는 아래와 같은 상황일 때 주로 사용한다.

- 하나의 Docker Image가 아니라 여러 Docker Image를 동시에 실행
- A Image로 Container를 띄우고, 그 이후에 B Container를 실행해야 하는 경우
    - ex. A는 Database고 B는 웹 서비스인 경우
- docker run할 때 옵션이 너무 다양하고, Volume Mount를 하지 않았다면 데이터가 모두 날라감


### Docker Compose 작성법

`Docker Compose`는 여러 컨테이너를 한번에 실행할 수 있으며 여러 컨테이너의 실행 순서, 의존도를 관리할 수 있으며 이를 `docker-compose.yml` 파일에 작성하여 관리한다.

```yml
# db 컨테이너와 app 컨테이너 

version: '3'    # docker compose 버전

services:   # 실행할 컨터이너 정의. 각 서비스는 하나의 컨테이너로 세부 설정을 저장
  db:
    image: mysql:5.7.12  # 이미지 명시
    environment:    # 환경 변수
      MYSQL_ROOT_PASSWORD: root
      MYSQL_DATABASE: my_database
    ports:  # 포트 설정
      - 3306:3306

  app:
    build:
      context: .
    environment:
      DB_URL: mysql+mysqldb://root:root@db:3306/my_database?charset=utf8mb4
    ports:
      - 8000:8000
    depends_on: # 여기에(db) 명시된 서비스가 뜬 이후에 실행
      - db
    restart: always # 컨테이너 재실행 정책
```

그 외에도

- volumes : 볼륨 정의. 호스트와 컨테이너의 저장소를 지정
- secrets : 보안이 필요한 데이터 전달
- configs : 컨테이너에 사용할 config 파일
- command : 컨테이너가 시작될 때 실행할 명령 지정

### Docker Compose 실행

Docker 이미지는 `docker-compose up` 로 실행하면 된다.

docker-compose.yml 파일을 분석하여 컨테이너 실행하며 이 때 필요한 이미지를 당기거나 구축하는 것과 같은 과정도 포함한다.

- `docker-compose up -d` : 백그라운드에서 실행하기(docker run -d와 동일)
- `docker-compose down` : 서비스 중단(컨테이너, 볼륨 등 삭제)
- `docker-compose log <서비스명>` : 로그 확인
- `docker-compose start` : 서비스 시작(`docker-compose up`을 한 후 가능)
- `docker-compose stop` : 서비스 중지(컨테이너 삭제 X, 나중에 다시 시작 가능)

참고로 docker-compose.yml 파일을 수정하고 `docker-compose up`을 즉 **컨테이너를 재생성하고, 서비스를 재시작**한다.

`docker-compose up`이 완료되면 `docker ps` 명령어나 `docker-compose ps`명령어로 현재 실행 중인 컨테이너를 확인할 수 있다.