---
title: "Linux, 쉘 스크립트"
description: "서버 프로그래밍을 위해 가장 중요한 Linux에 대한 지식과 Shell Command들에 대한 내용을 정리한 포스트입니다."

categories: [Naver-Boostcamp, AI Production]
tags: [Linux, Shell Script, Command Line]

permalink: /naver-boostcamp/ai-production/01

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-11-17
last_modified_at: 2025-11-17
---

## Linux
----

`Linux`는 서버에서 자주 사용하는 OS로 오픈소스이다. 유닉스 기반이라 Stability, Reliability가 보장된다. 

Linux를 알아야하는 주요한 이유는 쉘 커맨드와 쉘 스크립트를 이용하기 때문이다. 

- `CLI(Command Line Interface)` : Terminal
- `GUI(Graphic User Interface)` : Desktop

Linux는 굉장히 다양한 종류의 배포판이 있는데 대표적으로 아래와 같다.

- Debian
- Ubuntu
- Redhat
- CentOS


## Shell Command
----

`Shell`은 사용자가 문자를 입력해 컴퓨터에 명령할 수 있도록 하는 프로그램이다.

`터미널/콘솔`은 쉘을 실행하기 위해 문자 입력을 받아 컴퓨터에 전달하며 프로그램의 출력을 화면에 작성한다.

- sh : 최초의 쉘
- bash : Linux 표준 쉘
- zsh : Mac 카탈리나 OS 기본 쉘

쉘을 사용하는 경우는 다음과 같다.

- 서버에 접속해서 사용하는 경우
- crontab 등 Linux의 내장 기능을 활용하는 경우
- 데이터 전처리를 하기 위해 쉘 커맨드를 사용
- Docker를 사용하는 경우
- 수백대의 서버를 관리할 경우
- Jupyter Notebook의 Cell에서 앞에 !를 붙이면 쉘 커맨드가 사용
- 터미널에서 python3, jupyter notebook도 쉘 커맨드
- Test Code 실행
- 배포 파이프라인 실행(Github Action 등에서 실행)

### 기본 쉘 커맨드

- `man` : 쉘 커맨드의 매뉴얼 문서를 보고 싶은 경우
    - ex. `man python3`
    - 종료: `:q` 입력
- `mkdir` : 폴더 생성하기(Make Directory)
    - ex. `mkdir linux-test`
- `ls` : 현재 접근한 폴더의 파일 확인(List Segments)
    - `-a` : All의 약자로 전체 파일 출력
    - `-l` : Long의 약자로 퍼미션, 소유자, 만든 날짜, 용량까지 출력
    - `-h` : human-readable의 약자로 용량을 사람이 읽기 쉽도록 GB, MB 등 표현
- `pwd` : 현재 폴더 경로를 절대 경로로  보여줌(Print Working Directory)
- `cd` : 폴더 변경 및 이동(Change Directory)
- `echo` : 터미널에 텍스트 출력
- `cp` : 파일 또는 폴더 복사하기(Copy)
    - `-r` : 디렉토리를 복사할 때 디렉토리 안에 파일이 있으면 recursive(재귀적)으로 모두 복사
    - `-f` : 복사할 때 강제로 실행. Force의 약자
- `vi` : vim 편집기로 파일 생성
    - INSERT 모드에서만 수정이 가능하며 `i`를 눌러서 모드를 변경할 수 있다.
    - vi에는 3가지 mode가 있다.
        - Command Mode : 방향키를 통해 커서를 이둉(기본 모드)
        - Insert Mode : 파일을 수정할 수 있음
        - Last Line Mode : ESC + : 를 누르면 나오는 모드
- `bash` : 쉘 스크립트 실행
    - ex. `bash vi-test.sh`
- `sudo` : 관리자 권한으로 실행하고 싶은 경우 커맨드 잎에 sudo를 붙임
    - 최고 권한을 가진 슈퍼 유저로 프로그램을 실행
- `mv` : 파일, 폴더 이동하기(또는 이름 바꿀 때도 활용)
    - ex. `mv vi-test.sh vi-test3.sh`
- `cat` : 특정 파일 내용 출력(concatenate)
- `clear` : 터미널 창을 깨끗하게 해줌
- `history` : 최근에 입력한 쉘 커맨드
    - !를 이용해서 커맨드 활용 가능
- `find` : 파일 및 디렉토리를 검색할 때 사용
- `export` : 환경 변수 설정
    - 띄어쓰기 하면 안됨
    - ex. `export water="물"` &rarr; `export $water` 를 실행하면 '물' 이라고 출력
    - 터미널이 꺼지면 사라지게 되므로 저장을 하려면 `.bashrc`, `.zshrc`에 저장
- `alias` : 터미널에서 현재 별칭으로 설정된 것을 볼 수 있음
- `tree` : `apt-get install tree` 를 통해 설치할 수 있으며 폴더의 하위 구조를 계층적으로 표현
- `head` or `tail` : 파일의 앞/뒤 n행 출력
- `sort` : 행 단위 정렬
    - `-r` : 정렬을 내림차순으로 정렬(Default: 오름차순)
    - `-n` : Numeric Sort
- `uniq` : 중복된 행이 연속으로 있는 경우 중복 제거
    - `-c` : 중복 행의 개수 출력
- `grep` : 파일에 주어진 패턴 목록과 매칭되는 라인 검색으로 `grep 옵션 패턴 파일명` 으로 쓴다.
- `cut` : 파일에서 특정 필드 추출
    - `-f` : 잘라낼 필드 지정
    - `-d` : 필드를 구분하는 구분자. Defaultsms `\t`
- `awk` : 텍스트 처리 도구
- `ps` : 현재 실행되고 있는 프로세스 출력하기(Process Status)
    - `-e` : 모든 프로세스
    - `-f` : Full Format으로 자세히 보여줌
- `curl` : Data Transfer 커맨드(Client URL)로 Request를 테스트할 수 있는 명령어
    - 웹 서버를 작성한 후 요청이 제대로 실행되는지 확인할 수 있음
- `df` : 현재 사용 중인 디스크 용량 확인
    - `-h` : 사람이 읽기 쉬운 형태로 출력
- `ssh` : 안전하게(데이터가 모두 암호화) 원격으로 컴퓨터에 접속하고 명령을 실행할 수 있는 프로토콜
    - 사용하는 이유는 보완, 원격 접속, 터널링 기능 떄문이다.
    - `ssh -i /path/toprivate-key.pem/username@hostname(ip) -p 포트번호`
- `scp` : SSH을 이용해 네트워크로 연결된 호스트 간 파일을 주고 받는 명령어(Secure Copy)
    - `-r` : 재귀적으로 복사
    - `-P` : ssh 포트 지정
    - `-i` : SSH 설정을 활용해 실행
- `nohup` : 터미널 종료 후에도 계속 작업이 유지하도록 실행(백그라운드 실행)
    - nohup으로 실행될 파일은 Permission이 755여야 함
- `chmod` : 파일의 권한을 변경하는 경우 사용
    - 유닉스에서 파일이나 디렉토리의 시스템 모드를 변경함

> **터널링**
>
> 방화벽 등의 이슈로 직접적으로 접근이 제한될 경우, 안전한 터널을 만들고 터널을 통해 우회
>
> <img src="https://velog.velcdn.com/images/devhslee02/post/b6f1cb38-d701-4da8-8dad-fee38da104b4/image.png">
> 
> - ex. ssh -L 8080:localhost:30952 사용자명@SSH_서버
>   - 명령어를 실행시킨 컴퓨터의 8080 포트가 오픈
>   - 8080 포트로 들어오는 트래픽은 SSH 터널을 통해 SSH 서버의 30952 포트로 전탈
>   - 사용자의 컴퓨터 localhost:8080 에 접속하면 SSH 서버의 30952 포트에 연결
    

## 쉘 스크립트
-----

`.sh` 파일을 생성하고 그 안에 쉘 커맨드를 추가한다. 파이썬처럼 if, while, case 문이 존재하며 작성 시 `bah [Filename].sh` 로 실행 가능

- 인자: $1, $2 등의 특별한 변수를 사용하여 각각 첫 번쨰, 두 번째, ... N 번째 인수를 참조
    - $# : 변수를 사용하여 전달된 인수의 개수를 확인
    - $@ : 모든 변수에 접근
- 함수 : 파이썬 같이 함수 정의 및 사용 가능
    - 함수명 + 인자로 사용
    - 다른 변수 = $(함수 + 인자) 로 결과값을 다른 변수에 할당 가능
- 조건문 : ==, != 연사자를 사용
- #!/bin/bash : Shebang으로 이 스크립트를 Bash 쉘로 해석
- $(date + %s) : date를 %s(unix timestamp)로 변형하고 
    - START=$(date+%s) : 변수 저장