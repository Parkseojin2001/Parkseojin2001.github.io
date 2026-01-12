---
title: "[BoostCamp AI Tech / Product Serving] Fast API(1)"
description: "FastAPI를 사용하여 간단한 REST API를 구축하는 방법을 정리한 포스트입니다."

categories: [NAVER BoostCamp AI Tech, AI Production]
tags: [model serving, Fast API, REST API]

permalink: /naver-boostcamp/model-serving/04

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2026-01-12
last_modified_at: 2026-01-12
---

`FastAPI`는 대표적인 Python Web Framework로 특징은 다음과 같다.

- High Performance : Node.js, go와 대등한 성능
- Easy : Flask와 비슷한 구조로 Microservice에 적합
- Productivity : Swagger 자동 생성, Pydantic을 이용한 Serialization
    - Swagger가 API 명세서를 자동 생성
    - Pydantic을 통해 config 관리 데이터에 대한 validation을 확인

Fast API의 간단한 프로젝트 구조는 다음과 같다.

```markdown
├── app     
│   ├── __main__.py
│   ├── main.py
│   └── model.py
```

- 프로젝트의 코드가 들어갈 모듈 설정(app). 대안 : 프로젝트 이름, src 등
    - `__main__.py`는 간단하게 애플리케이션을 실행할 수 있는 entrypoint 역할 (참고 공식 문서)
    - entrypoint : 프로그래밍 언어에서 최상위 코드가 실행되는 시작점 또는 프로그램 진입점
- `main.py(or app.py)` : FastAPI의 애플리케이션과 Router 설정
- model.py : ML model에 대한 클래스와 함수 정의

FastAPI를 살펴보기 전 HTTP Method의 차이에 대해 잠깐 살펴보자

|처리 방식|GET|POST|
|:-----:|:---:|:---:|
|URL에 데이터 노출 여부| O | X |
|대표적인 상황|웹페이지 접근 시|웹페이지에서 FORM 제출 시|
|URL 예시|localhost:8080/login?id=kyle|localhost:8080/login|
|데이터의 위치|Header|Body|

## FastAPI - Hello World
-------

가장 기초적인 웹 서버를 생성하는 코드는 아래와 같다.

```python
from fastapi import FastAPI

# FastAPI 객체 생성
app = FastAPI()

# "/"로 접근하면 return을 보여줌
@app.get("/")
def read_root():
    return {"Hello": "World"}
```

- 루트(“/”)로 접근하면 Hello World가 출력되는 웹 서버
- 터미널(혹은 CLI)에서 uvicorn 01_simple_webserver:app --reload 명령
    -  `uvicorn` :ASGI( Asynchronous Server Gateway Interface). 비동기 코드를 처리할 수 있는 Python 웹 서버, 프레임워크 간의 표준 인터페이스

만약 uvicorn을 작성하기 싫다면 코드 내 `uvicorn.run을` 추가하면 된다.

```python
from fastapi import FastAPI
import uvicorn

# FastAPI 객체 생성
app = FastAPI()

# "/"로 접근하면 return을 보여줌
@app.get("/")
def read_root():
    return {"Hello": "World"}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000) 
```

- FastAPI는 데코레이터로 GET, POST를 표시
    - `@app.get`, `@app.post`
- localhost:8000/docs로 이동하면 Swagger 문서를 확인할 수 있다.

    <img src="https://i.sstatic.net/TYB97.png">

- localhost:8000/redoc로 이동하면 Redoc을 확인할 수 있다.

    <img src="https://i.sstatic.net/HkWkn.png">

## Swagger

그러면 왜 Swagger 기능을 필요할까?

그 이유는 만든 API를 클라이언트에서 호출하는 경우(협업) **Request할 때 어떤 파라미터를 넣어서 요청하면 되는지** 알 수 있기 때문이다.

- Swagger Usecase
    - REST API 설계 및 문서화할 때 사용
    - 다른 개발팀과 협업하는 경우
    - 구축된 프로젝트를 유지보수하는 경우

- Swaager 기능
    - API 디자인
    - API 빌드
    - API 문서화
    - API 테스팅

## URL Paramters

URL Paramters는 웹에서 GET Method를 사용해 데이터를 전송할 수 있는 파라미터이다.

이 때 전달하는 방법으로 2가지가 있다.

- Path Parameter 방식: /users/402 
    - 서버에 402라는 값을 전달하고 변수로 사용
- Query Parameter 방식: /users?id=402
    - Query String
    - API 뒤에 입력 데이터를 함께 제공하는 방식으로 사용
    - Query String은 Key, Value의 쌍으로 이루어지며 &로 연결해 여러 데이터를 넘길 수 있음

그러면 어떤 상황에서 어떤 방식을 사용하면 될까?

- Resource를 식별해야 하는 경우 : Path Parameter가 더 적합
- 정렬, 필터링을 해야 하는 경우 : Query Parameter가 더 적합

### Path Paramter

`Path Paramter`는 GET Method 정보를 READ하거나 유저 정보에 접근하는 API 만들때 사용한다.

```python
from fastapi import FastAPI
import uvicorn

# FastAPI 객체 생성
app = FastAPI()


@app.get("/users/{user_id}")
def get_user(user_id):
    return {"user_id": user_id}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000) 
```

### Query Paramter

함수의 파라미터가 Query Parameter로 사용된다.

```python
from fastapi import FastAPI
import uvicorn

# FastAPI 객체 생성
app = FastAPI()

fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]


@app.get("/items/")
def read_item(skip: int = 0, limit: int = 10):
    return fake_items_db[skip: skip + limit]

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000) 
```

Query Paramter를 적용한 URL 뒤에 ?를 붙이고 Key, Value 형태로 연결하면 된다.
- ex. `localhost:8000/items/?skip=20`

### Optional Paramter

`Optional Paramter`은 특정 파라미터는 Optional(선택적)으로 하고 싶은 경우에 사용한다.

- typing 모듈의 Optional을 사용
- Optional을 사용해 이 파라미터는 Optional임을 명시(기본 값은 None)

```python
from typing import Optional
from fastapi import FastAPI
import uvicorn

# FastAPI 객체 생성
app = FastAPI()

@app.get("/items/{item_id}")
def read_item(item_id: str, q: Optional[str] = None):
    if q:
        return {"item_id": item_id, "q": q}
    return {"item_id": item_id}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000) 
```


### Request Body

클라이언트에서 API에 데이터를 보낼 때, Request Body(=Payload)를 사용한다.

- 클라이언트 &rarr; API : Request Body
- API의 Response &rarr; 클라이언트 : Response Body

Request Body에 데이터가 항상 포함되어야 하는 것은 아니다.

- **Request Body에 데이터를 보내고 싶다면 POST Method를 사용**
    - GET Method는 URL, Request Header로 데이터 전달