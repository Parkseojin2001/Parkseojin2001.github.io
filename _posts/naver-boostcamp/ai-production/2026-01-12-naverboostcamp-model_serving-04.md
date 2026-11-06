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
    -  `uvicorn` : ASGI( Asynchronous Server Gateway Interface). 비동기 코드를 처리할 수 있는 Python 웹 서버, 프레임워크 간의 표준 인터페이스

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

    <img src="https://fastapi.tiangolo.com/img/index/index-03-swagger-02.png">

- localhost:8000/redoc로 이동하면 Redoc을 확인할 수 있다.

    <img src="https://fastapi.tiangolo.com/img/index/index-06-redoc-02.png">

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


Request Body에 데이터가 항상 포함되어야 하는 것은 아니다.

하지만, **Request Body에 데이터를 보내고 싶다면 POST Method를 사용**하며 이때 GET Method는 URL, Request Header로 데이터 전달한다.

- POST Method는 Request Body에 데이터를 넣어 보냄
- Body의 데이터를 설명하는 Content-Type는 Header 필드가 존재하고, 어떤 데이터 타입인지 명시해야 함
    - 대표적인 컨텐츠 타입
    - application/json : Body가 JSON 형태 (FastAPI는 기본적으로 이걸 사용)
    - application/x-www-form-urlencoded : BODY에 Key, Value 사용. & 구분자 사용
    - text/plain : 단순 txt 파일
    - multipart/form-data : 데이터를 바이너리 데이터로 전송

아래의 코드는 POST 요청으로 item을 생성하는 예제이다.

```python
from typing import Optional
from fastapi import FastAPI
import uvicorn

from pydantic import BaseModel

# pydantic.BaseModel로 Request Body 데이터 정의
class Item(BaseModel):
    name: str   # 필수
    description: Optional[str] = None
    price: float    # 필수
    tax: Optional[float] = None

app = FastAPI()
 
@app.post("/items/")
def create_item(item: Item):  # Type Hinting에 위에서 생성한 Class 주입
    return item

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Swagger 문서의 Schemas를 통해 pydantic으로 정의한 내용을 볼 수 있으며 마찬가지로 POST의 'Try it out'버튼을 클릭해도 확인할 수 있다.

<img src="../assets/img/post/naver-boostcamp/fastapi-response_body.png">

### Response Body

API가 클라이언트에게 데이터를 보낼 때, Response Body를 사용한다.

이때, Decorator의 `response_model` 인자로 주입 가능하다.

```python
from typing import Optional
from fastapi import FastAPI
import uvicorn

from pydantic import BaseModel

# request
class ItemIn(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

# response
class ItemOut(BaseModel):
    name: str
    price: float
    tax: Optional[float] = None

app = FastAPI()

@app.post("/items/", response_model=ItemOut)
def create_item(item: ItemIn):
    return item

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

`ItemOut` 클래스는 Request로 Input data가 들어오면 Response를 할 때 보낼 Output data를 해당 정의에 맞게 변형해주는 역할을 한다.

이때, 데이터 Validation과 Response에 대한 Json Schema 추가, 자동으로 문서화를 수행한다.


### Form

Form(입력) 형태로 데이터를 받고 싶은 경우에는 python-multipart를 설치해야한다. 또한, Jinja2 설치로 프론트엔드도 간단히 만들 수 있다.

대표적인 Form 형태로는 로그인 기능이 있다.

Form 클래스를 사용하면 Request의 Form Data에서 값을 가져온다.

이때, 웹 브라우저에서 URL 입력하거나 링크를 클릭할 때 기본적으로 “GET” 요청이 발생한다.

GET은 서버에 정보를 요청할 때 사용되고, 데이터 검색하거나 특정 페이지를 요청할 때 사용하고, POST는 사용자가 폼을 제출할 때 주로 사용한다.

로그인 폼에 사용자 이름, 비밀번호를 입력하고 제출 버튼을 누를 때 이 정보는 POST 요청을 통해 서버로 전송. POST는 데이터 생성, 업데이트할 때 사용된다.

- 웹사이트에 접근 : GET 요청
- 사용자가 데이터를 제출하는 행위 : POST 요청

```python
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates

import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="./")

@app.get("/login/")
def get_login_form(request: Request):
    return templates.TemplateResponse("login_form.html", context={"request": request})

@app.post("/login/")
# ellipsis(...)는 필수 요소를 의미
def login(username: str = Form(...), password: str = Form(...)): 
    return {"username": username}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### File

File을 업로드하고 싶은 경우 python-multipart를 설치해야 한다.

```python
from typing import List

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

import uvicorn

app = FastAPI()

# 파일을 Bytes로 표현하고, 여러 파일은 LIST에 설정
@app.post("/files/")
def create_files(files: List[bytes] = File(...)):
    return {"file_sizes": [len(file) for file in files]}

@app.post("/uploadfiles/")
def create_upload_files(files: List[UploadFile] = File(...)):
    return {"filenames": [file.filename for file in files]}

@app.get("/")
def main():
    content = """
    <body>
        <form action="/files/" enctype="multipart/form-data" method="post">
            <input name="files" type="file" multiple>
            <input type="submit">
        </form>
        <form action="/uploadfiles/" enctype="multipart/form-data" method="post">
            <input name="files" type="file" multiple>
            <input type="submit">
        </form>
    </body>
    """
    return HTMLResponse(content=content)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

