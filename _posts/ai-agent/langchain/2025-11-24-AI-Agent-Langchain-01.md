---
title: "Langchain Basic"
description: "랭체인 기본에 대한 간략한 내용을 정리한 포스트입니다."

categories: [AI Agent, LangChain]
tags: [LLM, LangChain]

permalink: /ai-agent/inflearn/langchain-01

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-11-24
last_modified_at: 2025-11-24
---

LLM(Large Language Model)를 사용해서 텍스트를 생성할 수 있다.

먼저, 관련된 패키지를 설치하고 API 키를 가져와야하며 이에 대한 코드는 아래와 같다.

```python
## 패키지 설치
%pip install -q langchain-ollama langchain-openai python-dotenv

## 환경 변수 설정 - .env 파일에서 API 키와 같은 환경 변수들을 로드
from dotenv import load_dotenv

load_dotenv()
```

텍스트를 생성하는 방법은 두 가지가 있다.

1. Ollama를 이용한 로컬 LLM 사용

    ```python
    from langchain_ollama import ChatOllama

    # Ollama를 이용한 로컬 LLM 설정
    llm = ChatOllama(model="llama3.2:1b")

    # 간단한 질문으로 테스트
    llm.invoke("What is the capital of France?")

    """
    AIMessage(content='The capital of France is Paris.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 7, 'prompt_tokens': 14, 'total_tokens': 21, ... )
    """
    ```

2. Open AI GPT(API 키 이용) 모델 사용

    ```python
    from langchain_openai import ChatOpenAI

    # OpenAI GPT 모델 설정
    llm = ChatOpenAI(model="gpt-4o-mini")

    llm.invoke("What is the capital of France?")  # OPENAI_API_KEY

    """
    AIMessage(content='The capital of France is Paris.', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 7, 'prompt_tokens': 14, 'total_tokens': 21, ... )
    """
    ```

여기서 LangChain을 사용하는 이유는 LLM 기반 애플리케이션을 더 쉽게, 효율적으로 개발할 수 있다. 이는 더욱 복잡한 태스크를 수행해야할 때 더욱 효율적으로 개발이 가능하다.

## PromptTemplate
-------

Prompt는 LLM을 호출할 때 사용하는 명령어이다.

- ex. "What is the capital of France?"

프롬프트를 구성할 때, `PromptTemplate`를 사용하면 변수가 포함된 템플릿을 만들 수 있다.

> **PromptTemplate 구성**
>
> - input에 정해진 type이 존재함
>    - `PromptValue` : Prompt Template를 통해 만들 수 있음
>    - `BaseMessages`: HumanMessage가 상속받는 것으로 이런 클래스가 대표적으로 4가지가 있음
>        - System(LLM APP의 목적, Persona)
>        - HumanMessage(사용자 message)
>        - AIMessage(LLM)
>        - ToolMessage(도구, agent 만들 때 사용하며 invoke할 때 생성) 

```python
from langchain_ollama import ChatOllama
from langhchain_core.prompts import ChatPromptTemplate

prompt_template = PromptTemplate(
    template="What is the capital of {country}?",
    input_variables=["country"],
)

prompt = prompt_template.invoke({"country": "France"})

llm = ChatOllama(model="llama3.2:1b")

llm.invoke(prompt)
```

### 메세지 기반 프롬프트

시스템 메시지, 사용자 메시지, AI 메시지를 조합하여 대화 형식의 프롬프트를 만들 수 있다.

```python
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# 대화 형식의 메시지 리스트 생성
message_list = [
    SystemMessage(content="You are a helpful assistant!")
    HumanMessage(content="What is the capital of France?"),
    AIMessage(content="The capital of France is Paris"),
    HumanMessage(content="What is the capital of Germany?")
]
llm.invoke(message_list)
```

위의 코드에서는 `AIMessage`를 넣어 AI가 답변 형식에 대한 예시를 추가하였으며, 이 방식은 마치 LLM과 대화를 나눈 것처럼 속이는 것으로 개발자가 원하는 방식대로 답하도록 만든다. 

이렇게 예시(shot)을 추가하면 AI 수행 능력이 굉장히 향상된다.

### ChatPromptTemplate

또 다른 방법으로는 `ChatPromptTemplate` 을 사용한 프롬프트 작성 방식이 있다.

이 방식은 나중에 나올 `LCEL`에 적용할 수 있어 확장성 측면에서 훨씬 유리하다는 장점이 있다.

```python
from langchain_core.prompts import ChatPromptTemplate

chat_prompt_template = ChatPromptTemplate.from_messages([
    ('system', "You are a helpful assistant!"),     # SystemMessage
    ('human', "What is the capital of {country}?")  # HumanMessage
])

chat_prompt = chat_prompt_template.invoke({"country": "France"})

# 생성된 메시지 확인
chat_prompt.messages

# LLM에 프롬프트 전달
llm.invoke(chat_prompt)    # AIMessage 출력
```

## Output Parser
------

LangChain을 사용하면 LLM의 출력 형식을 제어할 수 있다.

### 문자열 출력 파서

대표적으로 `StrOutputParser` 가 있다. 이 함수를 사용하면 LLM의 출력을 단순 문자열로 변환할 수 있다.

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 명시적인 지시사항이 포함된 프롬프트 템플릿 정의
prompt_template = PromptTemplate(
    template="What is the capital of {country}? Return the name of city only",
    input_variables=["country"],
)

# 템플릿에 변수 값을 대입
prompt = prompt_template.invoke({"country": "France"})

# LLM에 프롬프트 전달
ai_message = llm.invoke(prompt)

# 문자열 출력 파서를 사용하여 응답을 단순 문자열로 변환
output_parser = StrOutputParser()

answer = output_parser.invoke(llm.invoke(prompt_template.invoke({'country': 'France'})))

# AI 메시지의 content 속성이 타입 확인
type(ai_message.content)    # str

# 파싱된 응답 확인
print(answer)   # Paris
```

StrOutputParser는 string 타입으로 반환해주기 데이터 전송할 때 매우 편리하다.

### 구조화된 출력

Pydantic 모델을 사용하여 LLM의 출력을 구조화된 형식으로 받을 수 있다.

```python
from pydantic import BaseModel, Field

# 국가 정보를 담을 Pydantic 모델 정의
class CountryDetail(BaseModel):
    capital: str = Field(description="The capital of the country")
    population: int = Field(description="The population of the country")
    language: str = Field(description="The language of the country")
    currency: str = Field(description="The currency of the country")

# LLM에 구조화된 출력 형식 지정
structued_llm = llm.with_structured_output(CountryDetail)
```

```python
from langchain_core.output_parsers import JsonOutputParser

# JSON 형식으로 응답을 요청하는 프롬프트 템플릿
country_detail_prompt = PromptTemplate(
    template="""Give following information about {country}:
    - Capital
    - Population
    - Language
    - Currency

    return it in JSON format. and return the JSON dictionry only 
    """,
    input_variables=["country"],
)

country_detail_prompt.invoke({"country": "France"})

# 구조화된 LLM으로 응답 받기
json_ai_message = structued_llm.invoke(country_detail_prompt.invoke("country": "France"))

# 구조화된 응답 확인
(json_ai_message)

# Pydantic 모델의 특정 필드 접근
json_ai_message.model_dump()['capital']    # 'Paris'
```

문자열처럼 `JsonOutputParser` 함수를 통해 Json을 받을 수 있지만 위의 방식이 훨씬 안전하다.

## Runnable
------

`LangChain Expression Language(LCEL)`를 사용하여 여러 컴포넌트를 연결하고 복잡한 체인을 구성할 수 있다. 

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2:1b")

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt_template = PromptTemplate(
    template="What is the capital of {country}? Return the name of city only",
    input_variables=["country"],
)

# 문자열 출력 파서 설정
output_parser = StrOutputParser()

# 프롬프트 템플릿 -> LLM -> 출력 파서를 연결하는 체인 생성
capital_chain = prompt_template | llm | output_parser

# 생성된 체인 실행
capital_chain.invoke({"country": "France"})
```

이렇게 chain을 구현할 수 있는 이유는 바로 `llm`, `StrOutputParser`, `prompt_template` 같은 모든 주요 구성 요소들이 **LangChain Expression Language (LCEL)의 기본 단위인 Runnable 인터페이스**를 구현하기 때문이다.

여러 단계의 추론이 필요한 더 복잡한 체인을 구성해보자.

```python
# 국가를 추측하는 프롬프트 템플릿 정의
country_prompt = PromptTemplate(
    template="""Guess the name of the country in the {continent} based on the following information: {information}
    Return the name of the country only
    """,
    input_variables=["information", "continent"],
)

# 국가 추측 체인 생성
country_chain = country_prompt | llm | output_parser

from langchain_core.runnables import RunnablePassthrough

# RunnablePassthrough를 사용하여 입력을 다음 단계로 전달하는 복합 체인 구성
final_chain = {"information": RunnablePassthrough(), "continent": RunnablePassthrough()} | {"country": country_chain} | capital_chain

# 복합 체인 실행
# 정보와 대륙을 입력하면, 해당 국가를 추측하고 그 국가의 수도를 반환
final_chain.invoke({"information": "This country is very famous for its wine in Europe", "continent": "Europe"})    # Madrid
```

> **💡 프롬프트를 쪼개서 작성해야하는 이유**<br>
> 길게 작성하게 되면 하라고 훈련받았는데 하지 말라는 조건을 넣게 된다. 이렇게 출력하지 말라는 조건을 넣으면 잘 작동을 안하기 때문에 safety 프롬프트를 넣어서 Yes/no 를 거치고 이 결과가 통과(Yes)면 다음 logic을 거치도록 해야한다.
>
> 즉, 하나의 프롬프트가 아니라 여러 프롬프트를 거쳐서 원하는 결과를 얻는 것이 작성하는 것이 바람직하다.