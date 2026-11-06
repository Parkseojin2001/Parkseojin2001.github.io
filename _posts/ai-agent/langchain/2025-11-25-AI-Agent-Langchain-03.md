---
title: "LangChain을 활용한 RAG 구성"
description: "LangChain을 활용한 Retrieval Augmented Generation(RAG) 구성하는 과정과 코드를 정리한 포스트입니다."

categories: [AI Agent, LangChain]
tags: [LLM, LangChain, RAG]

permalink: /ai-agent/inflearn/langchain-03

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2026-01-04
last_modified_at: 2026-01-04
---

> 🔎 **RAG 파이프라인**
> 
> 1. 문서의 내용을 읽는다.
> 2. 문서를 쪼갠다.
>   - 토큰 수 초과로 문서(input) 이 길면 답변 생성이 올래걸리기 때문에 chunk 단위로 저장 필요
> 3. Embedding 모델을 이용해 문서를 vector로 변환해 데이터베이스에 저장
> 4. 질문이 있을 때, Vector Database에 유사도를 검색
> 5. 유사도 검색으로 가져온 문서와 질문을 LLM에 전달

## Pinecone 를 활용한 RAG 구성
----

`Pinecone`은 대규모 데이터를 다루기에 최적화된 클라우드 기반(SaaS) 완전 관리형 벡터 데이터베이스이며 인프라 설정이나 확장에 신경 쓸 필요 없이 API 연결만으로 수억 개의 벡터 데이터에서 초고속 유사도 검색을 수행할 수 있다.

### Knowledge Base 구성을 위한 데이터 생성

- `RecursiveCharacterTextSplitter`를 활용한 데이터 chunking
    - split 된 데이터 chunk를 Large Language Model(LLM)에게 전달하면 토큰 절약 가능
    - 비용 감소와 답변 생성시간 감소의 효과
    - LangChain에서 다양한 TextSplitter들을 제공
- `chunk_size` 는 split 된 chunk의 최대 크기
- `chunk_overlap`은 앞 뒤로 나뉘어진 chunk들이 얼마나 겹쳐도 되는지 지정

```python
import os

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
)

loader = Docx2txtLoader('./tax.docx')
document_list = loader.load_and_split(text_splitter=text_splitter)

# 환경변수 불러옴(.env)
load_dotenv()

# OpenAI에서 제공하는 Embedding Model을 활용해서 `chunk`를 vector화
embedding = OpenAIEmbeddings(model='text-embedding-3-large')

index_name = 'tax-index'
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

# 데이터를 추가할 때 사용
database = PineconeVectorStore.from_documents(document_list, embedding, index_name=index_name)
# 데이터를 추가한 이후에 사용
database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
```

### 답변 생성을 위한 Retrieval

- RetrievalQA에 전달하기 위해 retriever 생성
- `search_kwargs` 의 `k` 값을 변경해서 가져올 문서의 갯수를 지정할 수 있음
- `.invoke()` 를 호출해서 어떤 문서를 가져오는지 확인 가능

```python
query = '연봉 5천만원인 직장인의 소득세는 얼마인가요?'

# 'k' 값을 조절해서 얼마나 많은 데이터를 불러올지 결정
retriever = database.as_retriever(search_kwargs={'k': 4})
retriever.invoke(query)
```

### Augmentation을 위한 Prompt 활용

- Retrieval된 데이터는 LangChain에서 제공하는 프롬프트(`"rlm/rag-prompt"`) 사용

```python
from langchain_openai import ChatOpenAI
from langchain import hub

llm = ChatOpenAI(model='gpt-4o')
prompt = hub.pull("rlm/rag-prompt")
```

### 답변 생성

- `RetrievalQA` 를 통해 LLM에 전달

```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=database.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

ai_message = qa_chain.invoke({"query": query})
print(ai_message)

"""
{'query': '연봉 5천만원인 직장인의 소득세는 얼마인가요?',
 'result': '소득세는 여러 요소에 따라 달라질 수 있어서 정확한 계산이 필요합니다. 대략적으로 5천만원 연봉의 소득세는 약 5백만원 정도입니다. 하지만 정확한 금액은 국세청의 소득세 계산기를 통해 확인하는 것이 좋습니다.'}
"""
```

- `RetrievalQA` 는 `create_retrieval_chain`으로 대체
    - 실제 ChatBot 구현 시 `create_retrieval_chain` 으로 변경하는 과정을 볼 수 있음

```python
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
```

### Retrieval 효율 개선을 위한 키워드 사전 활용

- Knowledge Base에서 사용되는 keyword를 활용하여 사용자 질문 수정
- LangChain Expression Language (LCEL)을 활용한 Chain 연계

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

dictionary = ["사람을 나타내는 표현 -> 거주자"]

prompt = ChatPromptTemplate.from_template(f"""
    사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
    만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
    그런 경우에는 질문만 리턴해주세요
    사전: {dictionary}
    
    질문: {{question}}
""")

dictionary_chain = prompt | llm | StrOutputParser()
tax_chain = {"query": dictionary_chain} | qa_chain

new_question = dictionary_chain.invoke({"question": query})
ai_response = tax_chain.invoke({"question": query})
```

## 그 외의 Vector Database 
-----

- `Chroma`
    - 오픈소스 기반의 AI 전용 벡터 데이터베이스로, 임베딩된 벡터 데이터와 메타데이터를 간편하게 저장하고 검색할 수 있게 도와줌
    - 별도의 복잡한 설정 없이 Python 코드 몇 줄로 로컬 환경에서 즉시 실행 가능

- `FAISS`
    - 벡터 검색 라이브러리로 DB라기보다는 엔진에 가까워 속도가 압도적으로 빠르고 무료임
    - 서버 구축 없이 내 컴퓨터 메모리(RAM) 상에서 수만~수십만 개의 벡터를 초고속으로 검색할 때 씀

- `Milvus / Qdrant`
    - 오픈소스 기반의 고성능 전용 벡터 DB 로 직접 서버에 설치(Self-hosting)할 수 있어 데이터 보안이 중요한 기업용 프로젝트에 많이 쓰임

- `pgvector`
    - 기존에 널리 쓰이는 PostgreSQL 데이터베이스에 벡터 저장 기능을 추가한 것으로 새로운 DB를 배포할 필요 없이 기존 DB 하나로 일반 데이터와 벡터 데이터를 동시에 관리할 수 있어 운영이 매우 단순함