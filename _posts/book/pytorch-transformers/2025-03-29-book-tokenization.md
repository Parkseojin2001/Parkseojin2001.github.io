---
title: "[파이토치 트랜스포머를 활용한 자연어 처리와 컴퓨터 비전 심층학습] 토큰화"
description: "단어 및 글자 토큰화 / 형태소 토큰화 / 하위 단어 토큰화"

categories: [Book, deep-learning-with-pytorch-transformers]
tags: [NLP]

permalink: /pytorch-book/nlp/tokenization/

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-03-29
last_modified_at: 2025-03-30
---

**자연어 처리(NLP)**는 컴퓨터가 인간의 언어를 이해하고 해석 및 생성하기 위한 기술을 의미한다.

자연어 처리는 인공지능의 하위 분야 중 하나로 컴퓨터가 인간과 유사한 방식으로 인간의 언어를 이해하고 처리하는 것이 주요 목표 중 하나다. 인간 언어의 구조, 의미, 맥락을 분석하고 이해할 수 있는 알고리즘과 모델을 개발한다. 이런 모델을 개발하기 위해서는 해결해야할 문제가 있다.

- 모호성(Ambiguity): 인간의 언어는 맥락에 따라 여러 의미를 갖게 되어 모호한 경우가 많다. 알고리즘이 이런 다양한 의미를 이해하고 명확하게 구분할 수 있어야 한다.
- 가변성(Variability): 사투리, 강세, 신조어, 작문 스타일로 인해 매우 가변적이므로 이를 처리할 수 있어야 하며 사용 중인 언어를 이해할 수 있어야한다.
- 구조(Structure): 구문을 파악하여 의미를 해석해야하므로 알고리즘도 문장의 구조와 문법적 요소를 이해하여 의미를 추론하거나 분석할 수 있어야 한다.

이를 해결하는 모델을 만들려면 **말뭉치(Corpus)**를 일정한 단위인 **토큰(Token)**으로 나눠야 한다.
- 말뭉치: 자연어 모델을 훈련하고 평가하는 데 사용되는 대규모의 자연어
- 토큰: 개별 단어나 문장 부호와 같은 텍스트를 의미하며 말뭉치보다 더 작은 단위

**토큰화(Tokenization)**는 컴퓨터가 자연어를 이해할 수 있게 토큰으로 나누는 과정이다.
- 자연어 처리 과정에서 중요한 단계
- 토큰화를 위해 **토크나이저(Tokenizer)**를 사용
    - 토크나이저는 텍스트 문자열을 토큰으로 나누는 알고리즘 또는 소프트웨어를 의미

토큰을 나누는 기준은 구축하려는 시스템이나 주어진 상황에 따라 다르며 어떻게 나누었느냐에 따라 시스템의 성능이나 처리 결과가 크게 달라지기도 한다.
- 공백 분할: 텍스트를 공백 단위로 분리해 개별 단어로 토큰화
- 정규 표현식 적용: 정규 표현식으로 특정 패턴을 식별해 텍스트를 분할
- 어휘 사전(Vocabulary)적용: 사전에 정의된 단어 집합을 토큰으로 사용
    - 미리 정의된 단어를 활용하므로 없는 단어나 토큰이 존재할 수 있다. 이를 **OOV(Out of Vocab)**라고 한다.
    - OOV 문제를 해결하기 위해 더 큰 어휘 사전을 구축한다면 학습 비용이 증대하고 **차원의 저주**에 빠질 수 있는 단점이 있다.
- 머신러닝 활용: 데이터세트를 기반으로 토큰화하는 방법을 학습한 머신러닝을 적용


## 단어 및 글자 토큰화
---------

토큰화는 자연어 처리에서 매우 중요한 전처리 과정으로, 텍스트 데이터를 구조적으로 분해하여 개별 토큰으로 나누는 작업을 의미한다. 이를 통해 단어나 문장의 빈도수, 출현 패턴 등을 파악할 수 있다.

또한 작은 단위로 분해된 텍스트 데이터는 컴퓨터가 이해하고 처리하기 용이해 기계 번역, 문서 분류, 감성 분석 등 다양한 자연어 처리 작업에 활용할 수 있다.

입력된 텍스트 데이터를 단어(Word)나 글자(Character) 단위로 나누는 기법으로는 **단어 토큰화**와 **글자 토큰화**가 있다.

### 단어 토큰화

**단어 토큰화(Word Tokenization)**는 자연어 처리 분야에서 핵심적인 전처리 작업 중 하나로 텍스트 데이터를 의미있는 단위인 단어로 분리하는 작업이다.
- 띄어쓰기, 문장 부호, 대소문자 등의 특정 구분자를 활용해 토큰화를 수행
- 주로 품사 태깅, 개체명 인식, 기계번역 등의 작업에서 사용되며 가장 일반적인 토큰화 방법

```python
# 단어 토큰화
review = "현실과 구분 불가능한 cg. 시각적 즐거움은 최고! 더불어 ost는 더더욱 최고!!"
tokenized = review.split()
print(tokenized)
# ['현실과', '구분', '불가능한', 'cg.', '시각적', '즐거움은', '최고!', '더불어', 'ost는', '더더욱', '최고!!']
```

문자열 데이터 형태는 `split()` 메서드를 이용하여 토큰화한다. 
- 구분자를 통해 문자열을 리스트 데이터로 나눔
- 구분자를 입력하지 않으면 **공백(Whitespace)**가 기준

단어 토큰화는 한국어 접사, 문장 부호, 오타 혹은 띄어쓰기 오류 등에 취약하다.
ex) 'cg.', 'cg'를 다른 토큰으로 인식

### 글자 토큰화

**글자 토큰화(Character Tokenization)**는 띄어쓰기뿐만 아니라 글자 단위로 문장을 나누는 방식이다.
- 비교적 작은 단어 사전을 구축
- 작은 단어 사전을 사용하면 학습 시 컴퓨터 자원을 절약
- 전체 말뭉치를 학습할 때 각 단어를 더 자주 학습이 가능
- 언어 모델링과 같은 시퀀스 예측 작업에서 활용
    - 다음에 올 문자를 예측하는 언어 모델링

```python
# 글자 토큰화
review = "현실과 구분 불가능한 cg. 시각적 즐거움은 최고! 더불어 ost는 더더욱 최고!!"
tokenized = list(reivew)
print(tokenized)
# ['현', '실', '과', ' ', '구', '분', ' ', '불', '가', '능', '한', ' ', 'c', 'g', '.', ' ', '시', '각', '적', ' ', '즐', '거', '움', '은', ' ', '최', '고', '!', ' ', '더', '불', '어', ' ', 'o', 's', 't', '는', ' ', '더', '더', '욱', ' ', '최', '고', '!', '!']
```

글자 토큰화는 `list()`를 이용해 쉽게 수행할 수 있다. 
- 단어 토큰화와 다르게 공백도 토큰으로 나눔

영어의 경우는 각 알파벳으로 토큰화를 하지만 한글은 하나의 글자가 여러 자음과 모음의 조합으로 이루어져 있어 자소 단위로 나눠서 자소 단위 토큰화를 수행한다.
- **자모(jamo)** 라이브러리 활용
    - 한글 문자 및 자모 작업을 위한 한글 음절 분해 및 합성 라이브러리
    - 텍스트를 자소 단위로 분해해 토큰화를 수행

**컴퓨터가 한글을 인코딩하는 방식**<br>
- 완성형: 조합된 글자 자체에 값을 부여해 인코딩하는 방식
    ```python
    # 자모 변환 함수 - 입력된 한글을 조합형 한글로 변환
    retval = jamo.h2j(
        hangul_string
    )
    ```

- 조합형: 글자를 자모 단위로 나눠 인코딩한 뒤 이를 조합해 한글을 표현
    - 초성, 중성, 종성으로 분리

    ```python
    # 한글 호환성 자모 변환 함수 - 조합성 한글 문자열을 자소 단위로 나눠 반환
    retval = jamo.j2hcj(
        jamo
    )
    ```

자소 단위로 분해하여 토큰화를 수행하면 다음과 같이 분리된다.
    
```python
# 자소 단위 토큰화
from jamo import h2j, j2hcj

review = "현실과 구분 불가능한 cg. 시각적 즐거움은 최고! 더불어 ost는 더더욱 최고!!"
decomposed = j2hcj(h2j(review))
tokenized = list(decomposed)
print(tokenized)
# ['ㅎ', 'ㅕ', 'ㄴ', 'ㅅ', 'ㅣ', 'ㄹ', 'ㄱ', 'ㅘ', ' ', 'ㄱ', 'ㅜ', 'ㅂ', 'ㅜ', 'ㄴ', ' ', 'ㅂ', 'ㅜ', 'ㄹ', 'ㄱ', 'ㅏ', 'ㄴ', 'ㅡ', 'ㅇ', 'ㅎ', 'ㅏ', 'ㄴ', ' ', 'c', 'g', '.', ' ', 'ㅅ', 'ㅣ', 'ㄱ', 'ㅏ', 'ㄱ', 'ㅈ', 'ㅓ', 'ㄱ', ' ', 'ㅈ', 'ㅡ', 'ㄹ', 'ㄱ', 'ㅓ', 'ㅇ', 'ㅜ', 'ㅁ', 'ㅇ', 'ㅡ', 'ㄴ', ' ', 'ㅊ', 'ㅚ', 'ㄱ', 'ㅗ', '!', ' ', 'ㄷ', 'ㅓ', 'ㅂ', 'ㅜ', 'ㄹ', 'ㅇ', 'ㅓ', ' ', 'o', 's', 't', 'ㄴ', 'ㅡ', 'ㄴ', ' ', 'ㄷ', 'ㅓ', 'ㄷ', 'ㅓ', 'ㅇ', 'ㅜ', 'ㄱ', ' ', 'ㅊ', 'ㅚ', 'ㄱ', 'ㅗ', '!', '!']
```

**장점**<br>
- 단어 단위로 토큰화하는 것에 비해 비교적 적은 크기의 단어 사전 구축이 가능
- 단어 토큰화의 단점을 보완
    - 접사와 문장 부호의 의미 학습이 가능
- 작은 크기의 단어 사전으로도 OOV를 줄일 수 있음

**단점**<br>
- 개별 토큰은 아무런 의미가 없으므로 자연어 모델이 각 토큰의 의미를 조합해 결과를 도출해야 한다.
- 토큰 조합 방식을 사용해 문장 생성이나 **개체명 인식**등을 구현할 경우, 다의어나 동음이의어가 많은 도메인에서 구별하는 것이 어려울 수 있다.
- 모델 입력 **시퀀스(sequence)**의 길이가 길어질수록 연산량이 증가


## 형태소 토큰화
---------

**형태소 토큰화(Morpheme Tokenization)**란 텍스트를 형태소 단위로 나누는 토큰화 방법으로 언어의 문법과 구조를 고려해 단어를 분리하고 이를 의미있는 단위로 분류하는 작업이다.

- 한국어와 같이 교착어인 언어에서 중요하게 수행된다.
    - 각 단어가 띄어쓰기로 구분되지 않고 어근에 다양한 접사와 조사가 조합되어 하나의 낱말을 이루기 떄문이다.

햔국어에는 두 가지의 형태소가 있다.
- 자립 형태소: 단어를 이루는 기본 단위로 스스로 의미를 가지고 있음
    - 명사, 동사, 형용사
- 의존 형태소: 자립 형태소와 함께 조합되며 의미를 가지고 있지 않음
    - 조사, 어미, 접두사, 접미사 등


### 형태소 어휘 사전

**형태소 어휘 사전(Morpheme Vocabulary)**은 자연어 처리에서 사용되는 단어의 집합인 어휘 사전 중에서도 각 단어의 형태소 정보를 포함하는 사전이다.

일반적으로 형태소 어휘 사전에는 각 형태소가 어떤 품사에 속하는지와 해당 품사의 뜻 등의 정보도 함께 제공된다.

**품사 태깅(POS Tagging)**은 텍스트 데이터를 형태소 분석하여 각 형태소에 해당하는 **품사(Part Of Speech, POS)**를 태깅하는 작업을 말한다.

ex. 그(명사) + 는(조사) + 나(명사) + 에게(조사) + 인사(명사) + 를(조사) + 했다(동사)

### KoNLPy

**KoNLPy**는 한국어 자연어 처리를 위해 개발된 라이브러리로 명사 추출, 형태소 분석, 품사 태깅 등의 기능을 제공한다.

KoNLPy 형태소 분석기<br>
- Okt(Open Korean Text)
- 꼬꼬마(Kkma)
- 코모란(Komoran)
- 한나눔(Hannanum)
- 메캅(Mecab)

```python
# Okt 토큰화
from konlpy.tag import Okt

okt = Okt()

sentence = "무엇을 상상할 수 있는 사람은 무엇이든 만들어 낼 수 있다."

nouns = okt.nouns(sentence)     # 명사
pharses = okt.phases(sentence)  # 구
morphs = okt.morphs(setence)    # 형태소
pos = okt.pos(setence)  # 품사 태깅
```

Okt에서 지원하는 대표적인 메서드는 명사 추출(`okt.nouns`), 구문 추출(`okt.phrases`), 형태소 추출(`okt.morphs`), 품사 태깅(`okt.pos`)이다.

```python
# 꼬꼬마 토큰화
from konlpy.tag import Kkma

kkma = Kkma()

sentence = "무엇을 상상할 수 있는 사람은 무엇이든 만들어 낼 수 있다."

nouns = kkma.nouns(sentence)
setences = kkma.setences(sentence)
morphs = kkma.morphs(sentence)
pos = kkma.pos(sentence)
```

### NLTK

**NLTK(Natural Language Toolkit)**는 자연어 처리를 위해 개발된 라이브러리이다.

- 토큰화, 행태소 분석, 구문 분석, 개체명 인식, 감성 분석 등과 같은 기능을 제공한다.
- 토큰화나 품사 태깅 작업을 위해서는 해당 작업을 수행할 수 있는 패키지나 모델을 다운로드해야 한다.
    - 대표적으로 Punkt 모델과 Averaged Perceptron Tagger 모델이 있으며 두 모델 모두 **트리뱅크(Treebank)**라는 대규모 영어 말뭉치를 기반으로 학습됐다.

```python
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# 영문 토큰화
from nltk import tokenize

sentence = "Those who can imagine anything, can create the impossible."

word_tokens = tokenize.word_tokenize(setence)
sent_tokens = tokenize.sent_tokenize(setence)
```

- `word_tokenize()`는 문장을 입력받아 공백을 기준으로 단어를 분리하고, 구두점 등을 처리해 각각의 단어(token)를 추출해 리스트로 반환한다.
- `sent_tokenize()`는 문장을 입력받아 마침표(.), 느낌표(!), 물음표(?) 등의 구두점을 기준으로 문장을 분리해 리스트로 반환한다.

```python
# 영문 품사 태깅

from nltk import tag
from nltk import tokenize

sentence = "Those who can imagine anything, can create the impossible."

word_tokens = tokenize.word_tokenize(setence)
pos = tag.pos_tag(word_tokens)
```

### spaCy

**spaCy**는 사이썬 기반으로 개발된 오픈 소스 라이브러리로서, 자연어 처리를 위한 기능을 제공한다. 
- NLTK 라이브러리와의 주요한 차이점은 빠른 속도와 높은 정확도를 목표로 하는 머신러닝 기반의 자연어 처리 라이브러리이다.
    - NLTK는 자연어 처리를 위한 다양한 알고리즘 예제를 제공
    - spaCy는 효율적인 처리 속도와 높은 정화고를 제공
- NLTK에서 사용하는 모델보다 더 크고 복잡하다.

spaCy는 GPU 가속을 제공하며 24개 이상의 언어로 사전 학습된 모델을 제공한다 대표적으로 `en_core_web_sm` 모델은 영어로 사전 학습된 모델 중 하나이다.

```python
# spaCy 품사 태깅
import spacy

nlp = spacy.load("en_core_web_sm")
sentence = "Those who can imagine anything, can create the impossible."
doc = nlp(setence)

for token in doc:
    print(f"[{token.pos_:5} - {toekn.tag_:3}]: {token.text}")
# [PRON - DT] : Those
# [PRON - WP] : who
# [AUX  - MD] : can
# [VERB - VB] : imagine
#      .
#      .
#      .
# [ADJ  - JJ] : impossible]
# [PUNCT - .] : .
```

token 객체에는 여러 속성이 포함돼 있다.
- `pos_`: 기본 품사 속성
- `tag_`': 세분화 품사 속성
- `text`: 원본 텍스트 데이터
- `text_with_ws`: 토큰 사이의 공백을 포함하는 텍스트 데이터
- `vector`: 벡터
- `vector_norm`: 벡터 노름

## 하위 단어 토큰화
---------

현대 자연어 처리에서는 신조어의 발생, 오탈자, 축약어 등을 고려해야 하기 때문에 분석할 단어의 양이 많아져 어려움을 겪는다. 이를 해결하기 위한 방법 중 하나는 **하위 단어 토큰화(Subword Tokenization)**가 있다.
- 하나의 단어가 빈번하게 사용되는 **하위 단어(Subword)**의 조합으로 나누어 토큰화하는 방법
    - 'Reinforcement' &rarr; 'Rein', 'force', 'ment'
- 단어의 길이를 줄일 수 있어서 처리 속도가 빨라진다.
- OOV 문제, 신조어, 은어, 고유어 등으로 인한 문제 완화

하위 단어 토큰화 방법으로는 바이트 페어 인코딩, 워드피스, 유니그램 모델 등이 있다.

### 바이트 페어 인코딩

**바이트 페어 인코딩(Byte Pair Encoding, BPE)**이란 다이그램 코딩이라고도 하며 하위 단어 토큰화의 한 종류이다.
- 연속된 글자 쌍이 더 이상 나타나지 않거나 정해진 어휘 사전 크기에 도달할 때까지 조합 탐지와 부호화를 반복
    - 자주 등장하는 단어는 하나의 토큰으로 토큰화
    - 덜 등장하는 단어는 여러 토큰의 조합으로 표현

원문: abracadabra<br>
- 바이트 페어 인코딩은 입력 데이터에서 가장 많이 등장한 글자의 빈도수를 측정하고, 가장 빈도수가 높은 글자 쌍을 탐색한다.
    1. 'ab' 글자 쌍이 가장 빈도수가 높으므로 'A'로 치환
        - AracadAra
    2. 'ra' 글자 쌍이 가장 빈도수가 높아 'B'로 치환
        - ABcadAB
    3. 'AB'라는 글자 쌍이 존재하므로 'C'로 치환
        -  CcadC
    
바이트 페어 인코딩은 자주 등장하는 글자 쌍(치환한 글자 쌍)을 어휘 사전에 추가한다.

말뭉치에서도 바이트 페어 인코딩을 적용할 수 있다.

- 빈도 사전: ('low', 5), ('lower', 2), ('newest', 6), ('widest', 3)
- 어휘 사전: ['low', 'lower', 'newest', 'widest']
    1. 빈도 사전 낸 모든 단어를 글자 단어로 나눈다.
        - 빈도 사전: ('l', 'o', 'w', 5), ('l', 'o', 'w', 'e', 'r', 2), ('n', 'e', 'w', 'e', 's', 't', 6), ('w', 'i', 'd', 'e', 's', 't', 3)
        - 어휘 사전: ['d', 'e', 'i', 'l', 'n', 'o', 'r', 's', 't', 'w']
    2. 빈도 사전을 기준으로 가장 자주 등장한 글자 쌍을 찾는다.
        - 'e', 's'가 가장 많이 등장한다. &rarr; 빈도 사전에서 'e', 's'를 'es'로 병합하고 어휘 사전에 'es'를 추가한다.
        - 빈도 사전: ('l', 'o', 'w', 5), ('l', 'o', 'w', 'e', 'r', 2), ('n', 'e', 'w', 'es', 't', 6), ('w', 'i', 'd', 'es', 't', 3)
        - 어휘 사전: ['d', 'e', 'i', 'l', 'n', 'o', 'r', 's', 't', 'w', 'es']

    3. 위의 과정을 반복한다.

#### 센텐스피스

**센텐스피스(Sentencepiece)** 라이브러리와 **코포라(Korpora)** 라이브러리는 토크나이저 학습에 이용한다.

- 센텐스피스 라이브러리
    -  바이트 페어 인코딩과 유사한 라이브러리를 이용
    - 입력 데이터를 토큰화하고 단어 사전을 생성
    - 워드피스, 유니코드 기반의 다양한 알고리즘을 지원하며 사용자가 직접 설정할 수 있는 하이퍼파라미터들을 제공해 세밀한 토크나이징 기능을 제공
- 코포라 라이브러리
    - 말뭉치 데이터를 쉽게 사용할 수 있게 제공
    - API 제공

#### 토크나이저 모델 학습

```python
# 학습 데이터세트 생성
from Korpora import Korpora

corpus = Korpora.load("korean_petitions")
petitions = corpus.get_all_texts()
with open("../datasets/corpus.txt", "w", encoding="utf-8") as f:
    for petition in petitions:
        f.write(petition + "\n")
```

corpus의 `get_all_texts` 메서드로 데이터세트를 한 번에 불러올 수 있다.

```python
# 토크나이저 모델 학습
from sentencepiece import SentencePieceTrainer

SentencePieceTrainer.Train(     
    "--input=../datasets/corpus.txt\
    --model_prefix=petition_bpe\
    --vocab_size=8000 model_type=bpe"
) 
```
학습이 완료되면 petition_bpe.model과 petition_bpe.vocab 파일 생성한다.
- model 파일: 학습된 토크나이저가 저장된 파일
- vocab 파일: 어휘 사전이 저장된 파일

```python
# 바이트 페어 인코딩 토큰화
from sentencepiece import SentencePieceProcessor

tokenizer = SentencePieceProcessor()
tokenizer.load("petition_bpe.model")

sentence = "안녕하세요, 토크나이저가 잘 학습되었군요!"
sentences = ["이렇게 입력값을 리스트로 받아서", "쉽게 토크나이저를 사용할 수 있답니다"]

tokenized_sentence = tokenizer.encode_as_pieces(sentence)   # 문장 토큰화
encoded_sentence = tokenizer.encode_as_ids(sentence)    # 정수 인코딩
decoded_ids = tokenizer.decode_ids(encoded_sentence)    # 정수 인코딩에서  문장 변환
decoded_pieces = tokenizer.decode_pieces(encoded_setence)   # 하위 단어 토큰에서 문장 변환
```

SentencePieceProcessor 클래스를 통해 학습된 모델을 불러올 수 있다. 
- `encode_as_pieces` 메서드: 문장을 토큰화
- `encode_as_ids`: 토큰을 정수로 인코딩해 제공
- `decode_ids` or `decode_pieces`: 문자열 데이터로 변환

```python
# 어휘 사전 불러오기
from sentencepiece import SentencePieceProcessor

tokenizer = SentencePieceProcessor()
tokenizer.load("petition_bpe.model")

vocab = {idx: tokenizer.id_to_piece(idx) for idx in range(tokenizer.get_piece_size())}
print(list(vocab.items())[:5])
print("vocab size :", len(vocab))
```

- `get_piece_size`: 센텐스피스 모델에서 생성된 하위 단어의 개수 반환
- `id_to_piece`: 정수값을 하위 단어로 변환하는 메서드

### 워드피스

**워드피스(Wordpiece)** 토크나이저는 바이트 페어 인코딩 토크나이저와 유사한 방법으로 학습되지만, 빈도 기반이 아닌 확률 깁나으로 글자 쌍을 병합한다.
- 모델이 새로운 하위 단어를 생성할 때 이전 하위 단어와 함께 나타날 확률을 계산해 가장 높은 확률을 가진 하위 단어를 선택한다.
    - 이렇게 선택된 하위 단어는 이후에 더 높은 확률로 선택될 가능성이 높다.
- 모델이 좀 더 정확한 하위 단어로 분리될 수 있다.

$$
score = \frac{f(x, y)}{f(x),f(y)}
$$

- $f$ : 빈도(frequency)를 나타내는 함수<br>
- $x, y$ : 병합하려는 하위 단어 
- $f(x, y)$ : $x$ 와 $y$ 가 조합된 글자 쌍의 빈도를 의미($xy$ 글자 쌍의 빈도)
- $score$ : $x$ 와 $y$ 를 병합하는 것이 적절한지를 판단하기 위한 점수

**워드 피스의 어휘 사전 구축 방법**<br>
- 빈도 사전: ('l', 'o', 'w', 5), ('l', 'o', 'w', 'e', 'r', 2), ('n', 'e', 'w', 'e', 's', 't', 6), ('w', 'i', 'd', 'e', 's', 't', 3)
- 어휘 사전: ['d', 'e', 'i', 'l', 'n', 'o', 'r', 's', 't', 'w']

    |<center>문자 쌍<center>|<center>문자 1<center>|<center>문자 2<center>|<center>score<center>|
    |-----|-----|-----|-----|
    |'es' : 9번|'e' : 17번|'s' : 9번|$\frac{9}{17 * 9} \simeq 0.06$|
    |'id' : 3번|'e' : 3번|'s' : 3번|$\frac{3}{3 * 3} \simeq 0.33$|

바이트 페어 인코딩은 'e'와 's'를 병합하지만 워드피스는 'i'와 'd'쌍을 병합한다.

- 빈도 사전: ('l', 'o', 'w', 5), ('l', 'o', 'w', 'e', 'r', 2), ('n', 'e', 'w', 'e', 's', 't', 6), ('w', 'id', 'e', 's', 't', 3)
- 어휘 사전: ['d', 'e', 'i', 'l', 'n', 'o', 'r', 's', 't', 'w', 'id']

이 과정을 반복해 연속된 글자 쌍이 더 이상 나타나지 않거나 정해진 어휘 사전 크기에 도달할 때까지 학습한다.

#### 토크나이저

토크나이저는 라이브러리의 워드피스 API를 이용하면 쉽고 빠르게 토크나이저를 구현하고 학습할 수 있다. 대표적으로 허깅 페이스의 **토크나이저스(Tokenizers)**가 있다.

토크나이저스 라이브러리는 **정규화(Normalization)**와 **사전 토큰화(Pre-tokenization)**를 제공한다.

- 정규화: 일관된 형식으로 텍스트를 표준화하고 모호한 경우를 방지하기 위해 일부 문자를 대체하거나 제거하는 등의 작업을 수행
    - 불필요한 공백 제거, 대소문자 변환, 유니코드 정규화, 구두점 처리, 특수 문자 처리 등
- 사전 토큰화: 입력 문장을 토큰화하기 전에 단어와 같은 작은 단위로 나누는 기능을 제공
    - 공백 혹은 구두점을 기준으로 입력 문장을 나눠 텍스트 데이터를 효율적으로 처리하고 모델의 성능을 향상시킬 수 있다.

```python
# 워드피스 토크나이저 학습
from tokenizers import Tokenizer
from tokenizers.model import WordPiece
from tokenizers.normalizers import Sequence, NFD, Lowercase
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(WordPiece())
tokenizer.normalizer = Sequence([NFD(), Lowercase()]) # 유니코드 정규화, 소문자 변환
tokenizer.pre_tokenizer = Whitespace() # 공백과 구두점을 기준으로 분리

tokenizer.train(["../datasets/corpus.txt"])
tokenizer.save("../models/petition_wordpiece.json")
```

학습 결과는 JSON 형태로 저장되며 정규화 및 사전 토큰화 등의 메타데이터와 함께 어휘 사전이 저장된다. 이렇게 생성된 JSON 파일을 활용해 토크나이저를 수행할 수 있다.

```python
from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder

tokenizer = Tokenizer.from_file("../models/petition_wordpiece.json")
tokenizer.decoder = WordPieceDecoder()

sentence = "안녕하세요, 토크나이저가 잘 학습되었군요!"
sentences = ["이렇게 입력값을 리스트로 받아서", "쉽게 토크나이저를 사용할 수 있답니다"]

encoded_sentence = tokenizer.encode(sentence)
encoded_sentences = tokenizer.encode_batch(sentences)

print("인코더 형식 :", type(encoded_sentence))

print("단일 문장 토큰화 :", encoded_sentence.tokens)
print("여러 문장 토큰화 :", [enc.tokens for enc in encoded_sentences])

print("단일 문장 정수 인코딩 :", encoded_sentence.ids)
print("여러 문장 정수 인코딩 :", [enc.ids for enc in encoded_sentences])

print("정수 인코딩에서 문장 변환 :", tokenizer.decode(encoded_sentence.ids))
```

최근 연구 동향은 더 큰 말뭉치를 사용해 모델을 학습하고 OOV의 위험을 줄이기 위해 하위 단어 토큰화를 활용한다.

이 알고리즘 이외에도 바이트 단위에서 토큰화하는 **바이트 수준 바이트 페어 인코딩(Byte-level Byte-Pair-Encoding, BBPE)**이나 크기가 큰 어휘 사전에서 덜 필요한 토큰을 제거하며 학습하는 **유니그램(Unigram)** 등이 있다.





