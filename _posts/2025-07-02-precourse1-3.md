---
title: "[BoostCamp AI Tech / Pre-Course 1] 파이썬 기초 문법 2"
description: "네이버 부스트코스의 Pre-course 강의를 기반으로 작성한 포스트입니다."

categories: [NAVER BoostCamp AI Tech, Pre-Course]
tags: [Naver-Boostcourse, Pre-Course, python]

permalink: /boostcamp/pre-course/python-2/

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-07-02
last_modified_at: 2025-07-03
---

## Python Data Structure
--------

파이썬 기본 데이터 구조는 다음과 같다.

- 스택과 큐(stack & queue with list)
- 튜플과 집합(tuple & set)
- 사전(dictionary)
- Collection 모듈

### Stack & Queue

스택과 큐 모두 리스트를 사용하여 구현 가능하다.

#### 스택(Stack)

- 나중에 넣은 데이터를 먼저 반환하도록 설계된 메모리 구조
- Last In Firest Out (LIFO)
- Data의 입력을 Push, 출력을 Pop이라고 함
    - push를 `append()`, pop을 `pop()`를 사용

#### 큐(Queue)

- 먼저 넣은 데이터를 먼저 반환하도록 설계된 메모리 구조
- **First In First Out (FIFO)**
- Stack과 반대되는 개념
- put을 `append()`, get을 `pop(0)`를 사용

### Tuple & Set

#### 튜플(tuple)

- **값의 변경이 불가능한 리스트**
- 선언 시 "()"를 사용
- 리스트의 연산, 인덱싱, 슬라이싱 등을 동일하게 사용

> 값이 하나인 Tuple은 ","를 붙여야 수가 아닌 튜플로 인식한다.
> `t = (1)`은 정수로 인식 / `t = (1, )`은 튜플로 인식

#### 집합(set)

- 값을 순서없이 저장
- 중복을 불허하는 자료형
- set 객체 선언을 이용하여 객체 생성

```python
s = set([1, 2, 3, 1, 2, 3]) # s = {1, 2, 3}
s.add(1)    # 한 원소 추가
s.remove(1)  # 1 삭제
s.update([1, 4, 5, 6, 7])   # [1, 4, 5, 6, 7] 추가
s.discard(3)    # 3 삭제(3이 존재하지 않아도 오류가 안남)
s.clear()   # 모든 원소 삭제
```

집합(Set)는 수학에서 활용하는 집합 연산이 가능하다.
- 합집합: `s1.union(s2)` or `s1 | s2`
- 교집합: `s1.intersection(s2)` or `s1 & s2`
- 차집합: `s1.difference(s2)` or `s1 - s2`

### Dictionary

- 데이터를 저장할 때 구분 지을 수 있는 값을 함께 저장
- 구분을 위한 데이터 고유 값을 Identifier 또는 Key라고 함
- Key 값을 활용하여, 데이터 값(Value)를 관리
    - key와 value를 매칭하여 key로 value를 검색
- 다른 언어에서는 Hash Table이라는 용어를 사용
- {Key1: Value1, Key2: Value2, Key3: Value3 ...} 형태

```python
country_code = {"America": 1, "Korea": 82, "China": 86, "Japan": 81}  # dict()도 가능

country_code.items()    # Dict 데이터 출력
country_code.keys()     # Dict 키 값만 출력
country_code.values()   # Dict Value만 출력
country_code["German"] = 40 # Dict 추가
```

### Collection 모듈

- List, Tuple, Dict에 대한 Python Built-in 확장 자료 구조(모듈)
- 편의성, 실행 효율 등을 사용자에게 제공

존재하는 모듈은 아래와 같다.

```python
from collections import deque
from collections import Counter
from collections import OrderedDict
from collections import defaultdict
from collections import namedtuple
```

#### deque

- Stack과 Queue를 지원하는 모듈
- List에 비해 효율적 = 빠른 자료 저장 방식을 지원함
- rotate, reverse등 Linked List의 특성을 지원함
- 기존 list 형태의 함수를 모두 지원함

```python
from collections import deque

deque_list = deque()
for i in range(5):
    deque_list.append(i)    # 오른쪽에 삽입
deque_list.appendleft(10)   # 왼쪽에 삽입

deque_list.rotate(2)    # 2칸씩 앞으로 rotate

deque(reversed(deque_list))     # 뒤집기

deque_list.extend([5, 6, 7])    # 여러 원소를 오른쪽에 삽입
deque_list.extendleft([5, 6, 7])    # 여러 원소를 왼쪽에 삽입
```

- deque는 기존 list보다 횽ㄹ적인 자료구조를 제공
- 효율적 메모리 구조로 처리 속도 향상

#### OrderedDict

- Dict와 달리, 데이터를 입력한 순서대로 dict를 반환함
- But, python 3.6 부터는 입력한 순서를 보장하여 출력함

#### defaultdict

- Dict type의 값에 기본 값을 지정, 신규값 생성시 사용하는 방법

```python
from collections import defaultdict

word_count = defaultdict(lambda: 0) # Default 값을 0으로 설정
print(d["First"])   # 0을 출력
```

#### Counter

- Sequence type의 data element들의 개수를 dict 형태로 반환

```python
from collections import Counter

c = Counter()
c = Counter('gallahad')
print(c)    # Counter({'a': 3, 'l': 2, 'g': 1, 'd': 1, 'h': 1})
```
- Dict type, keyword parameter 등도 모두 처리 가능
- Set 연산들을 지원한다.

#### namedtuple

- Tuple 형태로 **Data 구조체**를 저장하는 방법
- 저장되는 data의 variable을 사전에 지정해서 저장함

```python
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
p = Point(11, y=22)
print(p[0] + p[1])  # 33
x, y = p
print(x, y) # 11, 22
print(p.x + p.y)    # 33
print(Point(x=11, y=22))    # Point(x=11, y=22)
```


## Pythonic code
---------

**파이썬 특유의 문법**을 활용하여 효율적으로 코드를 표현할 수 있다.

### split & join

#### split 함수

- string typle의 값을 "기준값"으로 나눠서 List 형태로 변환

```python
items = 'zero one two three'.split()    # 빈칸을 기준으로 문자열 나누기
print(items)    # ['zero', 'one', 'two', 'three']

example = 'python,java,javascript'  # ","을 기준으로 문자열 나누기
example.split(',')  # ['python', 'java', 'javascript']
```

#### join 함수

- String으로 구성된 list를 합쳐 하나의 string으로 반환

```python
colors = ['red', 'blue', 'green', 'yellow']
result = ' '.join(colors)   # 빈칸 1칸으로 연결
print(result)   # 'red blue green yellow


result = ', '.join(colors)  # ', '으로 연결
print(result)   # 'red, blue, green, yellow'
```

### list comprehension

- 기존 List 사용하여 간단히 다른 List를 만드는 기법
- 포괄적인 List, 포함되는 리스트라는 의미로 사용됨
- 파이썬에서 가장 많이 사용되는 기법 중 하나
- 일반적으로 for + append 보다 속도가 빠름

```python
result = [i for i in range(10) if i % 2 == 0]
print(result)   # [0, 2, 4, 6, 8]

word_1 = "abc"
word_2 = "dfg"
result = [i + j for i in word_1 for j in word_2]
print(result)  # ['ad', 'af', 'ag', 'bd', 'bf', bg', 'cd', 'cf', 'cg']

words = ['The', 'fox', 'in']
stuff = [[w.upper(), w.lower(), len(w)] for w in words]
print(stuff)    # [['THE', 'the', 3], ['FOX', 'fox', 3], ['IN', 'in', 2]]
```

#### Two dimensinal vs One dimensional

```python
case_1 = ['A', 'B', 'C']
case_2 = ['D', 'E', 'A']
result = [i + j for i in case_1 for j in case_2]
print(result)   # ['AD', 'AE', 'AA', 'BD', 'BE', 'BA', 'CD', 'CE', 'CA']

result = [[i + j for i in case_1] for j in case_2]
print(result)   # [['AD', 'BD', 'CD'], ['AE', 'BE', 'CE'], ['AA', 'BA', 'CA']]
```

> `pprint.pprint`: 데이터를 깔끔하게 출력할 때 사용하면 좋음

### enumerate & zip

- `enumerate`: list의 element를 추출할 때 번호를 붙여서 추출

```python
mylist = ['a', 'b', 'c', 'd']
print(list(enumerate(mylist)))  # [(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd')]

print({i:j for i, j in enumerate('A, B, C, D'.split())})
# {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
```

- `zip`: 두 개의 list의 값을 병렬적으로 추출함

```python
a, b, c = zip((1, 2, 3), (10, 20, 30), (100, 200, 300))
print(a, b, c)  # (1, 10, 100), (2, 20, 200), (3, 30, 300)

print([sum(x) for x in zip((1, 2, 3), (10, 20, 30), (100, 200, 300))])
# [111, 222, 333]

alist = ['a1', 'a2', 'a3']
blist = ['b1', 'b2', 'b3']

for i, (a, b) in enumerate(zip(alist, blist)):
    print(i, a, b)
    # 0, 'a1', 'b1'
    # 1, 'a2', 'b2'
    # 2, 'a3', 'b3'
```

### lambda & map & reduce

- lambda, map, reduce는 간단한 코들 다양한 기능을 제공함
- 하지만 코드의 직관성이 떨어져 python3에서 lambda나 reduce 사용을 권장하지 않음
    - 다양한 머신러닝 코드에서는 아직까지도 사용은 함

```python
ex = [1, 2, 3, 4, 5]
f = lambda x: x ** 2
print(list(map(f, ex))) # 1, 4, 9, 16, 25

from functools import reduce

print(reduce(lambda x, y: x + y, [1, 2, 3, 4, 5])) # 3, 6, 10, 15
```

### generator

- iterable object를 특수한 형태로 사용해주는 함수
- element가 사용되는 시점에 값을 메모리에 반환
- `:yield`를 사용해 한번에 하나의 element만 반환

```python
def generator_list(value):
    for i in range(value):
        yield i
```

#### generator comprehension

- list comprehension과 유사한 형태로 generator형태의 list 생성
- [] 대신 ()을 사용하여 표현

```python
gen_ex = (n*n for n in range(50))
```

> **When generator**
>
> - list 처럼 값을 반환해주는 함수는 generator로 생성
>   - 읽기 쉬운 장점, 중간 과정에서 loop가 중단될 수 있을 때 유용
> - 큰 데이터를 처리할 때는 generator expression을 고려
>   - 데이터가 커도 처리의 어려움이 없음
> - 파일 데이터를 처리할 때도 generator 사용

### Function passing arguments

함수에 입력되는 arguments는 다양한 형태가 존재한다.

- Keyword arguments: 함수에 입력되는 paramenter의 변수명을 사용, arguments를 넘김

```python
def print_somthing(my_name, your_name):
    print("Hello {0}, My name is {1}".format(your_name, my_name))

print_something("Sungchul", "TEAMLAB")
print_somthing(your_name="TEAMLAB", my_name="Sungchul")
```

- Default arguments: paramenter의 기본 값을 사용, 입력하지 않을 경우 기본값 출력

```python
def print_somthing_2(my_name, your_name="TEAMLAB"):
    print("Hello {0}, My name is {1}".format(your_name, my_name))

print_somthing_2("Sungchul", "TEAMLAB")
print_something_2("Sungchul")
```

- Variable-length arguments: 개수가 정해지지 않은 변수
    - Keyword arguments와 함께, argument 추가가 가능
    - **Asterisk(`*`)** 기호를 사용하여 함수의 parameter를 표시함
    - 입력된 값은 **tuple type**으로 사용할 수 있음
    - 가변인자는 오직 한 개만 맨 마지막 parameter 위치에 사용가능

#### Variable-length using asterisk

- 가변인자는 일반적으로 `*args`를 변수명으로 사용
- 기존 parameter 이후에 나오는 값을 tuple로 저장

```python
def asterisk_test(a, b, *args):
    return a + b + sum(args)

print(asterisk_test(1, 2, 3, 4, 5)) # a: 1, b: 2 args: (3, 4, 5)
```

#### Keyword variable-length

- Parameter 이름을 따로 지정하지 않고 입력하는 방법
- **asterisk(*) 두 개를 사용**하여 함수의 parameter를 표시함
- 입력된 값은 **dict type**으로 사용할 수 있음
- 가변인자는 오직 한 개만 기존 가변인자 다음에 사용

```python
def kwargs_test_3(one, two, *args, **kwargs):
    print(one + two + sum(args))
    print(kwargs)

kwargs_test_3(3, 4, 5, 6, 7, 8, 9, first=3, second=4, third=5)
# one = 3, two = 4, args=(5, 6, 7, 8, 9), kwargs = {'first': 3, 'second': 4, 'third': 5}
```

#### asterisk

- 흔히 알고 있는 `*`를 의미함
- 단순 곱셈, 제곱 연산, 가변 인자 활용 등 다양하게 사용됨


