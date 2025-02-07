---
title: "선형 자료구조"
excerpt: "배열 / 연결 리스트 / 스택, 큐 / 데크, 우선순위 큐 / 해시 테이블"

categories:
  - Data Structure
tags:
  - [data-structure]

permalink: /data-structure/linear/

toc: true
toc_sticky: true
math: true

date: 2025-02-05
last_modified_at: 2025-02-07
---

# 🦥 배열
>배열(Array)은 값 또는 변수 엘리먼트의 집합으로 구성된 구조로, 하나 이상의 인덱스 또는 키로 식별된다.

ADT의 실제 구현 대부분은 배열 또는 연결 리스트를 기반으로 한다.
배열은 크기를 지정하고 해당 크기만큼의 연속된 메모리 공간을 할당받는 작업을 수행하는 자료형을 말한다. 배열은 **큐 구현에 사용되는 자료형**이다.

```C
  int arr[5] = {4, 7, 29, 0, 1};
```
배열에는 정적 배열, 동적 배열 2가지가 존재한다.

## 1. 정적 배열
정적 배열은 **연속된, 정해진 크기의 메모리 공간을 할당하며 같은 타입의 원소만**을 담을 수 있다.
이유는 정해진 크기의 메모리 공간을 할당하고, 이를 원소 타입, 원소 개수에 맞춰서 분할하기 때문이다. 또한 한 번 생성한 배열은 크기 변경이 불가능하다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FkBNj8%2FbtqSduPlr65%2FWoDvz8CZuGa0CgNPokzW10%2Fimg.png" width="200px" height="450px" title="정적배열">

## 2. 동적 배열
실제로 정적 배열을 활용하기엔 비효율이 초래되는 경우가 많다. 따라서 **크기를 사전에 지정하지 않고 자동으로 조정**할 수 있도록 하는 동적 배열의 필요성이 대두되었다. 파이썬에서는 리스트로 동적 배열을 구현할 수 있다.

```python
# list를 이용한 동적 배열 구현
a = list()
b = [1, 2, 3]
```
**동적 배열 원리**<br>
미리 초기값을 작게 잡아 배열을 생성하고, 데이터가 추가되어 배열이 꽉 채워지게 되면 큰 사이즈의 배열을 새로운 메모리 공간에 할당하고 기존 데이터를 모두 복사한다.<br>
배열의 사이즈 증가는 대부분 2배씩 이루어지며 이를 *더블링(Doubling)* 이라고 한다.<br>
단, 기존 데이터를 모두 복사할 때 O(n) 비용이 발생하는 단점이 있다.

```python
# 파이썬 더블링 구조
//cpython/Objects/listobject.c
// The growth pattern is: 0, 4, 8, 16, 25, 35, 46, 58, 72, 88, ...
new_allocated = (size_t)newsize + (newsize >> 3) + (newsize < 9 ? 3 : 6);
```

재할당 비율을 그로스 팩터, 즉 '성장 인자'라고 한다. 파이썬의 그로스 팩터는 초반에는 2배씩 늘려 가지만, 전체적으로는 약 1.125배로, 다른 언어에 비해서는 적게 늘려가는 형태로 구현되어 있다.
<br>

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fcz19f9%2FbtqSjBAmHr8%2FDWUvAKDh1qtFd5Gg6OV8R0%2Fimg.png" title="동적 배열에 엘리먼트를 추가하는 구조">

# 🦥 연결 리스트
> 연결 리스트(Linked List)는 데이터 요소의 선형 집합으로, 데이터의 순서가 메모리에 물리적인 순서대로 저장되지는 않는다.

연결리스트(Linked List)는 동적으로 새로운 노드를 삽입하거나 삭제하기가 간편하며, 연결 구조를 통해 **물리 메모리를 연속적으로 사용하지 않아도** 되기 때문에 관리도 쉽다. 또한 데이터를 구조체로 묶어서 포인터로 연결한다는 개념은 여러 가지 방법으로 다양하게 활용이 가능하다.(연속된 메모리 공간에 할당되지 않고 메모리 어딘가에 scattered 되어있다고 볼 수 있음)

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FkXyV6%2FbtqSgMCbtH5%2FA9EdB0cTIw0BY5fzuSoFG1%2Fimg.png" width="200px" height="400px">

연결 리스트는 특정 인덱스에 접근하기 위해서는 전체를 순서대로 읽어야 하므로 탐색에 O(n)이 소요된다. 반면, 시작 또는 끝 지머에 아이템을 추가하거나 삭제, 추출하는 작업은 O(1)에 가능하다(포인터만 변경하면 구현 가능). 연결 리스트는 **스택 구현에 쓰이는 자료형**이다.

# 🦥 스택, 큐

파이썬은 리스트가 스택과 큐의 모든 연산을 지원한다. 다만 리스트는 동적 배열로 구현되어 있어 큐의 연산을 수행하기에는 효율적이지 않아 데크(Deque)라는 별도의 자료형을 사용해야 좋은 성능을 낼 수 있다. 성능을 고려하지 않는다면, 리스트는 스택과 큐를 구현하기에 충분하다.

## 스택
> 스택(Stack)은 2가지 주요 연산을 지원하는 요소의 컬렉션으로 사용되는 추상 자료형이다.

스택은 **후입선출(LIFO)**로 처리되는 자료구조이다. 즉, 스택에 가장 마지막에 들어간 요소가 가장 처음으로 꺼내진다. 스택은 흔히 **연결리스트**로 구현된다. 이 경우에 스택 선언을 위해 메모리 내의 연속된 공간을 할당할 필요가 없어지며, 실제로 스택 내의 요소 간 순서가 메모리 내의 물리적 순서와는 무관하게 될 것이다. 스택의 주 연산은 **삽입(push)**와 **제거(pop)**이다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc1ePBX%2FbtqUnKDzSGH%2FWdZe0vrceZANxwTQft9jbK%2Fimg.png">

### 연결 리스트를 이용한 스택 ADT 구현

```python
class Node:
  def __init__(self, item, next):
    self.item = item    # 노드의 값
    self.next = next    # 다음 노드를 가리키는 포인터
  
class Stack:
  def __init__(self):
    self.last = None    # 가장 마지막 자리를 가리키는 포인터
  
  def push(self, item):
    self.last = Node(item, self.last)
  
  def pop(self):
    item = self.item
    self.last = self.last.next
    return item
```
1부터 4까지 값을 스택에 입력하면 다음과 같다.
```python
stack = Stack()
stack.push(1)
stack.push(2)
stack.push(3)
stack.push(4)
```
위의 코드는 아래처럼 도식화할 수 있다.<br>
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc4DiZm%2FbtqUwynMEnx%2FziK926SKIno034HXFkCTPk%2Fimg.png">

### 스택의 연산
위의 스택에 5를 push하는 과정을 표현하면 다음과 같다. (연결리스트에 삽입 & 스택 포인터 이동)
```python
stack.push(5)
```
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FQl7x1%2FbtqUuNyY40Z%2FAmh98BIqOGD9jb2zc5Rx10%2Fimg.png">

스택에서 pop하는 과정을 표현하면 다음과 같다. 값을 복사한 후, 스택포인터 last를 다음 노드로 이동시킨다. 주의할 점은 여기서 연결리스트의 노드 삭제가 일어나는 것이 아니라, 단순히 스택포인터 last만 이동한다는 점이다.

```python
stack.pop()
```
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FvmIcF%2FbtqUsdd3WhH%2Fk2g65PF5KxW8ApKNwVCKjk%2Fimg.png">

**Stack in Python**<br>
파이썬에서의 스택 자료형을 별도로 제공하지는 않지만, 앞서 말한 바와 같이 기본 자료형인 리스트로 스택의 모든 연산을 수행할 수 있다. push 연산은 append로, pop연산은 pop으로 수행 가능하다. 

## 큐 
> 큐(Queue)는 시퀀스의 한쪽 끝에는 엔티티를 추가하고, 다른 반대쪽 끝에는 제거할 수 있는 엔티티 컬렉션이다.

큐는 **선입선출(FIFO)**로 처리되는 자료구조이다. 즉, 큐에 가장 처음에 들어간 요소가 가장 처음으로 꺼내지며 흔히 **배열**로 구현된다. 큐에는 선형 큐와 원형 큐가 있다. 선형 큐는 front와 rear이 있으며 이와 달리 원형 큐는 큐의 head와 rear가 연결되어 있는 구조이다.

### 1. 선형 큐

<img src="https://velog.velcdn.com/images%2Fsuitepotato%2Fpost%2F58b0805e-8bf0-443d-ba9a-d8c1f37383aa%2Fqueue_concept_01.PNG">

### 2. 원형 큐

<img src="https://user-images.githubusercontent.com/52641909/111613749-3090e580-8822-11eb-9c9b-b53a674614fb.png" width="300px" height="300px">

#### 원형 큐의 연산
원형 큐는 투 포인트와 비슷한 원리로 동작하며 주 연산은 2가지가 있다.
- enQueue() : rear포인터가 앞으로 이동
- deQueue() : front 포인터가 앞으로 이동
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F5sTsj%2Fbtr4wMYLHB7%2F5BQSIIzs07HeI4nRr78cAk%2Fimg.png">

**Queue in Python**<br>
파이썬의 리스트는 큐의 모든 연산을 지원하기 때문에 그대로 사용해도 무방하지만 좀 더 나은 성능을 위해서는 양방향 삽입, 삭제가 모두 O(1)에 가능한 데크를 사용하는 편이 가장 좋다.

# 🦥 데크, 우선순위 큐
데크는 스택과 큐의 연산을 모두 갖고 있는 복합 자료형이며, 우선순위 큐는 추출 순서가 일정하게 정해져 있는 않은 자료형이다.

## 데크
> 데크(Deque)는 더블 엔디드 큐의 줄임말로, 글자 그대로 양쪽 끝을 모두 추출할 수 있는, 큐를 일반화한 형태의 추상 자료형(ADT)이다.

데크는 양쪽에서 삭제와 삽입을 모두 처리할 수 있으며, 스택과 큐의 특징을 모두 갖고 있다. 이 추상 자료형(ADT)의 구현은 배열이나 연결 리스트 모두 가능하지만, 특별히 이중 연결 리스트로 구현하는 편이 가장 잘 어울린다.

<img src="https://blog.kakaocdn.net/dn/WKCGm/btsywvs5JDo/KGE6el8lGRvdDsoE4Ku5mK/img.png">

이중 연결 리스트로 구현하게 되면, 양쪽으로 head와 tail이라는 이름의 두 포인터를 갖고 있다가 새로운 아이템이 추가될 떄마다 앞쪽 뒤쪽으로 연결시켜 주기만 하면 된다. 연결 후에는 포인터를 이동하면 된다.

**Deque in Python**<br>
파이썬에서는 데크 자료형을 collections 모듈에서 deque라는 이름으로 지원한다.
```python
import collections
d = collections.deque()
d.append(1)   # 데크 오른쪽에 데이터 추가
d.appendleft(2)   # 데크 왼쪽에 데이터 추가
d.popleft()   # 데크 왼쪽 데이터 제거
d.pop()   # 데크 오른쪽 데이터 제거
```

## 우선순위 큐

> 우선순위 큐는 큐 또는 스택과 같은 추상 자료형과 유사하지만 추가로 각 요소의 '우선순위'와 연관되어 있다.

우선순위 큐는 어떠한 특정 조건에 따라 우선 순위가 가장 높은 요소가 추출되는 자료형이다. 대표적으로 최댓값 추출을 들 수 있다. 우선 순위 큐는 최단 경로를 탐색하는 다익스트라(Dijkstra)알고리즘 등 다양한 분야에 활용되면 힙(Heap) 자료구조와도 관련이 깊다.

**Priority queue in Python**<br>
queue 모듈의 PriorityQueue 클래스가 존재하지만 거의 항상 heapq을 사용한다.

# 🦥 해시 테이블
> 해시 테이블(Hash Table) 또는 해시 맵은 키를 값에 매핑할 수 있는 구조인, 연관 배열 추상 자료형(ADT)을 구현하는 자료구조이다.

해시 테이블의 가장 큰 특징은 대부분의 연산이 분할 상환 분석에 따른 시간 복잡도가 O(1)이라는 점이다. 덕분에 데이터 양에 관계없이 빠른 성능을 기대할 수 있다는 장점이 있다.

## 해시
> 해시 함수란 임의 크기 데이터를 고정 크기 값으로 매핑하는데 사용할 수 있는 함수를 말한다.

ex) ABC &rarr; A1 / 1324BC &rarr; CB / AF32B &rarr; D5 (화살표의 역할을 하는 함수가 해시 함수다.)

**해시 테이블(hash table)**은 해시를 인덱스(index) 데이터를 저장하는 자료구조 이다.
데이터가 저장되는 곳을 **테이블(table)**, **버킷(bucket)** 또는 **슬롯(slot)**이라고 히며 해시 테이블의 기본 연산은 삽입, 삭제, 탐색이다. 해시 테이블은 대부분의 연산이 분할 상환 분석에 따른 최적의 경우 O(1) 시간복잡도를 가진다.

해시 테이블을 인덱싱하기 위해 해시 함수를 사용하는 것을 **해싱(Hashing)**이라고 한다. 해싱은 정보를 가능한 한 빠르게 저장하고 검색하기 위해 사용하는 중요한 기법 중 하나이며 용도와 요구사항에 따라 각각 다르게 설계되고 최적화된다.

성능이 좋은 해시 함수들의 특징은 다음과 같다.

- 해시 함수 값 충돌의 최소화
- 쉽고 빠른 연산
- 해시 테이블 전체에 해시 값이 균일하게 분포
- 사용할 키의 모든 정보를 이용하여 해싱
- 해시 테이블 사용 효율이 높을 것

### 생일 문제

해시 함수 값의 충돌의 발생에 대한 예로는 생일 문제를 들 수 있다. 생일의 가짓수는 윤년을 제외하면 365개이므로, 여러 사람이 모였을 때 생일이 같은 2명이 존재할 확률은 꽤 낮을 것 같다.
하지만 실제로는 23명만 모여도 그 확률은 50%를 넘고, 57명이 모이면 99%를 넘어선다.
Python을 활용한 간단한 실험을 통해 어렵지 않게 이를 증명할 수 있다.

```python
import random 

TRIALS = 100000   # 10만 번 실험
same_birthdays = 0    # 생일이 같은 실험의 수

# 10만 번 실험 진행
for _ in range(TRIALS):
  birthdays = []
  # 23명이 모였을 때, 생일이 같을 경우 same_birthdays += 1
  for i in range(23):
    birthday = random.randint(1, 365)
    if birthday in birthdays:
      same_birthdays += 1
      break
    birthdays.append(birthday)

# 전체 10만 번 실험 중 생일이 같은 실험의 확률
print(f'{same_birthdays / TRIALS * 100}%')
```
이 코드의 결과는 50.708%가 나온다.

### 비둘기집 원리
>비둘기집 원리란, n개 아이템을 m개 컨테이너에 넣을 때, n>m이라면 적어도 하나의 컨테이너에는 반드시 2개 이상의 아이템이 들어 있다는 원리를 말한다.

비둘기집 원리는 충돌이 일어날 수 밖에 없는 이유를 잘 설명해주는 예시이다. 비둘기집 원리에 따라 9개의 공간이 있는 곳에 10개의 아이템이 들어온다면 반드시 1번 이상은 충돌이 발생하게 된다. 좋은 해시 함수라면 충돌을 최소화하여 단 1번의 충돌만 일어나게 하겠지만, 좋지 않은 해시 함수의 경우 심하면 9번을 모두 충돌할 수도 있다. 여러 번 충돌하는 것은 추가 연산이 더 필요하므로 가급적 충돌을 최소화하는 것이 좋다.

### 로드 팩터
> 로드 팩터(Load Factor)란 해시 테이블에 저장된 데이터 개수 n을 버킷의 개수 k로 나눈 것이다.

$$
loadfactor = \frac{n}{k}
$$

로드 팩터 값이 1이면 해시 테이블이 꽉 찬 것이고, 1보다 큰 경우 해시 충돌이 발생했음을 의미한다.
로드 팩터 비율에 따라서 **해시 함수를 재작성**해야 될지 또는 **해시 테이블의 크기를 조정**해야 할지를 결정한다. 또한 이 값은 해시 함수가 키들을 잘 분산해 주는지를 말하는 **효율성** 측정에도 사용된다. 일반적으로 로드 팩터가 증가할수록 해시 테이블의 성능은 점점 감소하게 된다.

### 해시 함수

아래 그림은 해시 함수를 통해 키가 해시로 변경되는 과정을 도식화하는 과정을 표현한 것이다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FEx0JS%2FbtsH9KmykVC%2F9qFkSn7FeydQwHjkAYpDnk%2Fimg.png">

해시 테이블을 인덱싱하기 위해 해시 함수를 사용하는 것은 **해싱(Hashing)**이라고 하는데 해싱에는 다양한 알고리즘이 있으며, 최상의 분포를 제공하는 방법은 데이터에 따라 다르다. 가장 단순하고 널리 쓰이는 해싱 기법은 모듈로 연산을 이용한 나눗셈 방식이다.

$$
h(x) = x \,\, mod \,\, m
$$

$h(x)$는 입력값 $x$의 해시 함수를 통해 생성된 결과다. $m$은 해시 테이블의 크기로, 일반적으로 2의 멱수에 가깝지 않은 소수를 택하는 것이 좋다.



## 충돌

아래 그림에서 Emily와 Bill은 해시 값이 04로 같은 값이 되어 충돌이 발생했다. 이러한 충돌이 발생하면 이를 처리해야 한다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbtLKmF%2FbtsIaE0toFh%2FwVZlHw2wrlkRWgJObQh2ik%2Fimg.png">

### 개별 체이닝

입력값을 표로 정리하면 다음과 같다.

|키|값|해시|충돌 여부|
|---|---|---|---|
|Luis| 15 |03|  |
|Daniel|7|04|충돌|
|Emily|47 |01|  |
|Bill| 17|04|충돌|

이 표를 **개별 체이닝(Separate Chaining)** 방식으로 구현하면 다음 그림과 같다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FrvCJl%2FbtsIaGqt93p%2FWdcWRLRbPVHjjqDmWbaDxK%2Fimg.png">

개별 체이닝은 **충돌 발생 시 연결 리스트로 연결하는 방식**이다. 이 방식은 원래 해시 테이블 구조의 원형이기도 하며 가장 전통적인 방식으로, 흔히 해시 테이블이라고 하면 바로 이 방식을 말한다.

**개별 체이닝 원리**<br>
1. 키의 해시 값을 계산한다.
2. 해시 값을 이용해 배열의 인덱스를 구한다.
3. 같은 인덱스가 있다면 연결 리스트로 연결한다.

잘 구현한 경우 대부분의 탐색은 O(1)이지만 최악의 경우(모든 해시 충돌이 발생)에는 O(n)이 된다.

### 오픈 어드레싱

**오픈 어드레싱(Open Addressing)** 방식은 충돌 발생 시 탐사를 통해 빈 공간을 찾아나서는 방식이다. 무한정 저장할 수 있는 체이닝 방식과 달리, 이 방식은 전체의 슬롯의 개수 이상은 저장할 수 없다. 충돌이 일어나면 테이블 공간 내에서 **탐사(Probing)를 통해 빈 공간을 찾아** 해결하며, 이 때문에 개별 체이닝 방식과 달리, 모든 원소가 반드시 자신의 해시값과 일치하는 주소에 저장된다는 보장이 없다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F9RiNm%2FbtsIehlCIrL%2FsAvOnyhcA5gI5fs2So9GeK%2Fimg.png">

오픈 어드레싱 방식도 여러가지가 있지만 그 중 가장 간단한 방식은 **선형 탐사(Linear Probing)**이다. 

#### 선형 탐사
선형 탐사는 충돌이 발생할 경우 **해당 위치부터 순차적으로 탐사를 하나씩 진행**한다. 특정 위치가 선점되어 있으면 바로 그다음 위치를 확인하는 식이다. 선형 탐사 방식은 구현 방법이 간단하면서도, 의외로 전체적인 성능이 좋은 편이기도 하다.

하지만 선형 탐사에는 한 가지 문제점이 있다. 그것은 해시 테이블에 저장되는 데이터들이 고르게 분포되지 않고 뭉치는 경향이 있다는 점이다. 해시 테이블 여기저기에 연속된 데이터 그룹이 생기는데 이를 **클러스터링(Clustering)**이라 하는데, 클러스터들이 점점 커지게 되면 인근 클러스터들과 서로 합쳐지는 일이 발생한다. 그렇게 되면 해시 테이블의 특정 위치에는 데이터가 몰리게 되고, 다른 위치에는 상대적으로 데이터가 거의 없는 상태가 될 수 있다. 이러한 현상은 탐시 시간을 오래 걸리게 하며, 전체적으로 해싱 효율을 떨어뜨리는 원인이 된다. 이러한 단점을 보완한 방식으로 **이중 해싱(Double Hashing)**이 있다.

#### 이중 해싱
**탐사할 해시값의 규칙성을 없애버려서 clustering을 방지하는 기법**이다. 2개의 해시함수를 준비해서 하나는 최초의 해시값을 얻을 때, 또 다른 하나는 해시충돌이 일어났을 때 탐사 이동폭을 얻기 위해 사용한다.

오픈 어드레싱 방식은 버킷 사이즈보다 큰 경우에는 삽입할 수 없다. 따라서 일정 이상 채워지면(로드 팩터 비율 이상), 그로스 팩터의 비율에 따라 더 큰 크기의 또 다른 버킷을 생성한 후 여기에 새롭게 복사하는 **리해싱(Rehasing)** 작업이 일어난다.(동적 배열의 더블링과 유사)

### 언어별 해시 테이블 구현 방식
리스트와 함께 파이썬에서 가장 흔하게 쓰이는 자료형인 딕션너리는 해시 테이블로 구현되어 있다. 파이썬의 해시 테이블은 충돌 시 오픈 어드레싱 방식으로 구현되어 있다. 그 이유는 체이닝 시 `malloc`으로 메모리를 할당하는 오버헤드가 높아 오픈 어드레싱을 택한 것이다.

<img src="https://blog.kakaocdn.net/dn/b1e9YW/btrXdNcUvXg/PUA1BePMcO4Z2hPY5lc6k1/img.png">

오픈 어드레싱의 한 방식인 선형 탐사 방식은 일반적으로 체이닝에 비해 성능이 좋지만 슬롯이 80% 이상이 차게 되면 급격한 성능 저하가 일어나며 로드 팩터 1 이상은 저장할 수 없다. 또한 선형 탐사 방식은 공간이 찰수록 탐사에 점점 더 오랜 시간이 걸리며, 가득 차게 될 경우 더 이상 빈 공간을 찾을 수 없다. 따라서 최근의 모던 언어들은 오픈 어드레싱 방식을 택해 성능을 높이는 대신, 로드 팩터를 작게 잡아 성능 저하 문제를 해결한다. 파이썬의 로드 팩터는 0.66이다.

|언어|방식|
|---|---|
|C++|개별 체이닝|
|자바|개별 체이닝|
|고(Go)|개별 체이닝|
|루비|오픈 어드레싱|
|파이썬|오픈 어드레싱|
