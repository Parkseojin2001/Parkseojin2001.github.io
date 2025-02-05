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

date: 2025-02-05
last_modified_at: 2025-02-06
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

**Stack in Python**<br>
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

### 생일 문제

### 비둘기집 원리

### 로드 팩터

### 해시 함수


## 충돌

### 개별 체이닝

### 오픈 어드레싱

### 언어별 해시 테이블 구현 방식

