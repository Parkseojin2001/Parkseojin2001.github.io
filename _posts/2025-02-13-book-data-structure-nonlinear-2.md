---
title: "[파이썬 알고리즘 인터뷰] 비선형 자료구조(2)"
description : "트리 / 힙 / 트라이"

categories: [Book, python-algorithm-interview]
tags: [data-structure]

permalink: /python-algo-interview/data-structure/nonlinear-2/

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-02-13
last_modified_at: 2025-02-17
---

## 트리
---------

> 트리는 계층형 트리 구조를 시뮬레이션하는 추상 자료형(ADT)으로, 루트 값과 부모-자식 관계의 서브트리로 구성되며, 서로 연결된 노드의 집합이다.

트리(Tree)는 하나의 뿌리에서 위로 뻗어 나가는 형상처럼 생겨서 **트리(나무)**라는 명칭이 붙었는데, 트리 구조를 표현할 때는 나무의 형상과 반대 방향으로 표현한다.<br>
트리의 속성 중 하나는 재귀로 정의된 자기 참조 자료구조라는 점이다. 여러 개의 트리가 쌓아 올려져 큰 트리가 된다. 흔히 서브트리로 구성된다고 말한다.

### 트리의 각 명칭
트리는 항상 루트(root)에서부터 시작된다. 루트는 자식(child)노드를 가지며, 간선(Edge)으로 연결되어 있다. <br>
- 차수(Degree): 자식노드의 개수
- 크기(size): 자신을 포함한 모든 자식 노드의 개수
- 높이(Height): 현재 위치에서부터 리프까지의 거리
- 깊이(Depth): 루트에서부터 현재 노드까지의 거리 

트리는 그 자식도 트리인 서브트리(Subtree) 구성을 띈다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcRzkvb%2Fbtq7C2kuAUB%2FXWLciUEf0AktmKcfKmfork%2Fimg.png">

레벨(level)은 0에서부터 시작한다. 트리는 항상 단방향이기 때문에, 간선의 화살표는 생략이 가능하다.

### 그래프 vs 트리

그래프와 트리의 가장 큰 차이점이 다음과 같다.<br>

***"트리는 순환 구조를 갖지 않는 그래프이다."***

핵심은 순환 구조(Cyclic)가 아니라는 데 있다. 트리는 특수한 형태의 그래프의 일종이며, 크게 그래프의 범주에 포함된다. 하지만 트리는 그래프와 달리 어떠한 경우에도 한 번 연결된 노드가 다시 연결되는 법이 없다. 이외에도 단방향, 양방향을 모두 가리킬 수 있는 그래프와 달리, 트리는 부모 노드에서 자식 노드를 가리키는 단방향뿐이다. 그뿐만 아니라 트리는 하나의 부모 노드를 갖는다는 차이점이 있으며 루트 또한 하나여야 한다.

### 이진 트리

트리 중에서도 가장 널리 사용되는 트리 자료구조는 **이진 트리**와 **이진 탐색 트리(BST)**다. 각 노드가 m개 이하의 자식을 갖고 있으면, m-ary 트리(다항 트리, 다진 트리)라고 한다. 여기서 m = 2일 경우, 즉 **모든 노드의 차수가 2 이하일 때**는 특별히 **이진 트리(Binary tree)**라고 구분해서 부른다. 이진 트리는 왼쪽, 오른쪽, 최대 2개의 자식을 갖는 매우 단순한 형태로, 다진 트리에 비해 훨씬 간결할 뿐만 아니라 여러 가지 알고리즘을 구현하는 일도 좀 더 간단하게 처리할 수 있어서, 대체로 트리라고 하면 대부분 이진 트리를 말한다.<br>

이진 트리에는 대표적으로 3가지 유형을 들 수 있다.
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbJvvRj%2Fbtq7FiNLxEL%2F3NP1kEr2VqZ2MrnXGYBAmk%2Fimg.png">

- 정 이진 트리(Full Binary Tree): 모든 노드가 0개 또는 2개의 자식 노드를 갖는다.

- 완전 이진 트리(Complete Binary Tree): 마지막 레벨을 제외하고 모든 레벨이 완전히 채워져 있으며, 마지막 레벨의 모든 노드는 가장 왼쪽부터 채워져 있다.

- 포화 이진 트리(Perfect Binary Tree): 모든 노드가 2개의 자식 노드를 갖고 있으며, 모든 리프 노드가 동일한 깊이 또는 레벨을 갖는다. 문자 그대로, 가장 완벽한 유형의 트리다.

### 이진 탐색 트리(BST)

**이진 탐색 트리(Binary Search Tree**)는 정렬된 트리를 말하는데, **노드의 왼쪽 서브 트리에는 그 노드의 값보다 작은 값들을 지닌 노드들로** 이뤄져 있는 반면, **노드의 오른쪽 서브트리에는 그 노드의 값과 같거나 큰 값들을 지닌 노드들로**이루어져 있는 트리를 뜻한다. 이렇게 왼쪽과 오른쪽 값들이 각각 값의 크기에 따라 정렬되어 있는 트리를 이진 탐색 트리라 한다. 이 트리의 가장 훌륭한 점은 **탐색 시 시간 복잡도가 O(log n)**이라는 점이다.

ex) 이진 탐색 트리를 이용해 원하는 값을 찾는 과정

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fmpj9E%2Fbtq7EgCmpJY%2FnXSer2QeZm5RDIU9AIBSS1%2Fimg.png">

균형이 많이 깨지면 탐색 시에 O(log n)이 아니라 O(n)에 근접한 시간이 소요될 수 있다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FLxX4e%2Fbtq7C1smBro%2Fjk7A4McDJwxkm2kHL14wgk%2Fimg.png">

운이 나쁘게 비효율적으로 구성된 경우인데, 이렇게 되면 연결 리스트와 다르지 않다. 그렇기 때문에 트리의 균형을 맞춰 줄 필요가 있으며 이를 위해 고안해낸 것이 바로 **자가 균형 이진 탐색 트리**다.



#### 자가 균형 이진 탐색 트리
> 자가 균형(또는 높이 균형) 이진 탐색 트리는 삽입, 삭제 시 자동으로 높이를 작게 유지하는 노드 기반의 이진 탐색 트리다.

자가 균형 이진 탐색 트리(Self-Balancing Binary Search Tree)는 최악의 경우에도 이진 트리의 균형이 잘 맞도록 유지한다. 즉 높이를 가능한 한 낮게 유지하는 것이 중요하다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FYxayN%2Fbtq7DnIJnO1%2FvEkKkGAFIrDRKkycRU0gc0%2Fimg.png">

1)은 불균형 트리로 7을 찾기 위해 7번의 연산이 필요하다. 2)는 균형 트리로 2번만에 7을 찾는게 가능하다. 노드의 개수가 많아질수록 불균형과 균형의 성능 차이가 점점 커진다. 따라서 트리의 균형, 즉 높이의 균형을 맞추는 작업은 매우 중요하다.

자가 균형 이진 탐색 트리의 대표적인 형태로는 AVL 트리와 레드-블랙 트리 등이 있으며, 특히 레드-블랙 트리는 높은 효율성으로 인해 실무에서도 매우 빈번하게 쓰이는 트리 형태이다.

**AVL 트리 특징**<br>
- 왼쪽, 오른쪽 서브 트리의 높이 차이가 최대 1이다.
- 어떤 시점에서 높이 차이가 1보다 커지면 회전(rotation)을 통해 균형을 잡아 차이를 줄인다.
- AVL 트리는 높이를 logN으로 유지하기 떄문에 삽입, 검색, 삭제의 시간 복잡도는 O(logN)이다.

<img src="https://blog.kakaocdn.net/dn/blxsRD/btq21CW9Fw3/WOk8F74J254K1pczckskEK/img.png">

**레드-블랙 트리 특징**<br>
- 모든 노드는 빨간색 혹은 검은색이다.
- 루트 노드는 검은색이다.
- 모든 리프 노드(NIL)들은 검은색이다.
  - NIL : null leaf, 자료를 갖지 않고 트리의 끝을 나타내는 노드
- 빨간색 노드의 자식은 검은색이다.
  - No Double Red. 빨간색 노드가 연속으로 나올 수 없다
- 모든 리프 노드에서 Black Depth는 같다.
  - 리프노드에서 루트 노드까지 가는 경로에서 만나는 검은색 노드의 개수가 같다.

<img src="https://velog.velcdn.com/images/kku64r/post/02fd3f93-505c-4952-943c-d7d68692fcf6/image.jpg">

### 트리 순회
> 트리 순회란 그래프 순회의 한 형태로 트리 자료구조에서 각 노드를 정확히 한 번 방문하는 과정을 말한다.

트리 순회(Tree Traversals) 또한 DFS 또는 BFS로 탐색하는데, 특히 이진 트리에서 DFS는 노드의 방문 순서에 따라 3가지 방식으로 구분된다.

1. 전위(Pre-Order) 순회(NLR)
  - 현재 노드를 먼저 순회한 다음 왼쪽과 오른쪽 서브트리를 순회
2. 중위(In-Order) 순회(LNR)
  - 왼쪽 서브트리를 순회한 다음 현재 노드와 오른쪽 서브트리를 순회
3. 후위 순위(Post-Order) 순회(LRN)
  - 왼쪽과 오른쪽 서브트리를 순회한 다음 현재 노드 순회

트리의 순회 방식은 재귀 또는 반복, 모두 구현이 가능하지만 트리의 재귀적 속성으로 인해 재귀 쪽이 훨씬 더 구현이 간단하다.

**전위 순회**
```python
def preorder(node):
  if node is None:
    return
  print(node.val)
  preorder(node.left)
  preorder(node.right)

# 전위 순회  실행
preorder(root)
```
**중위 순회**
```python
def inorder(node):
  if node is None:
    return
  inorder(node.left)
  print(node.val)
  inorder(node.right)

# 중위 순회 실행
inorder(root)
```

**후위 순회**
```python
def postorder(node):
  if node is None:
    return
  postorder(node.left)
  postorder(node.right)
  print(node.val)
  
# 후위 순회 실행
postorder(root)
```
## 힙
----------

> 힙은 힙의 특성(최소 힙에서는 부모가 항상 자식보다 작거나 같다)을 만족하는 거의 완전한 트리(Almost Complete Tree)인 특수한 트리 기반의 자료구조다.

힙(Heap)은 그래프나 트리와는 전혀 관계 없어 보이는 독특한 이름과 달리, 트리 기반의 자료구조다. 파이썬에는 최소 힙만 구현되어 있다. 우선순위 큐를 사용할 때 활용했던 heapq 모듈이 힙으로 구현되어 있으며, 파이썬에는 최소 힙은 부모가 항상 자식보다 작기 때문에 루트가 결국 가장 작은 값을 갖게 되며, 우선순위 큐에서 가장 작은 값을 추출하는 것은 매번 힙의 루트를 가져오는 형태로 구현된다. 

최소 힙은 부모 노드가 항상 작다는 조건만 만족할 뿐, 오른쪽의 자식 노드가 레벨 차이에도 불구하고, 왼쪽 노드보다 더 작은 경우도 얼마든지 있을 수 있다. **부모, 자식 간의 관계만 정의할 뿐, 좌우에 대한 관계는 정의하지 않는다**.

<img src="https://i.namu.wiki/i/sOvAReUFudDmVVsInW5_9-okGkTTKGA0eVvmxrDPq3V4Q0I3DX1Uv_ljVzcBzTihT4e2TleHAaz9fzSjs-eP7A.webp" width="400px" height="300px">

자식이 둘인 힙은 특별히 이진 힙(Binary Heap)이라 하며, 대부분은 이진 힙이 널리 사용된다.

<img src="https://i.namu.wiki/i/TxZ9HOn3UVo5ZMulyyfc2TnPAzFndu7TXXI29PmwTVKd9OUtaEnB6W5QfDLmGvaRBv9LuvDLe73rJEe6mQBCGQ.webp">

힙은 완전 이진 트리이기 때문에 배열에 순서대로 표현하기에 적합하며 이진 힙을 배열에 표현할 때 계산의 편의를 위해 인덱스는 1부터 사용한다. 힙은 우선순위 큐와 다익스트라 알고리즘에 활용된다. 다익스트라 알고리즘의 시간복잡도 $O(V^2)$에서 $O(E\,logV)$로 줄어 들 수 있다. 또한 프림 알고리즘, 중앙값의 근사값을 빠르게 구하는 데도 활용할 수 있다.

### 힙 연산
이진 힙을 구현하기 위한 클래스 정의를 하면 다음과 같다.
```python
class BinaryHeap(object):
  def __init__(self):
    self.items = [None] # 0 index는 사용하지 않기 위해 None으로 미리 설정
  
  def __len__(self):    # 매직 메소드로 Built-in 기능 동작
    return len(self.items) - 1
```

#### 삽입

힙에 요소를 삽입하기 위해서는 업힙(Up-Heap) 연산을 수행해야 한다. 업힙 연산은 `percolate_up()`이라는 함수로 정의한다.

1. 요소를 가장 하위 레벨의 최대한 왼쪽으로 삽입한다(배열에서는 가장 마지막에 삽입).
2. 부모 값과 비교해 값이 더 작은 경우 위치를 변경한다.
3. 계속해서 부모 값과 비교해 위치를 변경한다(가장 작은 값일 경우 루트까지 올라감).

<img src="https://i.namu.wiki/i/FFec9bpDVkPTs7ljXWPeCsAqmmTnuBbC98Rt3uCGjOFKGBKpJdzqsNYYK12sn4jue_lVHaoxFpIaUimNLY0Sri2jgDucsWdVqVpzLudZDJUHoEq6UAc6xxdJ7iJjzj9c2W-0d9wr_pRRwJPDIzVi8m-GhsTHHh2KmPm9hOUGmfs.webp" width="400px" height="550px">

```python
def _percolate_up(self):  # 2~3번 과정
  i = len(self)
  parent = i // 2
  while parent > 0:
    if self.items[i] < self.items[parent]:
      self.items[parent], self.items[i] = self.items[i], self.items[parent]
    i = parent
    parent = i // 2

def insert(self, k):  # 1번 과정
  self.items.append(k)
  self._percolate_up()
```
시간 복잡도는 O(log n)이다.

#### 추출

추출 과정은 루트를 추출하면 된다. 이 과정은 시간 복잡도가 O(1)이지만, 추출 이후에는 힙의 특성을 잃어버리기 때문에 이 유지하는 작업이 필요하다. 이 과정의 시간 복잡도는 O(log n)이다.

<img src="https://i.namu.wiki/i/FFec9bpDVkPTs7ljXWPeCmOEEGh2si6XdRY3vOPPg8-IrGwekSOJH9zCIsti8JHOycg7QHdLTCAJ8GGMhF8zCxC1YpCIVnxpMgZwmFjIEjeaGS8b-L9iRQOFclis4PBqnh3808Yaw7EZWfEn3C34nX30z9O9LMocxKTBHklMH4w.webp" width="400px" height="550px">

1. 루트 값을 추출한다.
2. 비어 있는 루트에는 가장 마지막 요소가 올라간다.
3. 자식 노드와 비교해서 자식보다 크다면 내려가는 다운힙(Down-Heap) 연산 수행

힙 추출에는 `percolate_down()`이라는 이름의 함수를 구현한다. 마찬가지로 인덱스 0은 비워둔다.

```python
def _percolate_down(self, idx):
  left = idx * 2
  right = idx * 2 + 1
  smallest = idx

  if left <= len(self) and self.items[left] < self.items[smallest]:
    smallest = left
  if right <= len(self) and self.items[right] < self.items[smallest]:
    smallest = right
  
  if smallest != idx:
    self.items[idx], self.items[smallest] = self.items[smallest], self.items[idx]

    self._precolate_down(smallest)
  
def extract(self):
  extracted = self.items[1]
  self.items[1] = self.items[len(self)]
  self._percolate_down(1)
  return extracted
```

**Heap in Python**<br>
- `heapq.heappush()`: insert() 함수에 대응한다.
- `heapq.heappop()`: extract() 함수에 대응한다.

**이진 힙 vs 이진 탐색 트리(BST)**<br>



## 트라이
---------

> 트라이(Trie)는 검색 트리의 일종으로 일반적으로 키가 문자열인, 동적 배열 또는 연관 배열을 저장하는 데 사용되는 졍련된 트리 자료구조다.

트라이는 실무에 매우 유용하게 쓰이는 자료구조로서, 특히 자연어 처리(NLP) 분야에서 문자열 탐색을 위한 자료구조로 널리 쓰인다. 트라이는 트리와 유사하지만, 이진 트리의 모습이 아닌 전형적인 다진 트리(m-ary Tree)의 형태를 띈다. 트라이는 각각의 문자 단위로 색인을 구축하는 하는 것과 유사하다.

<img src="https://velog.velcdn.com/images/seoy8n/post/64139686-441c-49fb-b9b4-f04669e7082b/image.png" width="450px" height="500px">