---
title: "비선형 자료구조"
excerpt: "그래프 / 트리 / 힙 / 트라이"

categories:
  - Data Structure
tags:
  - [data-structure]

permalink: /data-structure/nonlinear/

toc: true
toc_sticky: true
math: true

date: 2025-02-11
last_modified_at: 2025-02-11
---

# 🦥 그래프
> 수학에서, 좀 더 구체적으로 그래프 이론에서 그래프란 객체의 일부 쌍(pair)들이 '연관되어' 있는 객체 집합 구조를 말한다.

300여 년 전 도시의 시민 한 명이 "이 7개 다리를 한 번씩만 건너서 모두 지나갈 수 있을까?”라는 흥미로운 문제를 냈으며 이를 오일러가 해법을 발견하는 것이 그래프 이론의 시작이다.

**오일러 경로**<br>
아래 그림에서 A부터 D까지를 **정점(Vertex)**, a부터 g까지는 **간선(Edge)**으로 구성된 그래프라는 수학적 구조이다. 오일러는 모든 정점이 짝수 개의 **차수(Degree)**를 갖는다면 모든 다리를 한 번씩 건너서 도달할 수 있으며 이를 **오일러의 정리**라 부른다.

<img src="https://i.namu.wiki/i/V5asOLttGHZHjvN0y5ZyJvimN6cDdDiKC39QxbR83ENDaJmYU5y_Vrlke0Bl1-b0xGEzBEMha8KJaXZWKXcLRg.webp" width="400px" height="300px">

이처럼 모든 간선을 한 번씩 방문하는 유한 그래프를 오일러 경로 또는 한붓 그리기라고도 말한다.

**해밀턴 경로**<br>
> 해밀턴 경로는 각 정점을 한 번씩 방문하는 무항 또는 유향 그래프 경로를 말한다.

오일러는 간선을 기준으로 하지만 해밀턴 경로는 모든 정점을 한 번씩 방문하는 그래프를 말한다. 이러한 단순한 차이에도 불구하고 해밀턴 경로를 찾는 문제는 최적 알고리즘이 없는 대표적인 NP-완전 문제이다. 이 중 해밀턴 순환은 원래의 출발점으로 돌아오는 경로를 말하며 유명한 문제로는 최단 거리를 찾는 문제인 외판원 문제(TSP)가 있다.

## 그래프 정의

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbo7fx3%2FbtqR1EMrs51%2FoPAzkQbCkOEE0fnhbisaKk%2Fimg.png">

- vertex: 정점
  - 여러 가지 특성을 가질 수 있는 객체를 의미함. 노드(node)라고도 함.
- edge (간선)
  - 정점들 간의 관계를 의미. link라고도 함.

## 그래프 종류

### 무향 그래프 vs 방향 그래프

1. 무향 그래프 (undirected graph)
  -  무향 에지는 양방향의 의미임. 예컨대 조합(combination)처럼 (A,B) = (B,A)로 순서에 의미가 없다는 것.

2. 방향 그래프 (directed graph)
  - : 에지를 통해서 한쪽 방향으로만 갈 수 있다는 특징. 즉, <A.B> ≠ <B,A> 라는 특성을 보임.

<img src="https://chamdom.blog/static/43dcc5ebdae930f808c5563ac31f4159/b5a09/directed-and-undirected.png">

cf.) 가중치 그래프: 각 간선에 가중치를 부여한 형태의 그래프. 예를 들면 edge에 비용을 부과하는 형식으로 가중치가 부과될 수 있음.

<img src="https://chamdom.blog/static/9560e941305b79c8e4b9d82589d05740/5a46d/weighted-graph.png">

### 연결 그래프 vs 비연결 그래프

1. 연결 그래프 (Connected Graph)
  - 무방향 그래프에 있는 모든 노드에 대해 항상 경로가 존재하는 경우
2. 비연결 그래프 (Disconnected Graph)
  - 무방향 그래프에서 특정 노드에 대해 경로가 존재하지 않는 경우

<img src="https://chamdom.blog/static/edac49fc1e3d52be0824c7b46de733fe/b5a09/connected-and-disconnected.png">

### 사이클과 비순환 그래프
1. 사이클 (Cycle)
  - 단순 경로의 시작 노드와 종료 노드가 동일한 경우
  - 단순 경로 (Simple Path): 처음 정점과 끝 정점을 제외하고 중복된 정점이 없는 경로
2. 비순환 그래프 (Acyclic Graph)
  - 사이클이 없는 그래프

<img src="https://programmersarmy.com/trees-and-graph/images/cycles-1.png">


### 완전 그래프 
완전 그래프(Complete Graph)는 그래프의 모든 노드가 서로 연결되어 있는 그래프

<img src="https://chamdom.blog/static/046a8c1a3ac11df341ea042c220f38b2/772e8/complete-graph.png">

## 그래프 순회
> 그래프 순회란 그래프 탐색이라고도 불리우며 그래프의 각 정점을 방문하는 과정을 말한다.

그래프의 각 정점을 방문하는 그래프 순회에는 크게 **깊이 우선 탐색(DFS)**과 **너비 우선 탐색(BFS)**의 2가지 알고리즘이 있다. 일반적으로 DFS가 BFS에 비해 더 널리 쓰인다.

DFS는 주로 **스택으로 구현하거나 재귀로 구현**하며, 이후에 살펴볼 백트래킹을 통해 뛰어난 효용을 보인다. 반면, BFS는 주로 **큐로 구현**하며, 그래프의 최단 경로를 구하는 문제 등에 사용된다.<br>
그래프를 표현하는 방법에는 크게 **인접 행렬(Adjacency Matrix)**과 **인접 리스트(Adjacency List)**의 2가지 방법이 있다.

<img src="https://velog.velcdn.com/images%2Fjunhyeok-5%2Fpost%2Fccd81394-fdb4-418f-9f5d-76ecb2183995%2Fimage.png" width="300px" height="300px">

- 인접 리스트: 출발 노드를 키로, 도착 노드를 값으로 표현한다. 파이썬에서는 딕셔너리 자료형으로 나타낼 수 있으며 도착 노드는 여러 개가 될 수 있으므로 리스트 형태가 된다.

```python
# 그래프를 인접 리스트로 표현
graph = {
  1: [2, 3, 4],
  2: [5],
  3: [5],
  4: [],
  5: [6, 7],
  6: [],
  7: [3],
}
```

### DFS(깊이 우선 탐색)
일반적으로 DFS는 스택으로 구현하며, 재귀를 이용하면 좀 더 간단하게 구현할 수 있으며 코딩 테스트 시에도 재귀 구현이 더 선호되는 편이다.

**재귀 구조로 구현**<br>

```python
# 파이썬으로 구현
def recursive_dfs(v, discovered=[]):
  discovered.append(v)
  for w in graph[v]:
    if w not in discovered:
      discovered = recursive_dfs(w, discovered)
  return discovered

print(f'recursive_dfs: {recursive_dfs(1)}')
```
그래프 탐색 순서: 1 &rarr; 2 &rarr; 5 &rarr; 6 &rarr; 7 &rarr; 3 &rarr; 4 

**스택을 이용한 반복 구조로 구현**<br>

```python
def iterative_dfs(start_v):
  discovered = []
  stack = [start_v]
  while stack:
    v = stack.pop()
    if v not in discovered:
      discovered.append(v)
      for w in graph[v]:
        stack.append(w)
  return discovered

print(f'iterative dfs: {iterative_dfs(1)}')
```
그래프 탐색 순서: 1 &rarr; 4 &rarr; 3 &rarr; 5 &rarr; 7 &rarr; 6 &rarr; 2

### BFS(너비 우선 탐색)
BFS는 DFS보다 쓰임새는 적지만, 최단 경로를 찾는 다익스트라 알고리즘 등에 메우 유용하게 쓰인다. 재귀 구현은 불가능하다.

**큐를 이용한 반복 구조로 구현**<br>
```python
# 큐를 이용한 BFS 구현
def iterative_bfs(start_v):
  discovered = [start_v]
  queue = [start_v]
  while queue:
    v = queue.pop(0)  # 최적화를 위해 deque로 구현해 popleft()를 사용도 가능
    for w in graph[v]:
      if w not in discovered:
        discovered.append(w)
        queue.append(w)
  return discovered
```
그래프 탐색 순서: 1 &rarr; 2 &rarr; 3 &rarr; 4 &rarr; 5 &rarr; 6 &rarr; 7


## 백트래킹
> 백트래킹(Backtracking)은 해결책에 대한 후보를 구축해 나아가다 가능성이 없다고 판단되는 즉시 후보를 포기(백트랙-Backtrack)해 정답을 찾아가는 범용적인 알고리즘으로 제약 충족 문제에 특히 유용하다.

## 제약 충족 문제


# 🦥 트리




# 🦥 힙




# 🦥 트라이

