---
title: "[파이썬 알고리즘 인터뷰] 정렬"
description: "선택 정렬 / 버블 정렬 / 합병 정렬 / 삽입 정렬 / 힙 정렬 / 퀵 정렬"

categories: [Book, python-algorithm-interview]
tags: [algorithm]

permalink: /python-algo-interview/algorithm/sorting/

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-02-11
last_modified_at: 2025-06-30
---

> 정렬 알고리즘은 목록의 요소를 특정 순서대로 넣는 알고리즘이다. 대개 숫자식 순서(Numerical Order)와 사전식 순서(Lexicographical Order)로 정렬한다.

## 선택 정렬
---------

**선택 정렬(Selection sort)**은 배열(리스트)의 정렬되지 않는 부분에서 최솟값이나 최댓값을 찾아서 정렬되지 않는 첫 원소와 자리를 교환하는 방식으로 정렬하는 방법이다.

- 원리가 간단하여, 구현하기 쉽다.
- 비교할 항목이 적을 때 잘 작동한다.
- 이중 반복문을 사용하므로 시간복잡도는 $O(n^2)$이다.

|단계|0|1|2|3|4|비교|
|---|---|---|---|---|---|---|
|<center>값<center>|3|2|1|4|1|
|1단계|③|2|❶|4|1|인덱스 0~4에서 최솟값인 인덱스 2의 값과 인덱스 0의 값을 교환|
|2단계|❶|②|3|4|❶|인덱스 1~4에서 최솟값인 인덱스 4의 값과 인덱스 1의 값을 교환|
|3단계|❶|❶|③|4|❷|인덱스 2~4에서 최솟값인 인덱스 4의 값과 인덱스 2의 값을 교환|
|4단계|❶|❶|❷|4|❸|인덱스 3~4에서 최솟값인 인덱스 4의 값과 인덱스 3의 값을 교환|
|정렬끝|❶|❶|❷|❸|4|끝에서 두 번째 인덱스에 도달하면 끝|

### 반복문으로 구현하기

인덱스 0부터 마지막에서 두 번째 인덱스까지 반복한다.
- 현재 인덱스의 값을 최솟값으로 놓는다.
- (현재 인덱스 + 1)부터 마지막 인덱스까지 순회한다.
  - 최솟값을 현재 값을 비교하여 최솟값을 갱신한다.
- 최솟값과 현재 인덱스의 값을 교환한다.

```python
def selection_sort(arr):
  for i in range(len(arr) - 1):
    min_value_index = i
    for j in range(i + 1, len(arr)):
      if arr[j] < arr[min_value_index]:
        min_value_index = j
    arr[i], arr[min_value_index] = arr[min_value_index], arr[i]
```

### 재귀로 구현하기

- 기본 조건, 즉 재귀 함수를 종료하는 조건은 인덱스가 마지막에 도달했을 때다.
- 재귀 조건은 현재 인덱스가 마지막 인덱스보다 작을 때이다.

```python
def selection_sort_recursion(arr, n = 0):
  if n == len(arr) - 1:
    return
  min_index = n
  for i in range(n + 1, len(arr)):
    if arr[i] < arr[min_index]:
      min_index = i
  arr[n], arr[min_index] = arr[min_index], arr[n]
  selection_sort_recursion(arr, n + 1)
```

## 버블 정렬
---------

**버블 정렬(Bubble sort)**은 가장 간단한 정렬 알고리즘 중 하나로, 반복적으로 인접한 두 원소를 비교해서 정렬 순서에 따라 교환하는 과정을 반복한다.

```python
def bubbleSort(A):
  for i in range(1, len(A)):
    for j in range(0, len(A) - 1):
      if A[j] > A[j + 1]:
        A[j], A[j + 1] = A[j + 1], A[j]
```

버블 정렬은 $n$번의 라운드로 이뤄져 있으며, 각 라운드마다 배열의 아이템을 한 번씩 쭉 모두 살펴본다. 이는 구현 가능한 가장 느린 정렬 알고리즘이다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FTP0ub%2FbtrYrob3DhR%2FkTuGeUyanbWuwcdDdxjgn0%2Fimg.png">

- 회전: 배열크기 - 1
- 비교연산 : 배열크기 - 라운드
- 시간 복잡도 : $O(n^2)$

## 합병 정렬(병합 정렬)
---------

합병 정렬(Merge Sort)는 분할 정복(Divide and Conquer)의 진수를 보여주는 알고리즘이다. 
- 최선과 최악 모두 $O(nlog \ n)$ 시간 복잡도를 갖는다.
-  대부분의 경우 퀵 정렬보다는 느리지만 일정한 실행 속도뿐만 아니라 안정 정렬(Stable Sort)이라는 점에서 많이 사용되고 있다. 
- 퀵정렬로 풀리지 않은 문제가 병합 정렬로 풀리는 경우도 있다.

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fcc3r2Z%2Fbtra1ZJPfZb%2F4qxeTF4Ik9WlD0U93OWwL1%2Fimg.png" height="400" width="500">

1. 각각 더 이상 쪼갤 수 없을 때까지 계속해서 분할한다.
  - [38, 27, 43, 3] &rarr; [38, 27] / [43, 3] &rarr; [38] / [27] / [43] / [3]
2. 분할이 끝나면 정렬하면서 정복해 나간다.
  - [38] / [27] 정렬 &rarr; [27, 38]
  - [43] / [3] 정렬 &rarr; [3, 43]
  - [27, 38] / [3, 43] 정렬
    - 27과 3 비교(3 선택) &rarr; 27과 43 비교(27 선택) &rarr; 38과 43 비교(38 선택) &rarr; 43 선택
  - [3, 27, 38, 43] 정렬 완료


## 삽입 정렬
---------

**삽입 정렬(Insertion sort)**은 정렬되지 않는 배열(리스트)에서 현재 인덱스의 값을 왼쪽에 있는 모든 값과 차례로 비교하여, 값이 작으면(또는 크면) 자리를 교환하는 과정을 거쳐 바른 위치로 이동하도록 정렬하는 방법이다.

|단계|0|1|2|3|4|비교|
|---|---|---|---|---|---|---|
|<center>값<center>|3|2|4|1|5|
|1단계|③|❷|4|1|5|인덱스 1의 카드가 인덱스 0의 카드보다 작으므로 자리 교환|
|2단계|2|③|❹|1|5|인덱스 2의 카드가 인덱스 1의 카드보다 크므로 다음으로 넘어감|
|3단계|2|3|④|❶|5|인덱스 3의 카드가 인덱스 2의 카드보다 작으므로 자리 교환|
|3단계|2|③|❶|4|5|인덱스 2의 카드가 인덱스 1의 카드보다 작으므로 자리 교환|
|3단계|②|❶|3|4|5|인덱스 1의 카드가 인덱스 0의 카드보다 작으므로 자리 교환|
|4단계|1|2|3|④|❺|인덱스 4의 카드가 인덱스 3의 카드보다 크므로 다음으로 넘어감|
|정렬 끝|1|2|3|4|5|마지막 인덱스까지 진행하면 정렬 끝|


- 원리가 간단하여, 구현하기 쉽다.
- 평균 시간 복잡도는 $O(n^2)$이고, 최상은 $O(n)$이다.
- 일부가 정렬되어 있을 때 효율적인 정렬 방법이다.

### 반복문으로 구현

```python
def insertion_sort(arr):
  for i in range(1, len(arr)):
    while i > 0 and arr[i] < arr[i - 1]:
      arr[i], arr[i-1] = arr[i-1], arr[i]
      i -= 1
```

### 재귀로 구현

```python
def insertion_sort_recursion(arr, n = 1):
  if n == len(arr):
    return 
  i = n
  while i > 0 and arr[i] < arr[i-1]:
    arr[i-1], arr[i] = arr[i], arr[i-1]
    i -= 1
    insertion_srot_recursion(arr, n + 1)
```

## 힙 정렬
---------

**힙 정렬(Heap sort)**은 정렬한 자료를 최대 힙이나 최소 힙을 재배열하여 정렬을 하는 방법이다. 내림차순 정렬을 하려면 최소 힙으로 만들고 오름차순 정렬을 하려면 최대 힙을 사용한다.

- 모든 상황에서 시간 복잡도는 $O(nlog\ n)$이다.
- 큰 규모의 자료를 정렬할 때 효율적이지만, 자료가 매우 복잡하면 효율이 좋지 않다.
- 추가 메모리 공간이 필요하지 않다.
- 재귀를 사용하지 않아도 된다.
- 불안정 정렬이다.

> 최대 힙(Max Heap)은 부모가 항상 자식보다 크거나 같은 힙을 말하며 최소 힙(Min Heap)은 부모가 항상 자식보다 작거나 같은 힙을 말한다.

### 오름차순 정렬 과정

오름차순 정렬을 위해 최대 힙으로 구성해야 한다. 아래의 힙은 최대 힙 조건을 만족한다.

<img src="https://wikidocs.net/images/page/219635/heap_sort_01.png" height="300" width="300">

|인덱스|0|1|2|3|4|5|
|---|---|---|---|---|---|
|값|8|5|2|4|2|1|

- 가장 큰 값(루트)을 배열의 끝으로 보내면 된다. 즉, 배열의 첫 원소와 끝 원소를 교환한다.

|인덱스|0|1|2|3|4|5|
|---|---|---|---|---|---|
|값|**1**|5|2|4|2|**8**|

- 배열의 끝 원소는 정렬이 끝났으며 나머지 부분에 대해 정렬하면 된다. 즉, 위의 경우에는 인덱스 0부터 인덱스 4까지의 원소를 다시 최대 힙으로 만든다. 

<img src="https://wikidocs.net/images/page/219635/heap_sort_02.png" height="300" width="300">

|인덱스|0|1|2|3|4|5|
|---|---|---|---|---|---|
|값|5|4|2|1|2|8|

- 아직 정렬되지 않은 부분의 끝인 인덱스 4로 보낸다. 즉, 배열의 첫 원소와 인덱스 4의 원소를 교환한다.

|인덱스|0|1|2|3|4|5|
|---|---|---|---|---|---|
|값|**2**|4|2|1|**5**|8|

현재는 인덱스 4부터 인덱스 5까지는 정렬이 끝났다. 이후에는 아직 정렬되지 않은 인덱스 0부터 인덱스 3까지 같은 과정을 반복하면 정렬이 끝난다.

### 힙 정렬 구현하기

힙 정렬할 때는 위에서 본 것처럼 정렬이 한 번씩 끝날 때마다 정렬된 원소를 제외한 나머지 원소만 `heapify`한다.

- 배열의 첫 원소와 정렬되지 않는 부분 배열의 마지막 원소와 자리를 교환한다.
- 배열의 길이를 하나씩 줄이면서 위의 과정을 반복한다.

```python
def heappop(heap, end_index):
  heap[end_index], heap[0] = heap[0], heap[end_index]
  current, child = 0, 1
  while child < end_index:
    sibling = child + 1
    if sibling < end_index and heap[child] < heap[sibling]:
      child = sibling
      if heap[current] < heap[child]:
        heap[current], heap[child] = heap[child], heap[current]
        current = child
        child = current * 2 + 1
      else:
        break

# 최대 힙 
def heapify(arr, arr_len):
  last_parent == arr_len // 2 - 1
  for current in range(last_parent, -1, -1):
    while current <= last_parent:
        child = current * 2 + 1
        sibling = child + 1
        if sibling < arr_len and arr[child] < arr[sibling]:
          child = sibling
        if arr[current] < arr[child]:
            arr[current], arr[child] = arr[child], arr[current]
            current = child
        else:
            break

def heap_sort(arr):
  heapify(arr, arr_len)
  for end_index in range(len(arr) - 1, 0, -1):
      heappop(arr, end_index)
```

## 퀵 정렬
---------

퀵 정렬(Quick Sort)은 피벗(Pivot)을 기준으로 좌우를 나누는 특징 때문에 파티션 교환 정렬(Partition-Exchange Sort)이라고 불린다.
- 병합 정렬과 마찬가지로 분할 정복 알고리즘이다.
- 피벗보다 작으면 왼쪽, 크면 오른쪽과 같은 방식으로 파티셔닝하면서 쪼개 나간다.
- 여러 가지 변형과 개선 버전이 있다.
  - 여기서는 파티션 계획(Partition Scheme)을 다룰 것이다.

**로무토 파티션**<br>
- 항상 맨 오른쪽의 피벗을 택하는 단순한 방식이다.

```python
# 퀵 정렬 코드
def quicksort(A, lo, hi):
  if lo < hi:
    pivot = partition(lo, hi)
    quicksort(A, lo, pivot - 1)
    quicksort(A, pivot + 1, hi)

# 퀵 정렬 로무트 파티션 함수 코드
def partition(A, lo, hi):
  pivot = A[hi]
  left = lo
  for right in range(lo, hi):
    if A[right] < pivot:
      A[left], A[right] = A[right], A[left]
      left += 1
  A[left], A[hi] = A[hi], A[left]
  return left
```

- 피벗은 맨 오른쪽 값을 기준으로 한다.
- 위의 기준으로 2개의 포인터가 이동해서 오른쪽 포인터의 값이 피벗보다 작다면 서로 스왑한다.

<img src="https://velog.velcdn.com/images/hysong/post/e3eee8f8-f7e5-4565-a168-f1cbb2df0f05/image.png">

1. 오른쪽 right 포인터가 이동하면서 피벗의 값이 오른쪽 값보다 클 때, 왼쪽과 오른쪽의 스왑이 진행된다.
2. 스왑 이후에는 왼쪽 left 포인터가 함께 이동한다.
  - 결과는 피벗 값보다 작은 값은 왼쪽으로 큰 값은 오른쪽에 위치한다.
3. 피벗과 left 값 스왑을 통해 피벗을 중앙으로 이동시킨다.

> - left 값이 pivot보다 크면 left 포인터는 움직이지 않지만 right 포인터는 pivot보다 크면 움직이다가 작은 값을 가르키면 left 값과 right 값을 swap 시킨다.
> - swap이 되면 자연스럽게 left 값이 pivot보다 작아지므로 left 포인터도 움직인다.

```python
# 퀵정렬 전체 코드
def quicksort(A, lo, hi):
  def partition(lo, hi):
    pivot = A[hi]
    left = lo
    for right in range(lo, hi):
      if A[right] < pivot:
        A[left], A[right] = A[right], A[left]
        left += 1
    A[left], A[hi] = A[hi], A[left]
    return left

  if lo < hi:
    pivot = partition(lo, hi)
    quicksort(A, lo, pivot - 1)
    quicksort(A, pivot + 1, hi)
```
- 퀵 정렬은 매우 빠르며 굉장히 효율적인 알고리즘이지만 최악의 경우에는 $O(n^2)$이 된다.
  - 이미 정렬된 배열이 입력값으로 들어왔다면 피벗이 계속 오른쪽에 위치해 파티셔닝이 전형 이뤄지지않음
- 퀵 정렬은 입력값에 따라 성능 편차가 심한 편이다.

## 안정 정렬 vs 불안정 정렬
---------

> 안정 정렬(Stable Sort) 알고리즘은 중복된 값을 입력 순서와 동일하게 정렬한다.

안정 정렬은 동일한 값을 가진 요소들의 기존 순서를 유지하며, 데이터에 부가 정보가 있을 경우 유리하며 불안정 정렬은 기존 순서가 바뀔 수 있지만, 일반적으로 메모리 사용이 적고 속도가 빠를 수 있어 효율적인 구현에 쓰이기도 합니다.

| 정렬 알고리즘                    | 안정 정렬 여부 | 시간 복잡도 (최선 / 평균 / 최악)                      | 비고                              |
| -------------------------- | -------- | ------------------------------------------ | ------------------------------- |
| **선택 정렬**                  | ❌ 불안정    | `O(n²)` / `O(n²)` / `O(n²)`                | 단순하지만 느림, 항상 동일한 비교 횟수          |
| **버블 정렬**                  | ✅ 안정     | `O(n)` / `O(n²)` / `O(n²)`                 | 거의 사용 안 함, 최선은 정렬 상태일 때         |
| **합병 정렬**                  | ✅ 안정     | `O(n log n)` / `O(n log n)` / `O(n log n)` | 분할정복 방식, 안정 정렬                  |
| **삽입 정렬**                  | ✅ 안정     | `O(n)` / `O(n²)` / `O(n²)`                 | 데이터가 거의 정렬된 경우 효율적              |
| **힙 정렬**                   | ❌ 불안정    | `O(n log n)` / `O(n log n)` / `O(n log n)` | 힙을 이용한 정렬, 제자리 정렬 가능            |
| **퀵 정렬**                   | ❌ 불안정    | `O(n log n)` / `O(n log n)` / `O(n²)`      | 평균적으로 매우 빠름, 피벗 선택이 중요          |
| **Timsort (Python 기본 정렬)** | ✅ 안정     | `O(n)` / `O(n log n)` / `O(n log n)`       | 실제 Python, Java에서 사용되는 하이브리드 정렬 |
