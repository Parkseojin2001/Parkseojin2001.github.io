---
title: "정렬"
description: "버블 정렬 / 삽입 정렬 / 합병 정렬 / 힙 정렬 / 퀵 정렬"

categories: [Algorithm]
tags: [algorithm]

permalink: /algorithm/sorting/

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-02-11
last_modified_at: 2025-03-12
---

> 정렬 알고리즘은 목록의 요소를 특정 순서대로 넣는 알고리즘이다. 대개 숫자식 순서(Numerical Order)와 사전식 순서(Lexicographical Order)로 정렬한다.

## 버블 정렬
---------

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

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fcc3r2Z%2Fbtra1ZJPfZb%2F4qxeTF4Ik9WlD0U93OWwL1%2Fimg.png">

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

## 힙 정렬
---------


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


## 안정 정렬 vs 불안정 정렬
---------
