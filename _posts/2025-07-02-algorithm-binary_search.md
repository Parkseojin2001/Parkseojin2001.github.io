---
title: "이진 탐색"
description: "이진 탐색(Binary Search)이란 정렬된 배열에서 타겟을 찾는 검색 알고리즘이다."

categories: [CS, Algorithm]
tags: [algorithm]

permalink: /algorithm/binary_search/

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-07-02
last_modified_at: 2025-07-02
---


이진 탐색(검색)은 값을 찾아내는 시간 복잡도가 $O(log\ n)$이라는 점에서 대표적인 로그 시간 알고리즘이며, 이진 탐색 트리(BST)와도 유사한 점이 만다. 그러나 이진 탐색 트리가 정렬된 구조를 저장하고 탐색하는 '자료구조'라면, 이진 탐색(검색)은 정렬된 배열에서 값을 찾아내는 '알고리즘' 자체를 지칭한다.

이진 탐색의 원리는 단순한 값 탐색 뿐만 아니라, 조건을 만족하는 최대값 또는 최소값을 찾는 문제, 실수 범위 내에서의 탐색 문제 등에도 응용될 수 있습니다. 이러한 확장된 문제들에서는 이진 탐색을 통해 탐색 범위를 점진적으로 좁혀나가며 문제의 해를 찾습니다.

## 이진 탐색의 기본 원리

1. **정렬된 배열**: 이진 탐색을 수행하기 전에 데이터가 정렬되어 있어야 한다. 정렬되지 않는 상태이면 올바른 결과를 보장할 수 없다.

2. **중간점 선택**: 탐색 범위 내에서 중간 위치의 요소를 선택한다. 배열이 0부터 시작한다고 가정할 때, 중간점의 인덱스는 `(left + right) / 2`로 계산할 수 있다.

3. **탐색 조건 비교**: 중간점의 데이터와 찾고자 하는 값(target)을 비교한다.
    - 중간점의 데이터가 찾고자 하는 값과 일치하는 경우: 탐색 종료
    - 중간점의 데이터가 찾고자 하는 값보다 작은 경우: 탐색 범위의 왼쪽 부분을 버리고 오른쪽 부분을 새로운 탐색 범위로 설정
    - 중간점의 데이터가 찾고자 하는 값보다 큰 경우: 탐색 범위의 오른쪽 부분을 버리고 왼쪽 부분을 새로운 탐색 범위로 설정
4. **반복 수행**: 새로운 탐색 범위를 기반으로 2번과 3번의 과정을 찾고자 하는 값이 발견되거나 탐색 범위가 더 이상 존재하지 않을 때까지 반복한다.

## 이진 탐색의 장점

- **효울성**: 이진 탐색은 $O(log\ n)$의 시간 복접도를 가진다. 대규모 데이터셋에서도 탐색 시간이 로그 시간으로 증가하기 때문에 매우 빠르다.
- **간결함**: 알고리즘 구현이 간결하며, 이해하기 쉽다.

## 이진 탐색의 단점

- **정렬된 데이터**: 데이터가 미리 정렬되어 있어야 한다는 제한이 있다. 정렬되지 않은 데이터에는 적용이 불가하다.
- **동적 데이터셋**: 데이터셋이 자주 변경되어 재정렬이 필요한 경우, 이진 탐새의 효율이 떨어질 수 있다.

## 이진 탐색 구현

이진 탐색을 구현하는 방법은 크게 재귀적 방법과 반복적 방법으로 나눌 수 있다.

### 재귀적 방법

재귀적 방법은 함수가 자기 자신을 호출하여 문제를 해결하는 방식이다.

```python
def binary_search_recursive(arr, target, left, right):
    if left > right:
        return False

    mid = (left + right) // 2
    if arr[mid] == target:
        return True
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)
```

### 반복적 방법

반복적 방법은 while 루프 등을 사용하여 같은 작업을 반복하는 방식이다.

```python
def binary_search_iterative(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return True
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return False
```


