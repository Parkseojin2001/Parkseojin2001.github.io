---
title: "[Coding test] (바킹독 알고리즘) 배열 (feat. Python)"
description: "바킹독 알고리즘 강의 「배열」 내용을 바탕으로, 코딩 테스트에서 코드 작성 방법을 Python 기준으로 정리한 글입니다"

categories: [Algorithm, Coding test]
tags: [python, algorithm, coding-test]

permalink: /algorithm/coding-test/02

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2026-01-15
last_modified_at: 2026-01-15
---

### **INTRO**
-----

#### **🔑 KEY POINT**

> **배열의 성질**<br>
> 1. O(1)에 k번째 원소를 확인/변경 가능
> 2. 추가적으로 소모되는 메모리의 양(=overhead)가 거의 없음
> 3. Cache hit rate가 높음
> 4. 메모리 상에 연속한 구간을 잡아야 해서 할당에 제약이 걸림
> 
> **기능과 구현**<br>
> - 임의의 위치에 있는 원소를 확인/변경, O(1)
> - 원소를 끝에 추가, O(1) : `.append(x)`
> - 마지막 원소를 제거, O(1) : `.pop()`
> - 임의의 위치에 원소를 추가, O(N) : `.insert(i, x)`
> - 임의의 위치에 원소를 제거, O(N) : `.pop(i)`

**🔗 강의 링크**

[[실전 알고리즘] 0x03 - 배열](https://blog.encrypted.gg/927)

### **문제 풀이**
-----

강의에서는 C++ 언어로 문제를 풀이하셨고 저는 파이썬으로 문제를 풀려고 합니다.

문제에 대한 설명 또한 강의자님의 설명을 그대로 가져온 것입니다.

#### **문제 1**

<img src="../assets/img/post/barkingdog/0x03-problem_1.png">

**Solution**

```python
import sys
input = sys.stdin.readline

S = input().strip()

count = [0] * 26
list_s = list(S)

# ASCII CODE
for c in list_s:
    num = ord(c) - ord('a')
    count[num] += 1


for c in count:
    print(c, end=' ')
```


**1️⃣ 문자열(string)을 리스트(list)로**

- 문자열을 단어 단위 list로 변환 : `s.split()`
- 문자열을 원하는 값 기준으로 나누기 : `s.split('/')`
- 알파벡 하나씩 나누고 싶으면 : `list(s)`

**2️⃣ 리스트(list) 문자열로(string)** 

- `"".join(l)` : 공백없이 원소를 붙임
- `" ".join(l)` 은 공백을 추가하여 원소 붙임
- `"\n".join(l)` 은 한줄에 하나씩

단, 리스트의 구성요소가 모두 **문자열**이어야 가능!!


#### **문제 2**

이 문제는 앞의 기초 코드 작성 요령 강의에서 나온 문제입니다. 그 때 당시에는 강의에서 O($N^2$)으로 풀이하였으며 저의 코드는 O(N log N) 으로 풀었습니다.


<img src="../assets/img/post/barkingdog/0x01-problem_2.png">

하지만 이 문제의 O(N)으로 풀이가 가능한 문제입니다.

```python
def func2(arr, N):
    arr_100 = [0] * 100
    for i in range(N):
        if arr_100[100 - arr[i]] == 1:
            return 1
        else:
            arr_100[arr[i]] = 1
    
    return 0

test_cases_nums = [[[1, 52, 48], 3], [[50, 42], 2], [[4, 13, 63, 87], 4]]
    
for nums, N in test_cases_nums:
    print(func2(nums, N))
```

여기서 핵심 포인트는 나와 합해서 100이 되는 수의 존재 여부를 O(N)이 아닌 O(1)에 알아차리는 것이고, 이걸 배열을 이용해서 해결할 수 있습니다. 그 방법은 바로 각 수의 등장 여부를 체크하는 배열을 만드는 것입니다. 

나와 합이 100이 될 수 있는 수의 존재가 내 앞에 나왔는지 확인하는 과정을 통해서 해결합니다.