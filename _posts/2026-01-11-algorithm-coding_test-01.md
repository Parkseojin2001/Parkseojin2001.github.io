---
title: "[Coding test] (바킹독 알고리즘) 기초 코드 작성 요령 : 시간복잡도, 공간복잡도 & 코딩 테스트에서 코드 작성법(feat. Python)"
description: "바킹독 알고리즘 강의 「기초 코드 작성 요령」 내용을 바탕으로, 시간복잡도와 공간복잡도의 개념과 코딩 테스트에서의 코드 작성법을 Python 기준으로 정리한 글입니다"

categories: [Algorithm, Coding test]
tags: [python, algorithm, coding-test]

permalink: /algorithm/coding-test/01

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2026-01-11
last_modified_at: 2026-01-13
---

### **INTRO**
-----

> **🔑 KEY POINT**
>
> <div align='center'>
  <strong>문제를 보고 시간 복잡도와 공간 복잡도를 빠르게 파악하자!</strong><br>
  <strong>코딩 테스트와 개발은 다르다</strong><br>
  <strong>출력 맨 마지막 공백 혹은 줄바꿈이 추가로 있어도 상관없다</strong><br>
  <strong>디버거는 굳이 사용하지 않아도 된다.</strong></div>

이 포스트에서는 문제 풀이를 파이썬으로 풀어보기만 하도록 하겠습니다. 개념이나 내용이 궁금하다면 아래 링크에서 바킹독님의 강의를 참고하세요

**🔗 강의 링크**<br>
- [[실전 알고리즘] 0x01강 - 기초 코드 작성 요령 I](https://blog.encrypted.gg/922)
- [[실전 알고리즘] 0x01강 - 기초 코드 작성 요령 II](https://blog.encrypted.gg/923)


기초 1주차 강의에서 시간복잡도, 공간 복잡도 코딩테스트 문제 풀이 입니다.

#### **시간 복잡도 표**

문제에서 주어지는 시간 제한은 대부분 1초에서 5초 사이 정도로 입력 범위를 보고 문제에서 요구하는 시간복잡도가 어느 정도인지 알 수 있다.

<img src="https://velog.velcdn.com/images/doriskim/post/fadcc4a0-19e3-4e3d-a90c-b5acbdbab3c9/image.png">


### **문제풀이**
-----

강의에서는 C++ 언어로 문제를 풀이하셨고 저는 파이썬으로 문제를 풀려고 합니다.

문제에 대한 설명 또한 강의자님의 설명을 그대로 가져온 것입니다.

#### **문제 1**

<img src="../assets/img/post/barkingdog/0x01-problem_1.png">

```python
def func1(N):
    cnt = 0
    for i in range(1, N + 1):
        if i % 3 == 0 or i % 5 == 0:
            cnt += i

    return cnt

print(func1(16))
print(func1(34567))
print(func1(27639))
```

위의 solution의 시간복잡도는 O(n) 입니다.

이 문제를 O(N) 이 아닌 O(1)에 해결할 수 있는 방법도 있습니다.

```python
def func1_prime(N):
    cnt_3 = (N // 3) * (N // 3 + 1) // 2
    cnt_5 = (N // 5) * (N // 5 + 1) // 2
    cnt_15 = (N // 15) * (N // 15 + 1) // 2
    return 3 * cnt_3 + 5 * cnt_5 - 15 * cnt_15

print(func1_prime(16))
print(func1_prime(34567))
print(func1_prime(27639))
```

위 두 코드 실행시간은 아래와 같이 나옵니다.

- `func1` : 0.004319 seconds
- `func1-prime` : 0.000006 seconds

실제로 실행시간을 측정하니 O(1)이 월등하게 빠르다는 것을 알 수 있습니다.

#### **문제 2**

<img src="../assets/img/post/barkingdog/0x01-problem_2.png">

i가 0일 때 N-1개의 수에 대해 합을 100과 비교하고, i가 1일 때 N-2개의 수에 대해 합을 100과 비교하고, 이런식으로 쭉쭉쭉 가다가 i가 N-2일 때 1개의 수에 대해 합을 100과 비교하니 다 더하면 연산은 (N²-N)/2번 일어나고, 이걸 빅오표기법으로 나타내기 위해 상수 떼고 더 낮은 항을 없애고나면 O(N²)인걸 알 수 있습니다.

저의 첫 solution은 O(Nlog N)으로 푼 코드입니다.

```python
def func2(nums, N):
    left, right = 0, N - 1
    nums.sort()
    while left < right:
        if nums[left] + nums[right] == 100:
            return 1
        elif nums[left] + nums[right] < 100:
            left += 1
        else:
            right -= 1
    return 0
```

하지만 O(N)으로도 문제 2를 풀 수 있습니다.

```python
def func2_prime(nums, N):
    for i in range(N - 1):
        for j in range(i + 1, N):
            if nums[i] + nums[j] == 100:
                return 1
    
    return 0
```

#### **문제 3**

<img src="../assets/img/post/barkingdog/0x01-problem_3.png">


Python 코드로 문제를 풀면 아래와 같이 풀 수 있습니다.

```python
def func3(N):
    for i in range(N):
        if i * i > N:
            break
        elif i * i == N:
            return 1
    return 0
```

i가 1부터 올라가면서 1의 제곱이 N과 일치하는지 확인하고 2의 제곱이 N과 일치하는지 확인하고 계속 가다가 N과 일치하는 순간이 있으면 N이 제곱수이니 1을 반환하고, 일치하는 순간이 없이 i의 제곱이 N보다 커져 for문을 탈출하게 되었다면 제곱수가 아니니 0을 반환하면 됩니다.

이 방식은 i가 1부터 2, 3 이렇게 가다가 최대 $\sqrt{N}$까지 올라갈테니 시간복잡도는 $O(\sqrt{N})$이 됩니다. 이렇게 시간복잡도에 루트가 들어갈 수도 있습니다.

\* 참고: 문제 3은 O(log N)에 해결할 수 있는 방법이 있습니다. 강의를 완강하고 나서는 어떻게 해결이 가능

#### **문제 4**

<img src="../assets/img/post/barkingdog/0x01-problem_3.png">

```python
def func4(N):
    cnt = 2
    while cnt <= N:
        cnt *= 2
    return cnt // 2
```

코드에서 val은 2의 거듭제곱이 저장되는 변수입니다. 맨 처음에는 1로 시작해서 2, 4, 8, . . . 이렇게 커지다가 val을 2배했을 때 N보다 커지게 되는 순간에 while문을 탈출해서 val을 반환하면 문제에서 요구하는 답을 구할 수 있게 됩니다.

이 방식의 시간복잡도는 N이 2k 이상 2k+1 미만이라고 할 때 while문 안에서 val은 최대 k번만 2배로 커집니다. 그러고나면 val은 2k가 되고, 이후 2*val <= N이 거짓이 되기 때문입니다. 그러니까 N이 2k 이상 2k+1 미만이라고 할 때 시간복잡도가 O(k)이고 로그의 정의에 입각해서 생각할 때 k는 log N이니 결국 시간복잡도는 O(log N)이 됩니다.