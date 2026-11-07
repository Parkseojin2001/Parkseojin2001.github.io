---
title: "[Coding test] (바킹독 알고리즘) 스택의 활용 - 수식의 괄호 쌍 (feat. Python)"
description: "바킹독 알고리즘 강의 「스택의 활용(수식의 괄호 쌍)」 내용을 바탕으로, 코딩 테스트에서 코드 작성 방법을 Python 기준으로 정리한 글입니다"

categories: [Algorithm, Coding test]
tags: [python, algorithm, coding-test]

permalink: /algorithm/coding-test/07

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2026-01-17
last_modified_at: 2026-01-17
---

## **INTRO**
-----

### **🔑 KEY POINT**

> **문제 해결 방법**<br>
> 1. 여는 괄호가 나오면 스택에 추가
> 2. 닫는 괄호가 나왔을 경우,
>   - 스택이 비어있으면 올바르지 않은 괄호 쌍
>   - 스택의 top이 짝이 맞지 않는 괄호일 경우 올바르지 않은 괄호 쌍
>   - 스택의 top이 짝이 맞는 괄호일 경우 pop
> 3. 모든 과정을 끝낸 후 스택에 괄호가 남아있으면 올바르지 않은 괄호 쌍, 남아있지 않으면 올바른 괄호 쌍

**🔗 강의 링크**

[[실전 알고리즘]0x08강 - 스택의 활용(수식의 괄호 쌍)](https://blog.encrypted.gg/936)

## 문제 풀이
--------

강의에서는 C++ 언어로 문제를 풀이하셨고 저는 파이썬으로 문제를 풀려고 합니다.

문제에 대한 설명 또한 강의자님의 설명을 그대로 가져온 것입니다.

<img src="../assets/img/algorithm/coding-test/0x08-problem_1.png">

**My Solution**

```python
# Authored by : 21011645
# https://www.acmicpc.net/problem/4949

import sys
input = sys.stdin.readline

parentheses = {
    ')' : '(',
    ']' : '['
}

while True:
    stack = []
    is_valid = 'yes'
    S = input().rstrip('\n')
    if S == '.':
        break
    for c in S:
        if c in list(parentheses.values()):
            stack.append(c)
        elif c in list(parentheses.keys()):
            if stack and stack[-1] == parentheses[c]:
                stack.pop()
            else:
                is_valid = 'no'
                break
    
    if stack:
        is_valid = 'no'
        
    print(is_valid)
```


위의 코드에서 사용한 `sys.stdin.readline`과 `input`의 차이점을 알아보자.

### **💡 `input()` vs `sys.stdin.readline()` 정리**

1. 기본 동작 차이

| 구분         | `input()` | `sys.stdin.readline()` |
| ---------- | --------- | ---------------------- |
| 개행 문자 `\n` | **자동 제거** | **포함됨** |
| 공백(` `)    | 유지        | 유지 |
| 반환 값       | 개행 없는 문자열 | 개행 포함 문자열|
| 사용 목적      | 일반 입력     | 빠른 대량 입력|

2. 실행 시간(성능) 비교

- `sys.stdin.readline()`은 버퍼 기반 I/O
- `input()`은 내부적으로 **추가 처리(개행 제거, 예외 처리)**가 있음
- 대량 입력에서는 `readline()`이 더 빠름

| 입력 규모        | 권장                     |
| ------------ | ---------------------- |
| 소량 (≤ 1e4 줄) | `input()`              |
| 대량 (≥ 1e5 줄) | `sys.stdin.readline()` |
