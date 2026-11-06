---
title: "[파이썬 알고리즘 인터뷰] 비트 조작"
description: "부울 연산자 / 비트 연산자 / 비트 조작 / 2의 보수"

categories: [Book, python-algorithm-interview]
tags: [algorithm]

permalink: /python-algo-interview/algorithm/bit-manipulation/

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-07-02
last_modified_at: 2025-07-06
---

비트를 조작하는 것은 하드웨어와 관련이 깊다. 전기회로를 연구하면서 true, false의 2개 값으로 논리 연산을 살명하는 부울대수(Boolean Algebra)를 회로에 적용했고, 논리 게이트(Logic Gate)를 만들어냈다. 이를 이용한 논리 회로(Logic Circuit)는  다양한 부분에 널리 활용되고 있다.

## 부울 연산자

부울 연산(Boolean Operation)에서 기본으로는 `AND`, `OR`, `NOT`이 있으며 이를 결합하거나 조합해 다른 보조 연산을 만들어 낼 수 있다. 대표적으로 XOR이 보조 연산에 해당하며, 기본 연산들의 조합으로 다음과 같이 `XOR`을 구성할 수 있다.

```python
x = y = True
print((x and not y) or (not x and y))

# False
```

## 비트 연산자

비트 연산자(Bitwise Operator)는 부울 연산자와 마찬가지로 동일하게 잘 작동한다. 그러나 비트 연산자 `NOT`(Bitwise NOT)인 `~`(틸드)는 부울 변수에 적용하면 `True`는 1로 간주되어 -2가 된다. 그 이유는 비트 연산자 `NOT`은 2의 보수에서 1을 뺀 값과 같기 때문이다. 십진수로 표현할 때는 `NOT x = -x - 1`이 된다. 

## 비트 조작

비트 연산 수행 결과를 살펴보면 다음과 같다.

1. `bin(0b0110 + 0b0010)`은 자릿수가 초과할 때 다음 자리로 넘겨주는 방식으로 처리하면 된다. 결과는 `0b1000`이다.
2. `bin(0b0011 * 0b0101)`은 십진수 곱셈과 동일하다. 결과는 `0b1111`이다.
3. `bin(0b0101 >> 2)`은 오른쪽으로 시프팅을 하면 결과를 구할 수 있다. `0b11`이다.
4. `bin(0b1101 << 2)`은 왼쪽으로 시프팅 수행을 하면 `0b110100`이다.
5. `bin(0b0101 ^ ~0b1100)`은 `bin(0b0101 ^ -b1101)`이고 이를 계산하면 `-0b1010`이다.

> `^`는 비트 연산자에서 `XOR` 연산을 의미한다.

### 자릿수 제한 비트 연산

- 자릿수 만큼의 최댓값을 지닌 비트 마스크 `MASK`를 만들고, 그 값과 XOR을 통해 값을 만든다.
    - `MASK = 0b1111`
    - `bin(0b0101 ^ (0b1100 ^ MASK)) = bin(0b0101 ^ 0b0011) = 0b0110`

## 2의 보수

2의 보수(Two's Complement)를 알아야 비트 조작을 제대로 할 수 있으며 특히 음수 처리할 때 유용하다.

### 2의 보수 숫자 포맷

- 컴퓨터에서 수를 할당할 때 앙수인 경우 `0xxx`를 사용하고 음수의 경우 `1xxx`를 사용한다.
- 4비트로 2의 보수를 표현하려면 `MASK`를 사용하면 된다.
    - `bin(-7 & MASK) = 0b1001`
    - `bin(1 & MASK) = 0b1`

<img src="https://i.namu.wiki/i/mK3vgX8q08DXax_Gx5nza8ZpT9ym9UKQP22wZbnID7XSzBU6MhWjNUtaZsgm8Cdscmew5Zz4MIX8zMVEbyQPwQ.webp" width="400" height="400">



### 비트 연산자 NOT & 2의 보수 수학 연산

2의 보수 수학 연산은 양수를 음수로, 음수를 양수로 바꾸는 작업을 말한다. 방법은 다음과 같다.

- '비트 연산자 NOT`은 모든 비트를 반전시키는 연산
    - 1001의 비트 연산자 NOT: 1001 -> 0110
- '2의 보수 수학 연산'은 비트 연산자 NOT에서 1을 더한 것    
    - 0111의 2의 보수 연산: 1000(NOT 연산) + 1 = 1001