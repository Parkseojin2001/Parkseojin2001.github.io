---
title: "비트 조작"
description: "부울 연산자 / 비트 연산자 / 비트 조작 / 2의 보수"

categories: [CS, Algorithm]
tags: [algorithm]

permalink: /algorithm/bit-manipulation/

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-07-02
last_modified_at: 2025-07-02
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
5. `bin(0b0101 ^ ~0b1100)`은 `bin(0b0101 ^ -b1101)`이고 이를 계산하면 `bin(0b)

> `^`는 비트 연산자에서 `XOR` 연산을 의미한다.

### 자릿수 제한 비트 연산

## 2의 보수

### 2의 보수 숫자 포맷

### 2의 보수 수학 연산


### 비트 연산자 NOT

