---
title: "Python 문법"
excerpt: "zip( ) / 아스테리스크(*) "

categories:
  - Python
tags:
  - [python]

permalink: /python/grammer/

toc: true
toc_sticky: true

date: 2025-02-26
last_modified_at: 2025-02-26
---

## 🦥 zip() 함수

`zip()` 함수는 2개 이상의 시퀀스를 짧은 길이를 기준으로 일대일 대응하는 새로운 튜플 시퀀스를 만드는 역할을 한다.

```python
a = [1, 2, 3, 4, 5]
b = [2, 3, 4, 5]
c = [3, 4, 5]
zip(a, b)
# <zip object at 0x105b6d9b0>
```

파이썬 3+에서는 제너레이터를 리턴한다. 제너레이터에서 실제값을 추출하기 위해서는 `list()`로 한 번 더 묶어주면 된다.

```python
list(zip(a, b))
# [(1, 2), (2, 3), (3, 4), (4, 5)]
list(zip(a, b, c))
# [(1, 2, 3), (2, 3, 4), (3, 4, 5)]
```

`zip()`의 결과 자체는 리스트 시퀀스가 아닌 튜플 시퀀스를 만들기 때문에, 값을 변경하는 게 불가능하다. 불변(Immutable) 객체다.

즉, `zip()`은 여러 시퀀스에서 동일한 인덱스의 아이템을 순서대로 추출하여 튜플로 만들어 주는 역할을 한다.


## 🦥 아스테리스크(*)

파이썬에서 `*`는 언팩(Unpack)이다. 시퀀스 언패킹 연산자(Sequence Unpacking Operator)로 말 그대로 시퀀스를 풀어헤치는 연산자를 뜻하며, 주로 튜플이나 리스트를 언패킹하는 데 사용한다.

```python
collections.Counter(nums).most.common(k)
# [(1, 3), (2, 2)]

# 언패킹을 했을 때
list(zip(*collections.Counter(nums).most_common(k)))
# [(1, 2), (3, 2)]

# 언패킹을 하지 않았을 때
list(zip(collections.Counter(nums).most_common(k)))
#[((1, 3),), ((2, 2),)]
```
