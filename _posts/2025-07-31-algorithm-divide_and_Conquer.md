---
title: "분할 정복"
description: "분할 정복은 다중 분기 재귀를 기반으로 하는 알고리즘 디자인 패러다임을 말한다."

categories: [Book, 파이썬 알고리즘 인터뷰]
tags: [algorithm]

permalink: /algorithm/divide-and-conquer/

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-07-31
last_modified_at: 2025-07-31
---

분할 정복(Divide and Conquer)은 직접 해결할 수 있을 정도로 간단한 문제가 될 때까지 문제를 재귀적으로 쪼개나간 다음, 그 하위 문제의 결과들을 조합하여 원래 문제의 결과로 만들어 낸다.

대표적인 분할 정복 알고리즘으로는 병합 정렬을 들 수 있다.

<img src="https://i.namu.wiki/i/9zV6vDi73NI53UE4rtn30ec9YHGUVzxA3nmgku6mTYfRy0aB5nTjlsURqd087RRJnZaQ4ob0A8efFFyyyG73hg.webp">

- 분할: 문제를 동일한 유형의 여러 하위 문제로 나눈다.
- 정복: 가장 작은 단위의 하위 문제를 해결하여 정복한다.
- 조합: 하위 문제에 대한 결과를 원래 문제에 대한 결과로 조합한다.

분할 정복은 최적 부분 구조(Optimal Substructure)를 풀이하는 매우 중요한 기법 중 하나이다.
