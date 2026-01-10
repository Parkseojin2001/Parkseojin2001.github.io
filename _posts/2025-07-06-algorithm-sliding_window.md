---
title: "슬라이딩 윈도우"
description: "고정된 사이즈의 윈도우를 이용해 문제를 푸는 알고리즘"

categories: [Book, 파이썬 알고리즘 인터뷰]
tags: [algorithm]

permalink: /algorithm/sliding-window/

toc: true
toc_sticky: true
math: true
mermaid: true

date: 2025-07-06
last_modified_at: 2025-07-06
---

슬라이딩 윈도우(Sliding Window)란 고정 사이즈의 윈도우가 이동하면서 윈도우 내에 있는 데이터를 이용해 문제를 풀이하는 알고리즘을 말한다. 투 포인터는 사이즈가 변하지만 슬라이딩 윈도우는 고정 사이즈이다.

<img src="https://velog.velcdn.com/images/iberis/post/6fc5e78d-ca22-4f96-ac48-1328ef03981f/image.jpg" height="412" width="712">

- 투 포인터는 주로 정렬된 배열을 대상으로 하며 좌우 포인터가 자유롭게 이동할 수 있는 방식이다.
- 슬라이딩 윈도우는 정렬되어 있지 않은 배열에도 적용할 수 있으며 좌 또는 우 한쪽 방향으로만 이동한다.
