# 이미지 폴더

글 이미지는 `_posts`의 주제/세부 주제 구조에 맞춰 저장합니다.

| 글 폴더 | 이미지 폴더 |
| --- | --- |
| `_posts/naver-boostcamp/ai-core` | `assets/img/naver-boostcamp/ai-core` |
| `_posts/naver-boostcamp/pre-course` | `assets/img/naver-boostcamp/pre-course` |
| `_posts/database/basics` | `assets/img/database/basics` |
| `_posts/operating-system/course-notes` | `assets/img/operating-system/course-notes` |
| `_posts/deep-learning/nlp` | `assets/img/deep-learning/nlp` |
| `_posts/data-engineering/spark` | `assets/img/data-engineering/spark` |
| `_posts/machine-learning/scikit-learn` | `assets/img/machine-learning/scikit-learn` |

- 여러 부스트캠프 과정에서 함께 쓰는 이미지는 `naver-boostcamp/shared`에 둡니다. 기존 미사용 이미지 `resnet.png`도 이곳에 보관합니다.
- Database는 `sql`과 `basics`(Basics)로 나눕니다. 저장 함수·프로시저·트리거 이미지도 `sql`에 둡니다.
- Spark의 `ch01`~`ch05`, scikit-learn의 챕터별 폴더는 그대로 유지합니다.
- 논문 리뷰 이미지는 세부 주제에 따라 `paper-review/nlp` 등에 둡니다.
- 프로젝트 대표 이미지는 `projects/<프로젝트 이름>`, 개발 기록 이미지는 그 아래 `devlog`에 둡니다.
- `avatar.jpg`와 `favicons`는 사이트 공용 이미지입니다.

새 글은 `bash tools/new.sh`로 만들면 해당 이미지 폴더도 자동으로 생성됩니다.

글에서는 아래처럼 사이트 루트 기준 경로로 참조합니다.

```markdown
![설명](/assets/img/naver-boostcamp/ai-core/파일명.png)
```
