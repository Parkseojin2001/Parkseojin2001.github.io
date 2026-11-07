# Database / Operating System Category Implementation Plan

**Goal:** Database를 SQL / Basics으로, Operating System을 Course Notes로 정리한다.

**Spec:** `docs/superpowers/specs/2026-11-07-database-os-categories-design.md`

- [x] 기존 29개 글과 104개 이미지의 본문, permalink, 해시를 임시 감사 파일에 기록한다.
- [x] 글의 categories와 폴더를 SQL / Basics / Course Notes 구조로 정리한다.
- [x] 이미지 폴더와 참조 경로를 통합한다.
- [x] 이미지 README와 기존 카테고리 테스트의 예상 분류를 갱신한다.
- [x] 전체 테스트와 빌드된 계층을 검증하고 본문·주소·이미지 보존을 확인한다.

기존 대화에서 편집한 내용은 보존하고, 커밋이나 배포는 수행하지 않는다.

검증: 전체 74 tests / 526 assertions 통과. 빌드된 카테고리 계층 확인. 글 29개의 본문과 주소, 이미지 104개의 해시 보존 및 전체 글 로컬 이미지 참조 571개 연결 확인.
