# Seojin Devlog

AI와 데이터 엔지니어링을 공부하며 만든 프로젝트와 학습 기록을 모아 두는 개인 블로그입니다.
[Jekyll](https://jekyllrb.com/)과 [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy) 소스를 바탕으로 터미널 스타일로 직접 다시 디자인했습니다.

**https://parkseojin2001.github.io**

## 구성

| 메뉴 | 내용 |
|---|---|
| Projects | 프로젝트 소개와 상세 페이지 |
| Study | 공부한 내용 정리 (카테고리·태그로 탐색) |
| Devlog | 프로젝트를 만들며 쓴 개발 기록 |
| About | 소개, 기술, 경력, 연락처 |

글의 `project:` 값이 있으면 Devlog, 없으면 Study로 나뉩니다.

## 실행

```bash
bundle install
npm install
bash tools/run.sh      # http://127.0.0.1:4000
```

레이아웃이나 JS를 바꿨다면 `npm run build`로 `assets/js/dist`를 다시 만들어야 합니다.

## 새 글 쓰기

```bash
bash tools/new.sh      # 글 / Devlog / 프로젝트 파일을 대화식으로 생성
```

## 테스트

```bash
bundle exec ruby test/run.rb
```

사이트를 한 번 빌드한 뒤 페이지 구조, URL, 태그·카테고리 이름 등을 검사합니다.
빌드를 다시 하지 않으려면 `SKIP_BUILD=1`을 앞에 붙입니다.

## 배포

`main` 브랜치에 푸시하면 GitHub Actions(`.github/workflows/deploy.yml`)가 빌드해 GitHub Pages에 배포합니다.

## 라이선스

[MIT License](LICENSE). 테마의 원본은 [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy)(MIT)입니다.
