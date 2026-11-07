# 블로그 리디자인 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chirpy 블로그를 v5 디자인(상단바, 터미널 라벨)으로 바꾸고, 프로젝트 페이지·devlog 연결·글 생성 스크립트를 추가한다.

**Architecture:** Chirpy 소스를 그대로 개조한다. 기능 코드(검색·목차·다크 모드·MathJax·Mermaid·댓글)는 유지하고, `_layouts/default.html`의 틀과 SCSS 테마 변수를 교체하며, `projects` 컬렉션·레이아웃·Jekyll 플러그인·bash 스크립트를 추가한다. 검증은 사이트를 한 번 빌드한 결과 HTML을 Nokogiri로 검사하는 minitest 스위트로 한다.

**Tech Stack:** Jekyll 4.3, Chirpy 7.3 (Bootstrap 5.3, rollup, purgecss), Liquid, SCSS, Ruby minitest + Nokogiri, bash 3.2

**Spec:** `docs/superpowers/specs/2026-11-07-blog-redesign-design.md`

## Global Constraints

- 작업 브랜치: `redesign`. 태스크마다 커밋. 커밋 메시지 끝에 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- 기존 글 184개의 URL, 카테고리·태그 URL, `/feed.xml`, `/sitemap.xml`은 바뀌면 안 된다. 사라져도 되는 경로는 `/page<N>/`뿐이다.
- 색은 스펙 3장의 표 값을 그대로 쓴다(다크 배경 `#0d1117`, 포인트 `#7ee787` / 라이트 배경 `#ffffff`, 포인트 `#1a7f37` 등).
- 글꼴: 일반 `Pretendard`(jsDelivr `pretendard@1.3.9`), 고정폭 `JetBrains Mono`(Google Fonts). 대체: `-apple-system, sans-serif` / `SF Mono, Menlo, monospace`.
- 섹션 라벨 문자열은 정확히: `HELLO, I'M SEOJIN`, `./projects`, `./now`, `./blog`, `./categories`, `./devlog`, `// 목차`, `↳ project: `.
- 저장된 테마 선택이 없으면 다크.
- `_layouts`·`_includes`·`_javascript`를 바꾼 태스크는 커밋 전에 `npm run build`를 실행하고, 재생성된 `_sass/vendors/_bootstrap.scss`(purgecss 결과)와 `assets/js/dist/*.min.js`도 함께 커밋한다. 이 둘은 git으로 추적되는 생성물이다. 최초 1회 `npm ci` 필요.
- 테스트 실행: `bundle exec ruby test/run.rb` (전체) 또는 `bundle exec ruby test/run.rb test/<파일>_test.rb` (일부).
- 테스트용 샘플(프로젝트 2개, devlog 글 1개)은 `published: false`로 두어 실제 배포에는 나오지 않게 한다. 테스트 빌드는 `--unpublished`로 이것들을 포함한다.

## Review Focus

1. **카테고리가 하나뿐인 글**(예: `[Git]`): 글 위치 표시는 `blog / git`, 카테고리 트리에는 하위 없이 상위만 나와야 한다. → Task 3, Task 4 테스트
2. **`description`이 없는 글**: 블로그 목록에 빈 설명 줄(`<p></p>`)이 생기면 안 된다. → Task 4 테스트
3. **공백·대문자가 들어간 카테고리 이름**(`NAVER BoostCamp AI Tech`, `Coding test`): 트리와 홈 버튼의 링크가 실제로 빌드된 카테고리 페이지를 가리켜야 한다. → Task 4, Task 7 테스트
4. **선택 항목이 빠진 프로젝트**(`team`·`links`·`thumbnail` 없음): 빈 줄이나 빈 버튼이 없고, 썸네일 자리에는 첫 글자 기본 이미지가 나와야 한다. → Task 5 테스트
5. **스크립트를 저장소 루트가 아닌 곳에서 실행**(예: `_posts/` 안에서 `bash ../tools/new.sh`): 파일은 항상 저장소 루트 기준 경로에 만들어져야 한다. → Task 9 테스트

---

## 파일 구조

| 파일 | 역할 | 태스크 |
|---|---|---|
| `test/run.rb`, `test/test_helper.rb` | 테스트 실행기, 1회 빌드 + HTML 조회 헬퍼 | 0 |
| `test/fixtures/baseline-paths.txt` | 개편 전 HTML 경로 목록 | 0 |
| `test/urls_test.rb` | URL 회귀 검사 | 0 |
| `_sass/themes/_dark.scss`, `_light.scss` | 색 변수(v5) | 1 |
| `_sass/base/_typography.scss` | 글꼴, `.mono` 유틸리티 | 1 |
| `_data/origin/cors.yml`, `basic.yml` | 웹폰트 URL | 1 |
| `_javascript/theme.js` | 다크 기본 | 1 |
| `_layouts/default.html` | 새 틀(상단바·main·선택적 패널·푸터) | 2 |
| `_includes/topbar.html` | 로고·메뉴·검색·테마 전환·햄버거 | 2 |
| `_includes/footer.html` | 푸터 | 2 |
| `_sass/layout/_topbar.scss`, `_footer.scss`, `_sidebar.scss`(삭제) | 틀 스타일 | 2 |
| `_javascript/modules/layouts/nav.js` (신규), `sidebar.js`(삭제) | 모바일 메뉴 | 2 |
| `_layouts/post.html`, `_includes/post-path.html` (신규) | 글 페이지 헤더 | 3 |
| `_sass/pages/_post.scss` | 글 페이지 스타일 | 3 |
| `blog/index.html`, `_layouts/blog.html`, `_includes/category-tree.html`, `_includes/post-item.html` | 블로그 목록 | 4 |
| `_sass/pages/_blog.scss` | 블로그·카테고리·태그·아카이브 스타일 | 4 |
| `_config.yml` | 컬렉션, 기본값, `paginate_path` | 4, 5 |
| `_projects/sample-project.md`, `_projects/sample-solo.md`, `_posts/projects/sample-project/2026-01-01-sample-devlog.md` | 테스트 샘플 | 5 |
| `_layouts/project.html`, `_layouts/projects.html`, `projects.md`, `_includes/project-card.html`, `_includes/project-thumb.html` | 프로젝트 페이지 | 5 |
| `_sass/pages/_project.scss` | 프로젝트 스타일 | 5 |
| `_includes/project-backlink.html` | 글 상단 프로젝트 링크 | 6 |
| `_plugins/project-validator.rb` | 필수 항목·연결 검사 | 6 |
| `_layouts/home.html`(교체), `_data/now.yml` | 새 홈 | 7 |
| `_sass/pages/_home.scss` | 홈 스타일 | 7 |
| `_tabs/about.md` | About 틀 | 8 |
| `tools/new.sh`, `tools/templates/*.md` | 글 생성 스크립트 | 9 |

---

### Task 0: 테스트 하네스와 URL 기준선

**Files:**
- Modify: `Gemfile` (`test` 그룹에 `gem "minitest", "~> 5.25"`)
- Create: `test/run.rb`, `test/test_helper.rb`, `test/fixtures/baseline-paths.txt`, `test/urls_test.rb`
- Modify: `.gitignore` (`test/.site` 추가)

**Interfaces:**
- Produces:
  - `SiteBuild.dir -> String`: 빌드 결과 폴더(`test/.site`). 프로세스당 1회만 `bundle exec jekyll build --unpublished -d test/.site`를 `JEKYLL_ENV=production`으로 실행하고, 빌드 로그를 `SiteBuild.log -> String`에 보관한다. `SKIP_BUILD=1`이면 기존 결과를 재사용한다.
  - `html(path) -> Nokogiri::HTML::Document`: `path`는 사이트 URL(`"/blog/"`, `"/projects/x/"`). 끝이 `/`면 `index.html`을 붙인다. 파일이 없으면 테스트 실패.
  - `site_file?(path) -> Boolean`
  - `test/run.rb`: 인자로 받은 테스트 파일들(없으면 `test/**/*_test.rb` 전부)을 require하고 minitest 실행.

- [ ] **Step 1:** 기준선 생성. 아직 레이아웃을 바꾸지 않은 지금 상태(`main`과 사이트 결과가 같음)에서 `bundle exec jekyll build -d /tmp/baseline-site` 후 `cd /tmp/baseline-site && find . -name '*.html' | sed 's|^\.||' | sort > <repo>/test/fixtures/baseline-paths.txt`. 기대: 약 384줄.
- [ ] **Step 2:** `test/urls_test.rb` 작성

```ruby
class UrlsTest < Minitest::Test
  REMOVABLE = %r{\A/page\d+/index\.html\z}
  def test_every_baseline_path_still_exists
    missing = File.readlines("test/fixtures/baseline-paths.txt", chomp: true)
                  .reject { |p| p.match?(REMOVABLE) }
                  .reject { |p| site_file?(p) }
    assert_empty missing
  end
  def test_feed_and_sitemap_exist
    assert site_file?("/feed.xml")
    assert site_file?("/sitemap.xml")
  end
end
```

- [ ] **Step 3:** 헬퍼·실행기 구현 후 `bundle install && bundle exec ruby test/run.rb`. 기대: 2 tests, 0 failures (아직 아무것도 안 바꿨으므로 통과가 정상).
- [ ] **Step 4:** 커밋 `test: add site build test harness and URL baseline`

---

### Task 1: 색·글꼴·다크 기본

**Files:**
- Modify: `_sass/themes/_dark.scss`, `_sass/themes/_light.scss` (기존 변수 이름 유지, 값만 v5로. `--sidebar-*` 변수는 Task 2에서 삭제)
- Modify: `_sass/base/_typography.scss`
- Modify: `_data/origin/cors.yml`, `_data/origin/basic.yml` (`webfonts`)
- Modify: `_javascript/theme.js`
- Create: `test/style_test.rb`

**Interfaces:**
- Produces: 새 CSS 변수 `--accent-color`, `--accent-blue`, `--accent-purple`, `--accent-orange`, `--surface-bg`, `--divider-color`, `--font-sans`, `--font-mono`(다크·라이트 모두). 유틸리티 클래스 `.mono`(고정폭 글꼴), `.section-label`(고정폭, 0.8rem, `--text-muted-color`). 이후 모든 태스크가 이 이름을 쓴다.
- 변수 매핑: `--main-bg`=배경, `--surface-bg`=카드·코드 배경, `--main-border-color`=테두리, `--divider-color`=구분선, `--text-color`=본문, `--heading-color`=제목, `--text-muted-color`=보조, `--accent-color`=포인트 초록, `--link-color`=`--accent-blue`.

- [ ] **Step 1:** 실패 테스트 `test/style_test.rb`

```ruby
class StyleTest < Minitest::Test
  def css = File.read(File.join(SiteBuild.dir, "assets/css/jekyll-theme-chirpy.css"))
  def test_dark_palette  = assert_includes(css, "#0d1117") && assert_includes(css, "#7ee787")
  def test_light_palette = assert_includes(css, "#1a7f37")
  def test_fonts_loaded
    head = html("/").at("head").to_html
    assert_includes head, "JetBrains+Mono"
    assert_includes head, "pretendard"
  end
  def test_mono_utility  = assert_match(/\.mono\{[^}]*font-family:var\(--font-mono\)/, css)
end
```

- [ ] **Step 2:** `bundle exec ruby test/run.rb test/style_test.rb` → FAIL (`#0d1117` 없음)
- [ ] **Step 3:** 구현. `webfonts`는 JetBrains Mono(400;500;700) Google Fonts URL 하나로 바꾸고, Pretendard CSS(`https://cdn.jsdelivr.net/npm/pretendard@1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css`)는 `cors.yml`에 `pretendard:` 키로 추가해 `head.html` 웹폰트 링크 아래에 한 줄 추가. `theme.js`의 `visualState`는 저장된 모드가 없으면 시스템 설정 대신 `this.DARK`를 반환.
- [ ] **Step 4:** 테스트 PASS, `npm run build`, `npm test`(stylelint) PASS
- [ ] **Step 5:** 수동 확인: `bundle exec jekyll serve` 후 시크릿 창에서 열면 다크로 시작, 전환 버튼으로 라이트 전환 가능
- [ ] **Step 6:** 커밋 `feat: apply v5 color palette, fonts and dark default`

---

### Task 2: 새 틀 (상단바·푸터, 사이드바 제거)

**Files:**
- Modify: `_layouts/default.html`, `_includes/topbar.html`, `_includes/footer.html`, `_includes/search-results.html`
- Modify: `_sass/layout/_topbar.scss`, `_sass/layout/_footer.scss`, `_sass/layout/_index.scss`, `_sass/layout/_panel.scss`, 테마 파일의 `--sidebar-*` 변수 삭제
- Delete: `_includes/sidebar.html`, `_sass/layout/_sidebar.scss`, `_javascript/modules/layouts/sidebar.js`
- Create: `_javascript/modules/layouts/nav.js`
- Modify: `_javascript/modules/layouts.js`, `_javascript/{commons,home,post,page,categories,misc}.js` (`initSidebar` → `initNav`), `_javascript/modules/components/search-display.js` (`sidebar-trigger` 참조 제거)
- Create: `test/shell_test.rb`

**Interfaces:**
- Produces:
  - 상단바 DOM: `header#topbar-wrapper` 안에 `a.site-logo`(텍스트 `Seojin Park`, 앞에 `span.logo-dot`), `nav#site-nav` 안에 `a.nav-link` 3개(`/projects/`, `/blog/`, `/about/`, 현재 섹션에 `.active`), 기존 `#search`·`#search-trigger`·`#search-cancel`, `button#mode-toggle`, 모바일용 `button#nav-trigger`.
  - `initNav()`: `#nav-trigger` 클릭 시 `body`에 `nav-open` 속성 토글, `#mask` 클릭 시 닫기.
  - `default.html`은 `layout.panel_includes`가 있을 때만 `aside#panel-wrapper`를 렌더링한다(최근 업데이트·인기 태그 include는 제거). 이후 레이아웃은 front matter의 `panel_includes`로 오른쪽 영역을 채운다.
  - `.active` 판정: 페이지 URL이 `/projects/`로 시작 → Projects, `/blog/`로 시작하거나 `page.layout`이 `post`·`category`·`tag`·`categories`·`tags`·`archives` → Blog, `/about/` → About.

- [ ] **Step 1:** 실패 테스트 `test/shell_test.rb`

```ruby
class ShellTest < Minitest::Test
  PAGES = ["/", "/about/", "/categories/", "/data-engineering/learning-spark/05/"]
  def test_topbar_on_every_page
    PAGES.each do |p|
      doc = html(p)
      assert_equal %w[/projects/ /blog/ /about/], doc.css("#site-nav a.nav-link").map { _1["href"] }, p
      assert_equal "Seojin Park", doc.at("a.site-logo").text.strip, p
      assert doc.at("#mode-toggle"), p
      assert doc.at("#search-input"), p
    end
  end
  def test_sidebar_and_side_panel_removed
    PAGES.each { |p| assert_nil html(p).at("#sidebar"), p }
    assert_nil html("/").at("#panel-wrapper")
  end
  def test_post_keeps_toc_panel = assert(html("/data-engineering/learning-spark/05/").at("#toc-wrapper"))
  def test_blog_nav_active_on_post
    assert_equal "/blog/", html("/data-engineering/learning-spark/05/").at("#site-nav a.active")["href"]
  end
end
```

- [ ] **Step 2:** FAIL 확인
- [ ] **Step 3:** 구현. 상단바는 `position: sticky; top: 0`, 데스크톱(≥992px)에서 메뉴 가로 배치, 미만에서 `#site-nav`는 숨기고 `body[nav-open]`일 때 상단바 아래 세로 목록으로 표시. 로고·메뉴는 `--font-sans`, 로고 점은 `--accent-color`. 푸터는 `© <연도> Seojin Park` + GitHub·이메일 아이콘(`site.social.links[1]`, `site.social.email`).
- [ ] **Step 4:** 테스트 전체 PASS(`test/urls_test.rb` 포함), `npm run build`, `npm test` PASS
- [ ] **Step 5:** 수동 확인: 데스크톱·모바일(390px) 폭에서 메뉴, 햄버거, 검색 입력·결과 표시, 테마 전환 동작
- [ ] **Step 6:** 커밋 `feat: replace sidebar with top navigation bar`

---

### Task 3: 글 페이지 스타일

**Files:**
- Create: `_includes/post-path.html`
- Modify: `_layouts/post.html` (제목 위 위치 표시, 메타 줄 `.mono`), `_includes/toc.html` (제목 `// 목차`)
- Modify: `_sass/pages/_post.scss`
- Create: `test/post_test.rb`

**Interfaces:**
- Produces: `{% include post-path.html categories=page.categories %}` → `<div class="post-path mono">blog / a / b</div>`. 각 단계는 `category | slugify`. Task 5에서 `prefix="projects"`, `categories=` 대신 `segments=` 배열로 재사용한다(파라미터: `prefix`(기본 `blog`), `segments`(기본 slugify된 categories)).

- [ ] **Step 1:** 실패 테스트

```ruby
class PostTest < Minitest::Test
  SPARK = "/data-engineering/learning-spark/05/"
  def test_path_two_levels = assert_equal("blog / data-engineering / spark", html(SPARK).at(".post-path").text.strip)
  def test_path_one_level   # Review Focus 1
    url = File.read(Dir["_posts/git/*.md"].first)[/^permalink:\s*(\S+)/, 1]
    assert_equal "blog / git", html(url).at(".post-path").text.strip
  end
  def test_toc_heading = assert_equal("// 목차", html(SPARK).at("#toc-wrapper .panel-heading").text.strip)
  def test_meta_is_mono = assert(html(SPARK).at(".post-meta.mono"))
end
```

- [ ] **Step 2:** FAIL 확인
- [ ] **Step 3:** 구현. 본문 폭 최대 760px, 본문 `line-height: 1.8`, 코드 블록 배경 `--surface-bg`, 목차 활성 항목 `--accent-color`.
- [ ] **Step 4:** 테스트 PASS, `npm run build`
- [ ] **Step 5:** 수동 확인: 수식 글(`math: true`), Mermaid 글, 코드 복사, 이미지 확대, 이전·다음 글, 댓글이 다크·라이트에서 정상
- [ ] **Step 6:** 커밋 `feat: restyle post page with path and mono meta`

---

### Task 4: 블로그 목록과 카테고리 트리

**Files:**
- Modify: `_config.yml` (`paginate_path: "/blog/page:num/"`)
- Create: `blog/index.html` (front matter `layout: blog`), `_layouts/blog.html`, `_includes/category-tree.html`, `_includes/post-item.html`
- Modify: `_includes/js-selector.html` (`blog` 레이아웃 → `home` JS 번들과 dayjs 사용)
- Create: `_sass/pages/_blog.scss`, Modify: `_sass/pages/_index.scss`, `_archives.scss`, `_categories.scss`, `_category-tag.scss` (새 색·`.mono` 적용)
- Create: `test/blog_test.rb`

**Interfaces:**
- Consumes: `.mono`, `.section-label` (Task 1)
- Produces:
  - `{% include post-item.html post=post %}` → `article.post-item` 안에 `a.post-title`, (description이 있을 때만) `p.post-desc`, `div.post-item-meta.mono`(날짜 `YYYY.MM.DD` · `categories | join: '/'`). Task 7에서 재사용.
  - `{% include category-tree.html %}` → `nav.category-tree` 안에 상위마다 `details.tree-node` > `summary`(`span.tree-arrow`, `a.tree-link`(href = `/categories/{{ name | slugify | url_encode }}/`), `span.tree-count`), 하위는 `ul > li > a.tree-link + span.tree-count`. 상위는 글 수 내림차순, 하위는 이름순. 상위 맨 위에 `all`(`/blog/`, 전체 글 수).
  - 블로그 목록은 `hidden: true`인 글을 제외하고, `pin: true` 글을 먼저 보여주는 기존 `home.html` 로직을 그대로 옮긴다.
  - 모바일(<992px): 트리를 `details.tree-mobile`(기본 닫힘, `summary`는 `카테고리`)로 감싼다.

- [ ] **Step 1:** 실패 테스트

```ruby
class BlogTest < Minitest::Test
  def test_first_page_has_ten_posts = assert_equal(10, html("/blog/").css("article.post-item").size)
  def test_pagination_moved
    assert site_file?("/blog/page2/")
    refute site_file?("/page2/")
  end
  def test_tree_counts
    tree = html("/blog/").at("nav.category-tree")
    total = tree.at(".tree-node:first-child .tree-count").text.to_i
    assert_equal Dir["_posts/**/*.md"].size, total
    assert_equal "/blog/", tree.at(".tree-node:first-child a.tree-link")["href"]
  end
  def test_tree_links_resolve        # Review Focus 3
    html("/blog/").css("nav.category-tree a.tree-link").each { assert site_file?(_1["href"]), _1["href"] }
  end
  def test_single_category_top_has_no_children   # Review Focus 1
    node = html("/blog/").css(".tree-node").find { _1.at("a.tree-link").text.strip == "Git" }
    assert_nil node.at("ul")
  end
  def test_no_empty_description      # Review Focus 2
    (1..3).each { |n| p = n == 1 ? "/blog/" : "/blog/page#{n}/"
      html(p).css("p.post-desc").each { refute_empty _1.text.strip, p } }
  end
end
```

- [ ] **Step 2:** FAIL 확인
- [ ] **Step 3:** 구현. 데스크톱은 `grid-template-columns: 220px 1fr`, 페이지 상단 `.section-label` `./blog`, 제목 `Blog` + `.mono` `<N> posts`. 페이지 나눔은 기존 `post-paginator.html`을 새 경로로 재사용. 이 시점에 `/`는 아직 옛 홈(Task 7에서 교체)이지만, root가 더 이상 페이지 나눔 대상이 아니므로 `home.html`의 paginator 의존 부분이 비어도 빌드가 깨지지 않게 `site.posts | slice: 0, 10`으로 임시 대체한다.
- [ ] **Step 4:** 테스트 전체 PASS, `npm run build`, `npm test`
- [ ] **Step 5:** 수동 확인: 트리 접기·펴기, 모바일 접이식 메뉴, `/categories/`, `/tags/`, `/archives/`, 카테고리 상세 페이지가 새 색으로 보임
- [ ] **Step 6:** 커밋 `feat: add blog list page with category tree`

---

### Task 5: 프로젝트 컬렉션·상세·목록 페이지

**Files:**
- Modify: `_config.yml` (`collections.projects: {output: true, permalink: /projects/:name/}`, `defaults`에 `type: projects` → `layout: project, toc: true, comments: false`)
- Create: `_projects/sample-project.md`, `_projects/sample-solo.md`, `_posts/projects/sample-project/2026-01-01-sample-devlog.md` (모두 `published: false`)
- Create: `_layouts/project.html`, `_layouts/projects.html`, `projects.md` (`layout: projects`, `permalink: /projects/`), `_includes/project-card.html`, `_includes/project-thumb.html`
- Modify: `_includes/js-selector.html` (`project` → `post` JS 번들, toc·mermaid·clipboard·glightbox 포함), `_layouts/default.html`의 `main>article[data-toc]` 조건이 `project`에도 맞게
- Create: `_sass/pages/_project.scss`
- Create: `test/project_test.rb`

**Interfaces:**
- Consumes: `post-path.html` (`prefix="projects"`, `segments`=[파일 이름]), `post-item.html` (Task 4)
- Produces:
  - 샘플 데이터(테스트가 이 값에 의존):
    - `sample-project`: `title: 샘플 프로젝트`, `summary: 테스트용 프로젝트`, `period: 2026.01 – 2026.02`, `role: [역할 하나, 역할 둘]`, `stack: [Python, FastAPI]`, `team: 2인 팀`, `links: {github: https://github.com/Parkseojin2001, demo: https://example.com}`, `featured: true`, `order: 1`, 본문에 `## 문제 정의`, `## 결과`.
    - `sample-solo`: `title: 샘플 솔로`, `summary: 선택 항목이 없는 프로젝트`, `period: 2026.03`, `role: [혼자 개발]`, `stack: [Ruby]`, `featured: true`, `order: 2`. `thumbnail`·`team`·`links` 없음.
    - devlog: `title: 샘플 개발 기록`, `categories: [Projects, 샘플 프로젝트]`, `project: sample-project`, `permalink: /projects/sample-project/sample-devlog/`.
  - `{% include project-thumb.html project=p %}` → `thumbnail`이 있으면 `img.project-thumb`, 없으면 `div.project-thumb.fallback` 안에 `title`의 첫 글자.
  - `{% include project-card.html project=p %}` → `a.project-card`(href=`p.url`) 안에 thumb, `h3`, `p.project-summary`, `div.project-stack.mono`(stack을 ` · `로 연결). Task 7에서 재사용.
  - 상세 레이아웃: `article[data-toc]` 본문 + `panel_includes: [project-meta, toc]`. `_includes/project-meta.html` → `aside.project-meta` 안에 `dl`: `dt.mono` `PERIOD`/`TEAM`/`MY ROLE`/`STACK`, `dd`; 링크는 `div.project-links > a.btn-link-pill`(텍스트 = 키 이름 + ` ↗`, `target="_blank" rel="noopener"`). 값이 없는 항목은 `dt`·`dd` 모두 생략.
  - devlog: `site.posts | where: "project", page.slug | sort: "date"` → 결과가 있으면 `section.devlog` 안에 `.section-label` `./devlog` + `post-item` 목록, 없으면 섹션 자체를 출력하지 않음.
  - 목록 페이지: `site.projects | sort: "order"`(order 없으면 99로 간주) → `div.project-grid > project-card`.
  - 모바일(<1200px): `aside.project-meta`를 제목 아래로 옮겨 카드로 표시(CSS 순서 변경, 중복 렌더링 금지).

- [ ] **Step 1:** 샘플 파일 3개 작성 후 실패 테스트

```ruby
class ProjectTest < Minitest::Test
  P = "/projects/sample-project/"
  def meta(path = P) = html(path).at("aside.project-meta")
  def test_meta_fields
    assert_equal %w[PERIOD TEAM MY\ ROLE STACK], meta.css("dt").map { _1.text.strip }
    assert_equal ["역할 하나", "역할 둘"], meta.css("dd li").map { _1.text.strip }
    assert_equal ["github ↗", "demo ↗"], meta.css(".project-links a").map { _1.text.strip }
  end
  def test_devlog_listed
    sec = html(P).at("section.devlog")
    assert_equal "./devlog", sec.at(".section-label").text.strip
    assert_equal "/projects/sample-project/sample-devlog/", sec.at("a.post-title")["href"]
  end
  def test_optional_fields_absent    # Review Focus 4
    doc = html("/projects/sample-solo/")
    assert_nil doc.at("section.devlog")
    assert_equal %w[PERIOD MY\ ROLE STACK], meta("/projects/sample-solo/").css("dt").map { _1.text.strip }
    assert_nil doc.at(".project-links")
    assert_equal "샘", html("/projects/").css("a.project-card")[1].at(".project-thumb.fallback").text.strip
  end
  def test_list_sorted_by_order
    assert_equal %w[/projects/sample-project/ /projects/sample-solo/], html("/projects/").css("a.project-card").map { _1["href"] }
  end
  def test_path_and_toc
    assert_equal "projects / sample-project", html(P).at(".post-path").text.strip
    assert html(P).at("#toc-wrapper")
  end
end
```

- [ ] **Step 2:** FAIL 확인
- [ ] **Step 3:** 구현 (스펙 5장 "프로젝트 상세" 배치: 오른쪽 고정 영역 `position: sticky; top: <상단바 높이 + 1rem>`)
- [ ] **Step 4:** 테스트 전체 PASS, `npm run build`, `npm test`
- [ ] **Step 5:** 수동 확인: `bundle exec jekyll serve --unpublished`로 상세 페이지 데스크톱·모바일, 다크·라이트
- [ ] **Step 6:** 커밋 `feat: add projects collection with detail and list pages`

---

### Task 6: 글 → 프로젝트 링크와 검사 플러그인

**Files:**
- Create: `_includes/project-backlink.html`, Modify: `_layouts/post.html` (위치 표시 위에 include)
- Create: `_plugins/project-validator.rb`
- Create: `test/project_validator_test.rb`, Modify: `test/project_test.rb`

**Interfaces:**
- Consumes: 샘플 데이터 (Task 5)
- Produces:
  - `project-backlink.html`: `page.project`가 있고 같은 이름의 프로젝트가 있으면 `div.project-backlink.mono` → `↳ project: <a href="{{ p.url }}">{{ p.title }}</a>`.
  - `ProjectValidator::REQUIRED = %w[title summary period role stack]`
  - `ProjectValidator.problems(projects:, posts:) -> Array<String>`
    - `projects`: `[{ "path" => String, "name" => String, "data" => Hash }]`
    - `posts`: `[{ "path" => String, "project" => String|nil }]`
    - 메시지 형식(정확히): `"#{path}: 필수 항목 '#{key}' 이(가) 비어 있습니다"`, `"#{path}: project '#{name}' 에 해당하는 _projects 파일이 없습니다"`
    - 비어 있음 = 키 없음, `nil`, 공백 문자열, 빈 배열, 빈 문자열만 든 배열.
  - 훅: `Jekyll::Hooks.register :site, :post_read` → `problems`의 각 항목을 `Jekyll.logger.warn "ProjectValidator:", msg`.

- [ ] **Step 1:** 실패 테스트

```ruby
class ProjectValidatorTest < Minitest::Test
  FULL = { "title" => "t", "summary" => "s", "period" => "p", "role" => ["r"], "stack" => ["x"] }
  def prj(data) = [{ "path" => "_projects/a.md", "name" => "a", "data" => data }]
  def test_valid = assert_empty(ProjectValidator.problems(projects: prj(FULL), posts: []))
  def test_missing_and_blank
    msgs = ProjectValidator.problems(projects: prj(FULL.merge("summary" => " ", "role" => [""]).except("stack")), posts: [])
    assert_equal ["_projects/a.md: 필수 항목 'summary' 이(가) 비어 있습니다",
                  "_projects/a.md: 필수 항목 'role' 이(가) 비어 있습니다",
                  "_projects/a.md: 필수 항목 'stack' 이(가) 비어 있습니다"], msgs
  end
  def test_unknown_project
    msgs = ProjectValidator.problems(projects: prj(FULL), posts: [{ "path" => "_posts/x.md", "project" => "typo" }])
    assert_equal ["_posts/x.md: project 'typo' 에 해당하는 _projects 파일이 없습니다"], msgs
  end
  def test_real_build_has_no_warnings = refute_includes(SiteBuild.log, "ProjectValidator")
end
```

`test/project_test.rb`에 추가:

```ruby
def test_backlink_on_devlog
  link = html("/projects/sample-project/sample-devlog/").at(".project-backlink a")
  assert_equal ["/projects/sample-project/", "샘플 프로젝트"], [link["href"], link.text.strip]
end
def test_no_backlink_on_normal_post = assert_nil(html("/data-engineering/learning-spark/05/").at(".project-backlink"))
```

- [ ] **Step 2:** FAIL 확인 (`ProjectValidator` 미정의). `test/test_helper.rb`가 `_plugins/project-validator.rb`를 require하도록 추가.
- [ ] **Step 3:** 구현
- [ ] **Step 4:** 테스트 전체 PASS. 추가 확인: 임시로 `sample-solo.md`의 `summary`를 지우고 `bundle exec jekyll build --unpublished` 출력에 경고 1줄이 뜨는지 보고 되돌린다.
- [ ] **Step 5:** 커밋 `feat: link devlog posts to projects and validate project data`

---

### Task 7: 새 홈

**Files:**
- Modify: `_layouts/home.html` (전체 교체), `index.html` (front matter만 유지)
- Create: `_data/now.yml`, `_sass/pages/_home.scss`
- Create: `test/home_test.rb`

**Interfaces:**
- Consumes: `project-card.html` (Task 5), `post-item.html` (Task 4), `.section-label` (Task 1)
- Produces: 섹션 순서와 DOM
  1. `section.hero`: `.section-label` `HELLO, I'M SEOJIN`, `h1`(`site.tagline`), `p`(`site.description`에서 따옴표 제거), `div.hero-tags.mono`(`#AI #ML #DataEngineering`)
  2. `section#home-projects`: 라벨 `./projects`, `featured: true` 프로젝트를 `order` 순으로 최대 3개 `project-card`, 오른쪽에 `a.more`(`/projects/`)
  3. `section#home-now`: 라벨 `./now`, `site.data.now`의 각 항목 → `div.now-item` 안에 `span.now-label.mono`, `span.now-text`(link 있으면 `a`로 감쌈)
  4. `section#home-posts`: 라벨 `./blog`, `hidden`이 아닌 최근 글 5개 `post-item`, `a.more`(`/blog/`)
  5. `section#home-categories`: 라벨 `./categories`, 상위 카테고리를 글 수 내림차순으로 `a.category-chip`(href = `/categories/{{ name | slugify | url_encode }}/`, 텍스트 `이름 + span.mono 글수`)
- `_data/now.yml` 초기값:

```yaml
- label: 공부 중
  text: Spark로 데이터 엔지니어링
  link: /categories/spark/
- label: 관심사
  text: LLM Agent와 RAG
```

- [ ] **Step 1:** 실패 테스트

```ruby
class HomeTest < Minitest::Test
  def doc = @doc ||= html("/")
  def test_section_order
    assert_equal ["HELLO, I'M SEOJIN", "./projects", "./now", "./blog", "./categories"],
                 doc.css("main .section-label").map { _1.text.strip }
  end
  def test_featured_projects = assert_equal(%w[/projects/sample-project/ /projects/sample-solo/], doc.css("#home-projects a.project-card").map { _1["href"] })
  def test_now_items = assert_equal(["Spark로 데이터 엔지니어링", "LLM Agent와 RAG"], doc.css("#home-now .now-text").map { _1.text.strip })
  def test_recent_posts = assert_equal(5, doc.css("#home-posts article.post-item").size)
  def test_category_chips_resolve       # Review Focus 3
    chips = doc.css("#home-categories a.category-chip")
    refute_empty chips
    chips.each { assert site_file?(_1["href"]), _1["href"] }
  end
end
```

- [ ] **Step 2:** FAIL 확인
- [ ] **Step 3:** 구현 (프로젝트 카드 그리드: 3열 → 태블릿 2열 → 모바일 1열)
- [ ] **Step 4:** 테스트 전체 PASS, `npm run build`, `npm test`
- [ ] **Step 5:** 수동 확인: 홈 데스크톱·모바일, 다크·라이트. 프로젝트가 하나도 없을 때(샘플 제외 = 일반 `jekyll serve`) `#home-projects` 섹션이 통째로 숨겨지는지 확인
- [ ] **Step 6:** 커밋 `feat: add new home page with projects, now, posts and categories`

---

### Task 8: About 틀

**Files:**
- Modify: `_tabs/about.md`
- Create: `test/about_test.rb`

**Interfaces:**
- Produces: `## 소개`, `## 기술 스택`, `## 경력 · 활동`, `## 연락처` 네 섹션. 각 섹션 아래 `<!-- 여기에 작성 -->` 주석과 예시 한 줄. 연락처에는 `site.social.email`, GitHub 링크를 미리 채운다.

- [ ] **Step 1:** 실패 테스트: `html("/about/").css("main h2").map(&:text)`가 위 네 제목과 같다.
- [ ] **Step 2:** FAIL → 구현 → PASS
- [ ] **Step 3:** 커밋 `feat: add about page skeleton`

---

### Task 9: 글 생성 스크립트 `tools/new.sh`

**Files:**
- Create: `tools/new.sh`, `tools/templates/post.md`, `tools/templates/devlog.md`, `tools/templates/project.md`
- Create: `test/new_script_test.rb`

**Interfaces:**
- 실행: `bash tools/new.sh`. 스크립트 시작 시 `cd "$(dirname "$0")/.."`로 저장소 루트로 이동한다(Review Focus 5).
- 날짜: 환경변수 `NEW_TODAY`(YYYY-MM-DD)가 있으면 그 값, 없으면 `date +%F`.
- 입력 순서(stdin 한 줄씩):
  1. 종류 `1|2|3`
  2. (1) 카테고리 폴더: `_posts` 하위 폴더 목록(`projects/` 제외, 파일이 있는 가장 깊은 폴더들, 정렬)을 번호로 보여주고 번호 또는 새 경로(`a/b`) 입력 / (2) 프로젝트: `_projects/*.md` 목록(`이름  (title)`)에서 번호 / (3) 없음
  3. 파일 이름: `^[a-z0-9]+(-[a-z0-9]+)*$`. 위반 시 `  ✗ 영어 소문자, 숫자, 하이픈만 사용할 수 있습니다` 출력 후 다시 입력.
- 생성 경로:

| 종류 | 글 파일 | 이미지 폴더 | permalink |
|---|---|---|---|
| 1 | `_posts/<폴더>/<날짜>-<이름>.md` | `assets/img/post/<폴더>/` | `/<폴더>/<이름>/` |
| 2 | `_posts/projects/<프로젝트>/<날짜>-<이름>.md` | `assets/img/post/projects/<프로젝트>/` | `/projects/<프로젝트>/<이름>/` |
| 3 | `_projects/<이름>.md` | `assets/img/projects/<이름>/` | (없음) |

- 템플릿 치환 자리: `{{DATE}}`, `{{PERMALINK}}`, `{{PROJECT}}`, `{{NAME}}`. 템플릿의 front matter는 스펙 6장의 형식 그대로이며, 각 키 끝에 `# [필수] ...` / `# [선택] ...` 안내 주석을 단다. 프로젝트 템플릿은 `toc: true`, `math: false`, `mermaid: true`, `comments: false`와 본문 뼈대(`## 문제 정의`, `## 아키텍처`, `## 내가 한 일`, `## 결과`, 각 아래 안내 주석)를 포함한다.
- 출력(성공, 종료 코드 0). 정확히 이 접두사로 시작하는 줄들:
  - `  ✓ 글 파일      <경로>`
  - `  ✓ 이미지 폴더  <경로>/   (새로 만듦)` 또는 `(이미 있음)`
  - `  ✓ 주소         <URL>` (종류 3은 `/projects/<이름>/`)
  - 종류 2만: `  ✓ 연결         project: <프로젝트> → 프로젝트 페이지 ./devlog에 자동 표시`
  - 이어서 `  다음 할 일` 블록: 채울 항목과 이미지 사용 예시 `![설명](/<이미지 폴더>/파일명.png)`
- 실패:
  - 파일이 이미 있음 → `  ✗ 이미 있는 파일입니다: <경로>`, 종료 코드 1, 아무것도 만들지 않음
  - 종류 2인데 `_projects/*.md` 없음 → `  ✗ 프로젝트가 없습니다. 먼저 3) 프로젝트를 만드세요`, 종료 코드 1

- [ ] **Step 1:** 실패 테스트. 각 테스트는 임시 폴더에 `tools/`를 복사하고 `_posts/data-engineering/spark/x.md`, `_projects/rag-chatbot.md`(`title: RAG 챗봇`)를 만든 뒤 `Open3.capture2e({"NEW_TODAY"=>"2026-11-07"}, "bash", "tools/new.sh", stdin_data:, chdir:)`로 실행한다.

```ruby
def test_post        # 입력 "1\n1\nspark-06\n"
  # 파일 _posts/data-engineering/spark/2026-11-07-spark-06.md 존재, permalink: /data-engineering/spark/spark-06/
  # 출력에 "✓ 이미지 폴더  assets/img/post/data-engineering/spark/   (새로 만듦)"
end
def test_devlog      # 입력 "2\n1\nvector-db\n"
  # 파일 _posts/projects/rag-chatbot/2026-11-07-vector-db.md, front matter에 "project: rag-chatbot"
  # 출력에 "✓ 연결         project: rag-chatbot"
end
def test_project     # 입력 "3\nmy-app\n"
  # _projects/my-app.md 에 "title:", "role:", "stack:", "# [필수]", "mermaid: true", "## 내가 한 일" 포함
  # 디렉터리 assets/img/projects/my-app/ 존재, 출력에 "✓ 주소         /projects/my-app/"
end
def test_existing_image_dir_reported   # 같은 폴더에 두 번째 글 → "(이미 있음)"
def test_invalid_name_reprompts        # 입력 "3\nMy App\nmy-app\n" → 출력에 "✗ 영어 소문자", 파일 생성됨
def test_refuses_overwrite             # 같은 입력 두 번 → 두 번째 종료 코드 1, "✗ 이미 있는 파일입니다", 파일 내용 변화 없음
def test_devlog_without_projects       # _projects 비우고 "2\n" → 종료 코드 1, "✗ 프로젝트가 없습니다"
def test_runs_from_subdirectory        # Review Focus 5: chdir: "<tmp>/_posts", 명령 "bash ../tools/new.sh", 입력 "3\nsub-app\n" → <tmp>/_projects/sub-app.md 존재
def test_generated_project_builds_clean
  # 실제 저장소 대상 아님: 생성된 project.md의 front matter를 YAML.safe_load 했을 때 오류 없음
end
```

- [ ] **Step 2:** FAIL 확인
- [ ] **Step 3:** 구현 (bash 3.2 호환: 연관 배열·`mapfile` 사용 금지, `read -r`, `set -eu`)
- [ ] **Step 4:** 테스트 PASS
- [ ] **Step 5:** 수동 확인: 실제 터미널에서 세 종류를 한 번씩 실행해 출력이 깔끔한지 확인하고 생성 파일은 삭제
- [ ] **Step 6:** 커밋 `feat: add interactive new content script`

---

### Task 10: 최종 점검과 병합

- [ ] **Step 1:** `npm ci && npm run build` 후 `bundle exec ruby test/run.rb` 전체 PASS
- [ ] **Step 2:** `npm test` (stylelint) PASS
- [ ] **Step 3:** `JEKYLL_ENV=production bundle exec jekyll build -d /tmp/final-site && bundle exec htmlproofer /tmp/final-site --disable-external --ignore-urls "/^http/"` → 내부 링크·이미지 오류 0
- [ ] **Step 4:** 일반 빌드(`--unpublished` 없음) 결과에 `sample-project`, `sample-solo`, `sample-devlog`가 없음을 `find /tmp/final-site -path '*sample*'` 결과 0줄로 확인
- [ ] **Step 5:** 사용자 확인: `bundle exec jekyll serve`로 홈·프로젝트·블로그·글·About·카테고리·태그·아카이브·검색을 데스크톱·모바일, 다크·라이트에서 함께 확인
- [ ] **Step 6:** 사용자 승인 후 `main`에 병합하고 push (`superpowers:finishing-a-development-branch`)
