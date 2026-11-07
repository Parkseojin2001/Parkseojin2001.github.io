require_relative "test_helper"

# 카테고리·태그·아카이브·검색·페이지 넘김·404·작은 요소를 v5 스타일로 바꾼 결과를 고정한다
class PagesTest < Minitest::Test
  def css
    @css ||= File.read(File.join(SiteBuild.dir, "assets/css/jekyll-theme-chirpy.css"))
  end

  def label(path) = html(path).at("main .page-label")&.text&.strip

  # 1. 페이지 제목: ./경로 라벨 + 굵은 제목 + 개수
  def test_page_titles
    assert_equal "./tags", label("/tags/")
    assert_equal "./categories", label("/categories/")
    assert_equal "./archives", label("/archives/")
    assert_equal "./about", label("/about/")
    assert_match(/\A\d+ tags\z/, html("/tags/").at("main .page-count").text.strip)
    assert_match(/\A\d+ posts\z/, html("/archives/").at("main .page-count").text.strip)
  end

  # 2. 카테고리 페이지: 상위 카테고리 카드 + 하위 트리
  def test_categories_page_cards
    doc = html("/categories/")
    cards = doc.css(".category-card")
    refute_empty cards
    names = cards.map { _1.at(".category-card-title").text.strip }
    assert_equal names.sort_by(&:downcase), names
    dl = cards.find { _1.at(".category-card-title").text.strip == "Deep Learning" }
    subs = dl.css(".category-sub").map { _1.text.split.first }
    assert_equal ["├─", "├─", "└─"], subs
    doc.css(".category-card a").each { |a| assert site_file?(a["href"]), a["href"] }
    assert_nil doc.at(".fa-folder, .fa-folder-open")
  end

  # 3. 카테고리·태그 상세: 블로그 목록과 같은 형식
  def test_category_detail
    doc = html("/categories/spark/")
    assert_equal "./categories/spark", label("/categories/spark/")
    assert_equal "Spark", doc.at("main h1 .page-name").text.strip
    assert_equal 5, doc.css("article.post-item").size
    assert_nil doc.at(".dash")
  end

  def test_tag_detail
    doc = html("/tags/spark/")
    assert_equal "./tags/spark", label("/tags/spark/")
    assert_equal "#spark", doc.at("main h1 .page-name").text.strip
    refute_empty doc.css("article.post-item")
  end

  # 4. 태그 페이지: #태그 개수, 많이 쓴 순, 진하기 차이 없음
  def test_tags_page
    links = html("/tags/").css("a.tag-link")
    refute_empty links
    assert links.all? { _1.at(".tag-name").text.start_with?("#") }
    counts = links.map { _1.at(".tag-count").text.to_i }
    assert_equal counts.sort.reverse, counts
    assert links.none? { _1["style"].to_s.include?("opacity") }
  end

  # 5. 아카이브: 연도 → 월 묶음, MM.DD 날짜
  def test_archives_grouped_by_month
    doc = html("/archives/")
    refute_empty doc.css(".archive-year")
    assert doc.css(".archive-month").all? { _1.text.strip.match?(/\A── \d{2}\z/) }
    assert doc.css(".archive-date").all? { _1.text.strip.match?(/\A\d{2}\.\d{2}\z/) }
  end

  # 6. 검색 결과: $ grep 머리줄 + 블로그 목록 형식 결과
  def test_search_results
    doc = html("/")
    assert doc.at("#search-result-wrapper #search-grep")
    assert_includes doc.to_html, "post-item"
  end

  # 7. 블로그 페이지 넘김
  def test_paginator
    nav = html("/blog/page2/").at("nav.blog-pager")
    refute_nil nav
    assert_equal "← prev", nav.at(".pager-prev").text.strip
    assert_equal "next →", nav.at(".pager-next").text.strip
    assert_equal "2", nav.at(".pager-num.current").text.strip
  end

  # 8. 404
  def test_not_found_page
    text = html("/404.html").at("main").text
    assert_includes text, "No such file or directory"
    assert_includes text, "페이지를 찾을 수 없습니다"
    hrefs = html("/404.html").css("main a").map { _1["href"] }
    assert_includes hrefs, "/"
    assert_includes hrefs, "/study/"
    assert_includes hrefs, "/devlog/"
  end

  # 9. 작은 요소: 맨 위로 가기, 모바일 목차 팝업, 업데이트 알림, 이미지 확대 배경
  def test_small_elements
    assert_equal "↑ top", html("/").at("#back-to-top").text.strip
    assert css.match?(/#toc-popup\{[^}]*border:1px solid var\(--main-border-color\)/), "목차 팝업 v5 상자 아님"
    assert css.match?(/#notification\{[^}]*background-color:var\(--surface-bg\)/), "업데이트 알림 v5 상자 아님"
    assert css.match?(/\.goverlay\{background:var\(--main-bg\)/), "이미지 확대 배경 미적용"
  end

  # 검색 결과: Chirpy 기본의 두 칸 배치(홀짝 여백)와 큰 제목을 쓰지 않는다
  def test_search_results_single_column
    assert css.match?(/#search-results>article\{[^}]*width:100%[^}]*margin:0/), "검색 결과가 한 줄 배치가 아님"
    assert css.match?(/#search-results a\.post-title\{[^}]*font-size:1rem/), "검색 결과 제목 크기 미조정"
    assert_includes html("/").to_html, "replaceAll(', ', ' / ')"
  end

  # 블로그에서는 왼쪽 탐색 영역(카테고리 트리 위)에 목록 링크를 모은다
  def test_blog_links_to_index_pages
    doc = html("/blog/")
    hrefs = doc.css("aside.blog-tree nav.blog-index-links a").map { _1["href"] }
    assert_equal %w[/categories/ /tags/ /archives/], hrefs
    assert_equal 1, doc.css("nav.blog-index-links").size, "목록 링크는 한 곳에만"
  end

  # 다른 페이지에서도 푸터로 목록 페이지에 갈 수 있어야 한다
  def test_footer_links_to_index_pages
    ["/", "/data-engineering/learning-spark/05/", "/about/"].each do |p|
      hrefs = html(p).css("footer nav.footer-index-links a").map { _1["href"] }
      assert_equal %w[/categories/ /tags/ /archives/], hrefs, p
    end
  end

  # 둥근 알약 모양 대신 모서리가 살짝 둥근 상자
  def test_no_pill_shapes
    refute css.include?("border-radius:999px"), "알약 모양(999px)이 남아 있음"
  end

  # About 대표 이미지: Chirpy가 이미지를 a.img-link로 감싸는데, 좁은 화면에서 이 링크가
  # 가로로 늘어나 로딩 배경(shimmer)이 어두운 띠로 보이면 안 된다
  def test_about_avatar_link_not_stretched
    assert css.match?(/\.about-profile>a\.img-link\{[^}]*align-self:flex-start/), "About 이미지 링크가 가로로 늘어남"
    assert css.match?(/\.about-profile>a\.img-link\{[^}]*background:none/), "About 이미지 링크 배경 남음"
  end

  # 이미지 확대 팝업 번호 표시 스크립트가 글 페이지에 들어간다
  def test_lightbox_counter_on_post
    assert html("/pytorch-book/nlp/embedding-2/").at("script#lightbox-counter")
  end
end
