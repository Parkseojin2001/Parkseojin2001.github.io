require_relative "test_helper"

# 글 본문과 글 하단을 v5 스타일(B안: 제목 앞 ## / ###)로 바꾼 결과를 고정한다
class PostBodyTest < Minitest::Test
  SPARK = "/data-engineering/learning-spark/05/"

  def css
    @css ||= File.read(File.join(SiteBuild.dir, "assets/css/jekyll-theme-chirpy.css"))
  end

  def doc
    @doc ||= html(SPARK)
  end

  def assert_css(pattern, message)
    assert css.match?(pattern), message
  end

  def test_heading_markers
    assert_css(/\.content h2::before\{content:"##"/, "h2 앞 ## 표시 없음")
    assert_css(/\.content h3::before\{content:"###"/, "h3 앞 ### 표시 없음")
    assert_css(/\.content h4::before\{content:"####"/, "h4 앞 #### 표시 없음")
  end

  def test_code_header_without_window_dots
    assert_css(/\.code-header::before\{display:none/, "코드 헤더 점(● ● ●)이 남아 있음")
    assert_css(/\.code-header span::after\{[^}]*text-transform:lowercase/, "코드 언어 라벨이 소문자 고정폭이 아님")
  end

  def test_prompt_labels
    %w[TIP INFO WARNING DANGER].each do |label|
      assert_css(/content:"#{label}"/, "#{label} 라벨 없음")
    end
  end

  def test_tail_category_chip_and_tags
    chip = doc.at(".post-tail-wrapper a.post-chip")
    refute_nil chip
    assert_equal "Data Engineering / Spark", chip.text.strip
    assert_equal "/categories/spark/", chip["href"]
    tags = doc.css(".post-tail-wrapper a.post-tag-link").map { _1.text.strip }
    assert_includes tags, "#Spark"
  end

  def test_license_line
    line = doc.at(".post-tail-wrapper .post-license")
    refute_nil line
    assert_includes line.text, "CC BY 4.0"
    assert_includes line.text, "Seojin Park"
  end

  def test_post_nav_labels
    labels = doc.css("nav.post-navigation .post-nav-label").map { _1.text.strip }
    assert_equal ["← older", "newer →"], labels
  end

  def test_related_posts_as_list
    aside = doc.at("aside#related-posts")
    refute_nil aside
    assert_equal "./related", aside.at(".section-label").text.strip
    refute_empty aside.css("article.post-item")
  end

  def test_comments_label
    assert_includes doc.to_html, "comments-label"
    assert_includes doc.to_html, "./comments"
  end

  # 글 하단 순서: older/newer → ./related → ./comments
  def test_tail_order_nav_then_related
    first = doc.css("nav.post-navigation, aside#related-posts").first
    assert_equal "nav", first.name, "older/newer 링크가 ./related보다 먼저 나와야 함"
  end

  # md 본문 요소 (제안안)
  def test_list_markers
    assert_css(/\.content ul:not\(\.task-list\)>li::before\{content:"-"/, "목록 기호 - 없음")
    assert_css(/\.content ol>li::before\{content:counter\(md-item, decimal-leading-zero\) "\."/, "번호 목록 01. 형식 없음")
  end

  def test_bold_highlight_and_italic_color
    assert_css(/\.content strong\{[^}]*linear-gradient/, "굵은 글씨 형광펜 없음")
    assert_css(/\.content em\{[^}]*color:var\(--accent-purple\)/, "기울임 색 없음")
  end

  def test_inline_code_terminal_style
    # 코드 블록은 <div class="highlight"><code><table>… 구조라서, .highlight 안의 code는 인라인 코드가 아니다
    assert_css(/\.content :not\(pre\):not\(\.highlight\)>code\{[^}]*color:var\(--accent-color\)/, "인라인 코드 초록 글자 없음(코드 블록 제외)")
    refute css.match?(/\.content :not\(pre\)>code\{/), "코드 블록 전체에 인라인 코드 테두리가 입혀짐"
  end

  def test_image_frame_caption_hr_math
    assert_css(/\.content img\{[^}]*border-radius:8px/, "이미지 테두리 없음")
    assert_css(/content:"fig\. "/, "fig. 캡션 없음")
    assert_css(/\.content hr\{[^}]*dashed/, "점선 구분선 없음")
    assert_css(/mjx-container\[display=true\]::before\{content:"math"/, "수식 상자 라벨 없음")
  end
end
