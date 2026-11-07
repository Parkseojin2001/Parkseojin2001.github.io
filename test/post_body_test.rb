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
end
