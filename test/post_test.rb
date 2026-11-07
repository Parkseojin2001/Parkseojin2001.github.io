require_relative "test_helper"

class PostTest < Minitest::Test
  SPARK = "/data-engineering/learning-spark/05/"

  def test_path_two_levels
    assert_equal "blog / data-engineering / spark", html(SPARK).at(".post-path")&.text&.strip
  end

  # Review Focus 1: a post with a single category
  def test_path_one_level
    url = File.read(Dir["_posts/git/*.md"].first)[/^permalink:\s*(\S+)/, 1]
    assert_equal "blog / git", html(url).at(".post-path")&.text&.strip
  end

  def test_toc_heading
    assert_equal "// 목차", html(SPARK).at("#toc-wrapper .panel-heading")&.text&.strip
  end

  def test_meta_is_mono
    assert html(SPARK).at(".post-meta.mono")
  end

  # 좁은 화면에서 글 위에 나오는 목차 버튼은 "//" 없이 "목차"
  def test_mobile_toc_button_label
    assert_equal "목차", html(SPARK).at("#toc-solo-trigger .label")&.text&.strip
  end

  def test_mobile_toc_button_smaller
    css = File.read(File.join(SiteBuild.dir, "assets/css/jekyll-theme-chirpy.css"))
    size = css[/#toc-solo-trigger \.label\{font-size:([^;]+);/, 1]
    assert_equal ".9rem", size
  end
end
