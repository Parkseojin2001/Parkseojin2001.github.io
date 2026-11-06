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
end
