require_relative "test_helper"

class StyleTest < Minitest::Test
  def css
    @css ||= File.read(File.join(SiteBuild.dir, "assets/css/jekyll-theme-chirpy.css"))
  end

  def assert_css(fragment)
    assert css.include?(fragment), "CSS does not contain #{fragment}"
  end

  def test_dark_palette
    assert_css "#0d1117"
    assert_css "#7ee787"
  end

  def test_light_palette
    assert_css "#1a7f37"
  end

  def test_fonts_loaded
    links = html("/").css("head link[rel=stylesheet]").map { _1["href"] }
    assert links.any? { _1.include?("JetBrains+Mono") }, "JetBrains Mono not linked"
    assert links.any? { _1.include?("pretendard") }, "Pretendard not linked"
  end

  def test_mono_utility
    assert css.match?(/\.mono\{[^}]*font-family:var\(--font-mono\)/), ".mono utility missing"
  end
end
