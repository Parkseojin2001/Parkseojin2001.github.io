require_relative "test_helper"

class ShellTest < Minitest::Test
  POST = "/data-engineering/learning-spark/05/"
  PAGES = ["/", "/about/", "/categories/", POST]

  def test_topbar_on_every_page
    PAGES.each do |p|
      doc = html(p)
      assert_equal %w[/projects/ /study/ /devlog/ /about/], doc.css("#site-nav a.nav-link").map { _1["href"] }, p
      assert_equal "Seojin Park", doc.at("a.site-logo")&.text&.strip, p
      assert doc.at("#mode-toggle"), "#{p}: mode toggle"
      assert doc.at("#search-input"), "#{p}: search"
      assert doc.at("#nav-trigger"), "#{p}: mobile nav trigger"
    end
  end

  def test_sidebar_and_side_panel_removed
    PAGES.each { |p| assert_nil html(p).at("#sidebar"), p }
    assert_nil html("/").at("#panel-wrapper")
  end

  def test_post_keeps_toc_panel
    assert html(POST).at("#toc-wrapper")
  end

  def test_study_nav_active_on_post
    assert_equal "/study/", html(POST).at("#site-nav a.active")&.[]("href")
  end

  def test_about_nav_active
    assert_equal "/about/", html("/about/").at("#site-nav a.active")&.[]("href")
  end

  # 공용 목록 페이지(Study·Devlog 글이 함께 있음)에서는 메뉴를 강조하지 않는다
  def test_index_pages_no_active_nav
    tag = Dir.children(File.join(ROOT, "test/.site/tags")).find { File.directory?(File.join(ROOT, "test/.site/tags", _1)) }
    ["/categories/", "/tags/", "/archives/", "/tags/#{tag}/"].each do |p|
      assert_nil html(p).at("#site-nav a.active"), p
    end
  end
end
