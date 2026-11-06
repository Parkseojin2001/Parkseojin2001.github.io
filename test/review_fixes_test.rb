require_relative "test_helper"

# 최종 리뷰에서 나온 Critical/Important 지적을 고정하는 테스트
class ReviewFixesTest < Minitest::Test
  PROJECT = "/projects/sample-project/"
  POST = "/data-engineering/learning-spark/05/"

  def css
    @css ||= File.read(File.join(SiteBuild.dir, "assets/css/jekyll-theme-chirpy.css"))
  end

  def script_srcs(path)
    html(path).css("script[src]").map { _1["src"] }
  end

  def stylesheet_hrefs(path)
    html(path).css("link[rel=stylesheet]").map { _1["href"] }
  end

  # Critical: home/post 번들이 dayjs를 전역으로 쓰므로, 번들을 쓰는 모든 레이아웃이 dayjs를 로드해야 한다
  def test_dayjs_loaded_wherever_home_or_post_bundle_runs
    ["/blog/", "/blog/page2/", PROJECT].each do |p|
      assert script_srcs(p).any? { _1.include?("dayjs") }, "#{p}: dayjs script missing"
    end
  end

  # Important: 고정 상단바 아래로 목차가 들어가면 안 된다
  def test_desktop_toc_sticks_below_topbar
    top = css[/#toc-wrapper\{top:([^;]*);transition:top/, 1]
    refute_nil top, "#toc-wrapper rule not found"
    refute_equal "0", top
  end

  def test_mobile_toc_bar_sticks_below_topbar
    top = css[/#toc-bar\{position:-webkit-sticky;position:sticky;top:([^;]*);/, 1]
    refute_nil top, "#toc-bar rule not found"
    refute_equal "0", top
  end

  def test_toc_scripts_offset_by_topbar_height
    desktop = File.read("_javascript/modules/components/toc/toc-desktop.js")
    mobile = File.read("_javascript/modules/components/toc/toc-mobile.js")
    refute_includes desktop, "16 * 2 // 2rem"
    refute_includes mobile, "16 * 3; // 3rem"
  end

  # Important: 페이지 번호 "1"은 /blog/ 로 가야 한다
  def test_paginator_first_page_links_to_blog
    first = html("/blog/page2/").css("ul.pagination a.page-link").find { _1.text.strip == "1" }
    refute_nil first
    assert_equal "/blog/", first["href"]
  end

  # Important: description: "" 인 글에 빈 <p>가 생기면 안 된다
  def test_no_empty_description_on_post_page
    url = "/python-algo-interview/algorithm/dynamic-programming/"
    html(url).css("p.post-desc").each { |d| refute_empty d.text.strip, url }
  end

  # Important: 프로젝트 페이지도 이미지 확대(glightbox) CSS가 필요하다
  def test_project_page_has_image_popup_css
    assert stylesheet_hrefs(PROJECT).any? { _1.include?("glightbox") }, "glightbox css missing"
    assert stylesheet_hrefs(PROJECT).any? { _1.include?("loading-attribute-polyfill") }, "lazy-polyfill css missing"
  end
end
