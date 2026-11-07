require_relative "test_helper"

class BlogTest < Minitest::Test
  def blog
    @blog ||= html("/blog/")
  end

  def test_first_page_has_ten_posts
    assert_equal 10, blog.css("article.post-item").size
  end

  def test_pagination_moved
    assert site_file?("/blog/page2/")
    refute site_file?("/page2/")
  end

  def test_tree_all_node_counts_every_post
    first = blog.at("nav.category-tree .tree-node")
    assert_equal "/blog/", first.at("a.tree-link")["href"]
    # the test build runs with --unpublished, so every post file is counted
    assert_equal Dir["_posts/**/*.md"].size, first.at(".tree-count").text.to_i
  end

  # Review Focus 3
  def test_tree_links_resolve
    links = blog.css("nav.category-tree a.tree-link")
    refute_empty links
    links.each { |a| assert site_file?(a["href"]), a["href"] }
  end

  # Review Focus 1
  def test_single_category_top_has_no_children
    node = blog.css("nav.category-tree .tree-node").find { _1.at("a.tree-link").text.strip == "Git" }
    refute_nil node
    assert_nil node.at("ul")
  end

  def test_subcategories_listed
    node = blog.css("nav.category-tree .tree-node").find { _1.at("a.tree-link").text.strip == "Data Engineering" }
    assert_equal ["Spark"], node.css("ul a.tree-link").map { _1.text.strip }
  end

  # Review Focus 2
  def test_no_empty_description
    ["/blog/", "/blog/page2/", "/blog/page3/"].each do |p|
      html(p).css("p.post-desc").each { |d| refute_empty d.text.strip, p }
    end
  end

  def test_item_meta_is_mono
    assert blog.at("article.post-item .post-item-meta.mono")
  end

  # 카테고리 트리는 알파벳 순 (대소문자 구분 없이), "all"은 맨 위
  def test_tree_sorted_alphabetically
    tree = blog.at("nav.category-tree") # the page has a mobile and a desktop copy
    tops = tree.css("> .tree-node:not(.tree-all)").map { _1.at("a.tree-link").text.strip }
    assert_equal tops.sort_by(&:downcase), tops
    tree.css("details ul").each do |ul|
      subs = ul.css("a.tree-link").map { _1.text.strip }
      assert_equal subs.sort_by(&:downcase), subs
    end
  end
end
