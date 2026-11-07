require_relative "test_helper"

class HomeTest < Minitest::Test
  def doc
    @doc ||= html("/")
  end

  def test_section_order
    assert_equal ["HELLO, I'M SEOJIN", "./projects", "./now", "./study", "./devlog", "./categories"],
                 doc.css("main .section-label").map { _1.text.strip }
  end

  def test_featured_projects
    assert_equal %w[/projects/sample-project/ /projects/sample-solo/],
                 doc.css("#home-projects a.project-card").map { _1["href"] }
  end

  def test_now_items
    assert_equal ["Spark로 데이터 엔지니어링", "LLM Agent와 RAG"],
                 doc.css("#home-now .now-text").map { _1.text.strip }
    assert_equal "/categories/spark/", doc.at("#home-now .now-text a")["href"]
  end

  def test_recent_posts
    assert_equal 5, doc.css("#home-posts article.post-item").size
  end

  # Review Focus 3
  def test_category_chips_resolve
    chips = doc.css("#home-categories a.category-chip")
    refute_empty chips
    chips.each { |a| assert site_file?(a["href"]), a["href"] }
  end

  def test_hero
    assert doc.at("section.hero h1")
    assert_equal "#AI #ML #DataEngineering", doc.at(".hero-tags").text.split.join(" ")
  end

  # 홈 카테고리는 경로 형식: ~/slug (개수)
  def test_category_chips_as_paths
    chip = doc.css("#home-categories a.category-chip").find { _1.text.include?("deep-learning") }
    refute_nil chip
    assert_match(%r{\A~/deep-learning\s*\(\d+\)\z}, chip.text.strip.gsub(/\s+/, " ").sub("~/ ", "~/"))
  end
end
