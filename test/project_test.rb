require_relative "test_helper"

class ProjectTest < Minitest::Test
  P = "/projects/sample-project/"
  SOLO = "/projects/sample-solo/"

  def meta(path = P)
    html(path).at("aside.project-meta")
  end

  def test_meta_fields
    assert_equal ["PERIOD", "TEAM", "MY ROLE", "STACK"], meta.css("dt").map { _1.text.strip }
    assert_equal ["역할 하나", "역할 둘"], meta.css("dd li").map { _1.text.strip }
    assert_equal ["github ↗", "demo ↗"], meta.css(".project-links a").map { _1.text.strip }
  end

  def test_links_open_in_new_tab
    meta.css(".project-links a").each do |a|
      assert_equal "_blank", a["target"]
      assert_includes a["rel"], "noopener"
    end
  end

  def test_devlog_listed
    sec = html(P).at("section.devlog")
    refute_nil sec
    assert_equal "./devlog", sec.at(".section-label").text.strip
    assert_equal "/projects/sample-project/sample-devlog/", sec.at("a.post-title")["href"]
  end

  # Review Focus 4: optional fields missing
  def test_optional_fields_absent
    doc = html(SOLO)
    assert_nil doc.at("section.devlog")
    assert_equal ["PERIOD", "MY ROLE", "STACK"], meta(SOLO).css("dt").map { _1.text.strip }
    assert_nil doc.at(".project-links")
    card = html("/projects/").css("a.project-card")[1]
    assert_equal "샘", card.at(".project-thumb.fallback")&.text&.strip
  end

  def test_list_sorted_by_order
    hrefs = html("/projects/").css("a.project-card").map { _1["href"] }
    assert_equal [P, SOLO], hrefs
  end

  def test_card_shows_stack
    card = html("/projects/").at("a.project-card")
    assert_equal "Python · FastAPI", card.at(".project-stack").text.strip
  end

  def test_path_and_toc
    assert_equal "projects / sample-project", html(P).at(".post-path")&.text&.strip
    assert html(P).at("#toc-wrapper")
  end

  def test_samples_hidden_from_normal_builds
    %w[_projects/sample-project.md _projects/sample-solo.md
       _posts/projects/sample-project/2026-01-01-sample-devlog.md].each do |f|
      assert_match(/^published:\s*false/, File.read(f), f)
    end
  end
end
