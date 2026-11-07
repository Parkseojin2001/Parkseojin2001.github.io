require_relative "test_helper"

class ProjectTest < Minitest::Test
  P = "/projects/sample-project/"
  SOLO = "/projects/sample-solo/"

  def meta(path = P)
    html(path).at("aside.project-meta")
  end

  def test_meta_fields
    assert_equal ["PERIOD", "TEAM", "MY ROLE", "STACK", "LINKS"], meta.css("dt").map { _1.text.strip }
    assert_equal ["역할 하나", "역할 둘"], meta.css("dd li").map { _1.text.strip }
    assert_equal ["[github ↗]", "[demo ↗]"], meta.css(".project-links a").map { _1.text.strip }
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

  # Review Focus 4: 비어 있는 항목은 숨기지 않고 "None"으로 표시한다
  def test_optional_fields_show_none
    doc = html(SOLO)
    assert_nil doc.at("section.devlog")
    rows = meta(SOLO).css("dt").map { |dt| [dt.text.strip, dt.next_element.text.strip] }.to_h
    assert_equal ["PERIOD", "TEAM", "MY ROLE", "STACK", "LINKS"], rows.keys
    assert_equal "None", rows["TEAM"]
    assert_equal "None", rows["LINKS"]
    assert_equal "혼자 개발", rows["MY ROLE"]
    assert_nil doc.at(".project-links a")
    card = html("/projects/").at("a.project-card[href='#{SOLO}']")
    assert_equal "샘", card.at(".project-thumb.fallback")&.text&.strip
  end

  def test_list_sorted_by_order
    # 실제 프로젝트가 섞여 있어도 샘플끼리의 순서(order)만 확인한다
    hrefs = html("/projects/").css("a.project-card").map { _1["href"] }
    assert_equal [P, SOLO], hrefs & [P, SOLO]
  end

  def test_card_shows_stack
    card = html("/projects/").at("a.project-card[href='#{P}']")
    assert_equal "Python · FastAPI", card.at(".project-stack").text.strip
    assert_nil card.at(".project-stack-more")
  end

  # 카드의 스택은 7개까지만 보이고 나머지는 +N으로 줄인다
  def test_card_stack_capped
    html("/projects/").css("a.project-card .project-stack").each do |el|
      more = el.at(".project-stack-more")&.text.to_s
      shown = el.text.sub(more, "").strip.split(" · ")
      assert_operator shown.size, :<=, 7, el.text
      assert_match(/\A\+\d+\z/, more) if shown.size == 7 && !more.empty?
    end
  end

  def test_path_and_toc
    assert_equal "projects / sample-project", html(P).at(".post-path")&.text&.strip
    assert html(P).at("#toc-wrapper")
  end

  # 요약 박스는 화면 폭과 상관없이 본문 위에 놓인다 (옆 칸에는 목차만)
  def test_meta_above_content
    doc = html(P)
    assert_nil doc.at(".project-side .project-meta")
    assert_equal "aside", doc.at("article.project-page > header + aside")&.name
  end

  def test_samples_hidden_from_normal_builds
    %w[_projects/sample-project.md _projects/sample-solo.md
       _posts/projects/sample-project/2026-01-01-sample-devlog.md].each do |f|
      assert_match(/^published:\s*false/, File.read(f), f)
    end
  end

  def test_backlink_on_devlog
    link = html("/projects/sample-project/sample-devlog/").at(".project-backlink a")
    refute_nil link
    assert_equal ["/projects/sample-project/", "샘플 프로젝트"], [link["href"], link.text.strip]
    assert_includes html("/projects/sample-project/sample-devlog/").at(".project-backlink").text, "↳ project:"
  end

  def test_no_backlink_on_normal_post
    assert_nil html("/data-engineering/learning-spark/05/").at(".project-backlink")
  end

  # 썸네일이 확대용 링크로 감싸지면 카드(링크) 안에 링크가 생겨 카드가 깨진다
  def test_thumbnail_not_wrapped_in_link
    %w[/projects/ /].each do |page|
      raw = File.read(File.join(ROOT, "test/.site", page, "index.html"))
      refute_match(%r{<a class="project-card"[^>]*>\s*<a }, raw, page)
    end
  end
end
