require_relative "test_helper"
require_relative "../_plugins/project-validator"
require "tmpdir"

class ProjectValidatorTest < Minitest::Test
  FULL = { "title" => "t", "summary" => "s", "period" => "p", "role" => ["r"], "stack" => ["x"] }.freeze

  def prj(data)
    [{ "path" => "_projects/a.md", "name" => "a", "data" => data }]
  end

  def test_valid
    assert_empty ProjectValidator.problems(projects: prj(FULL), posts: [])
  end

  def test_missing_and_blank
    data = FULL.merge("summary" => " ", "role" => [""]).reject { |k, _| k == "stack" }
    assert_equal ["_projects/a.md: 필수 항목 'summary' 이(가) 비어 있습니다",
                  "_projects/a.md: 필수 항목 'role' 이(가) 비어 있습니다",
                  "_projects/a.md: 필수 항목 'stack' 이(가) 비어 있습니다"],
                 ProjectValidator.problems(projects: prj(data), posts: [])
  end

  def test_unknown_project
    posts = [{ "path" => "_posts/x.md", "project" => "typo" }, { "path" => "_posts/y.md", "project" => "a" },
             { "path" => "_posts/z.md", "project" => nil }]
    assert_equal ["_posts/x.md: project 'typo' 에 해당하는 _projects 파일이 없습니다"],
                 ProjectValidator.problems(projects: prj(FULL), posts: posts)
  end

  def test_real_build_has_no_warnings
    refute_includes SiteBuild.log, "ProjectValidator"
  end

  # Jekyll은 title이 없으면 파일 이름으로 채우므로, 원본 front matter를 직접 읽어야 누락을 잡을 수 있다
  def test_front_matter_reads_raw_file_without_title
    Dir.mktmpdir do |dir|
      file = File.join(dir, "no-title.md")
      File.write(file, "---\nsummary: s\nperiod: p\nrole: [r]\nstack: [x]\n---\n본문\n")
      data = ProjectValidator.front_matter(file)
      refute data.key?("title")
      msgs = ProjectValidator.problems(projects: [{ "path" => "_projects/no-title.md", "name" => "no-title", "data" => data }], posts: [])
      assert_equal ["_projects/no-title.md: 필수 항목 'title' 이(가) 비어 있습니다"], msgs
    end
  end
end
