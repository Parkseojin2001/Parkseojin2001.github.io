require_relative "test_helper"
require "tmpdir"
require "fileutils"
require "yaml"
require "date"

# Runs tools/new.sh against a throwaway copy of the repo layout.
class NewScriptTest < Minitest::Test
  def setup
    @dir = Dir.mktmpdir("new-sh")
    FileUtils.cp_r(File.join(ROOT, "tools"), @dir)
    FileUtils.mkdir_p(File.join(@dir, "_posts/data-engineering/spark"))
    File.write(File.join(@dir, "_posts/data-engineering/spark/2026-01-01-x.md"), "---\ntitle: x\n---\n")
    FileUtils.mkdir_p(File.join(@dir, "_projects"))
    File.write(File.join(@dir, "_projects/rag-chatbot.md"), "---\ntitle: RAG 챗봇\n---\n")
  end

  def teardown
    FileUtils.rm_rf(@dir)
  end

  def run_new(input, chdir: @dir, script: "tools/new.sh")
    out, status = Open3.capture2e({ "NEW_TODAY" => "2026-11-07" }, "bash", script, stdin_data: input, chdir: chdir)
    [out, status.exitstatus]
  end

  def path(rel) = File.join(@dir, rel)

  def front_matter(rel)
    YAML.safe_load(File.read(path(rel))[/\A---\n(.*?)\n---/m, 1], permitted_classes: [Date])
  end

  def test_post
    out, code = run_new("1\n1\nspark-06\n")
    assert_equal 0, code, out
    file = "_posts/data-engineering/spark/2026-11-07-spark-06.md"
    assert File.file?(path(file)), out
    assert_equal "/data-engineering/spark/spark-06/", front_matter(file)["permalink"]
    assert_includes out, "✓ 글 파일      #{file}"
    assert_includes out, "✓ 이미지 폴더  assets/img/post/data-engineering/spark/   (새로 만듦)"
    assert_includes out, "✓ 주소         /data-engineering/spark/spark-06/"
    assert Dir.exist?(path("assets/img/post/data-engineering/spark"))
  end

  def test_post_new_category_path
    out, code = run_new("1\nai-agent/langgraph\nagent-01\n")
    assert_equal 0, code, out
    assert File.file?(path("_posts/ai-agent/langgraph/2026-11-07-agent-01.md")), out
  end

  def test_devlog
    out, code = run_new("2\n1\nvector-db\n")
    assert_equal 0, code, out
    file = "_posts/projects/rag-chatbot/2026-11-07-vector-db.md"
    fm = front_matter(file)
    assert_equal "rag-chatbot", fm["project"]
    assert_equal "/projects/rag-chatbot/vector-db/", fm["permalink"]
    assert_includes out, "✓ 연결         project: rag-chatbot → 프로젝트 페이지 ./devlog에 자동 표시"
    assert_includes out, "assets/img/post/projects/rag-chatbot/"
  end

  def test_project
    out, code = run_new("3\nmy-app\n")
    assert_equal 0, code, out
    text = File.read(path("_projects/my-app.md"))
    ["title:", "summary:", "period:", "role:", "stack:", "# [필수]", "mermaid: true",
     "toc: true", "comments: false", "## 문제 정의", "## 아키텍처", "## 내가 한 일", "## 결과"].each do |s|
      assert_includes text, s
    end
    assert Dir.exist?(path("assets/img/projects/my-app"))
    assert_includes out, "✓ 주소         /projects/my-app/"
    assert_includes out, "/assets/img/projects/my-app/"
  end

  def test_generated_files_are_valid_yaml
    run_new("2\n1\nvector-db\n") # before my-app exists, so project 1 is rag-chatbot
    run_new("3\nmy-app\n")
    run_new("1\n1\nspark-06\n")
    ["_projects/my-app.md", "_posts/projects/rag-chatbot/2026-11-07-vector-db.md",
     "_posts/data-engineering/spark/2026-11-07-spark-06.md"].each do |f|
      assert_kind_of Hash, front_matter(f), f
    end
  end

  def test_existing_image_dir_reported
    run_new("1\n1\nspark-06\n")
    out, = run_new("1\n1\nspark-07\n")
    assert_includes out, "assets/img/post/data-engineering/spark/   (이미 있음)"
  end

  def test_invalid_name_reprompts
    out, code = run_new("3\nMy App\nmy-app\n")
    assert_equal 0, code, out
    assert_includes out, "✗ 영어 소문자, 숫자, 하이픈만 사용할 수 있습니다"
    assert File.file?(path("_projects/my-app.md"))
  end

  def test_refuses_overwrite
    run_new("3\nmy-app\n")
    File.write(path("_projects/my-app.md"), "edited")
    out, code = run_new("3\nmy-app\n")
    assert_equal 1, code
    assert_includes out, "✗ 이미 있는 파일입니다: _projects/my-app.md"
    assert_equal "edited", File.read(path("_projects/my-app.md"))
  end

  def test_devlog_without_projects
    FileUtils.rm_rf(path("_projects"))
    out, code = run_new("2\n")
    assert_equal 1, code
    assert_includes out, "✗ 프로젝트가 없습니다. 먼저 3) 프로젝트를 만드세요"
  end

  def test_invalid_type_reprompts
    out, code = run_new("9\n3\nmy-app\n")
    assert_equal 0, code, out
    assert File.file?(path("_projects/my-app.md"))
  end

  # Review Focus 5: run from a subdirectory
  def test_runs_from_subdirectory
    out, code = run_new("3\nsub-app\n", chdir: path("_posts"), script: "../tools/new.sh")
    assert_equal 0, code, out
    assert File.file?(path("_projects/sub-app.md"))
  end
end
