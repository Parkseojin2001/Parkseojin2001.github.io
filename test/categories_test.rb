require_relative "test_helper"
require "yaml"
require "date"

# Book 카테고리를 주제별 카테고리로 재분류한 결과를 고정한다
class CategoriesTest < Minitest::Test
  def categories_of(file)
    fm = File.read(file)[/\A---\r?\n(.*?)\r?\n---/m, 1] # 일부 글은 CRLF 줄바꿈
    YAML.safe_load(fm, permitted_classes: [Date, Time])["categories"]
  end

  def posts_in(top, sub)
    Dir["_posts/**/*.md"].select { categories_of(_1) == [top, sub] }
  end

  def test_book_and_old_nlp_top_categories_are_gone
    tops = Dir["_posts/**/*.md"].map { categories_of(_1)&.first }.uniq
    refute_includes tops, "Book"
    refute_includes tops, "NLP"
    refute Dir.exist?("_posts/book")
    refute Dir.exist?("_posts/nlp")
  end

  def test_new_category_counts
    {
      ["Machine Learning", "Scikit-learn"] => 18,
      ["Deep Learning", "PyTorch Basics"] => 6,
      ["Deep Learning", "NLP"] => 6,
      ["Deep Learning", "Computer Vision"] => 2,
      ["Algorithm", "Data Structure"] => 4,
      ["Algorithm", "Algorithm Theory"] => 7
    }.each { |(top, sub), n| assert_equal n, posts_in(top, sub).size, "#{top} / #{sub}" }
  end

  # 파일 폴더도 카테고리와 같은 구조로 둔다
  def test_files_live_in_matching_folders
    {
      ["Machine Learning", "Scikit-learn"] => "_posts/machine-learning/scikit-learn/",
      ["Deep Learning", "PyTorch Basics"] => "_posts/deep-learning/pytorch/",
      ["Deep Learning", "NLP"] => "_posts/deep-learning/nlp/",
      ["Deep Learning", "Computer Vision"] => "_posts/deep-learning/computer-vision/",
      ["Algorithm", "Data Structure"] => "_posts/algorithm/data-structure/",
      ["Algorithm", "Algorithm Theory"] => "_posts/algorithm/algorithm-theory/"
    }.each do |(top, sub), dir|
      posts_in(top, sub).each { |f| assert f.start_with?(dir), "#{f} should be in #{dir}" }
    end
  end

  def test_specific_posts_classified
    nlp = posts_in("Deep Learning", "NLP").map { File.basename(_1) }
    assert_includes nlp, "2025-10-28-nlp_augmented.md"
    cv = posts_in("Deep Learning", "Computer Vision").map { File.read(_1)[/^title:\s*(.+)$/, 1] }
    assert cv.any? { _1.include?("합성곱 신경망") }
    assert cv.any? { _1.include?("이미지 분류") }
  end

  def test_new_category_pages_built
    %w[machine-learning scikit-learn deep-learning pytorch pytorch-basics nlp computer-vision
       data-structure algorithm-theory].each do |slug|
      assert site_file?("/categories/#{slug}/"), slug
    end
    refute site_file?("/categories/book/")
  end
end
