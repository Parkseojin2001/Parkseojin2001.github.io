require_relative "test_helper"

class UrlsTest < Minitest::Test
  REMOVABLE = %r{\A/page\d+/index\.html\z}
  # 재분류·태그 정리로 의도적으로 없앤 카테고리·태그 페이지
  RETIRED = %w[
    /categories/book/index.html
    /categories/data-augmentation/index.html
    /categories/deep-learning-with-pytorch-transformers/index.html
    /categories/introduction-to-machine-learning-with-python/index.html
    /categories/python-algorithm-interview/index.html
    /tags/dataaugmentation/index.html
    /tags/augmented/index.html
    /tags/generative-model/index.html
    /tags/ml/index.html
    /tags/paper-reivew/index.html
    /tags/model-evauation-improvement/index.html
    /tags/pipe-line/index.html
    /tags/probability-theory/index.html
  ].freeze

  def test_every_baseline_path_still_exists
    missing = File.readlines("test/fixtures/baseline-paths.txt", chomp: true)
                  .reject { |p| p.match?(REMOVABLE) || RETIRED.include?(p) }
                  .reject { |p| site_file?(p) }
    assert_empty missing
  end

  def test_feed_and_sitemap_exist
    assert site_file?("/feed.xml")
    assert site_file?("/sitemap.xml")
  end
end
