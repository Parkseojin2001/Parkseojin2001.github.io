require_relative "test_helper"
require "yaml"
require "date"

# 철자 오타·표기 변형 태그를 하나로 합친 결과를 고정한다
class TagsTest < Minitest::Test
  MERGED = {
    "DataAugmentation" => "data augmentation",
    "Augmented" => "data augmentation",
    "Generative-Model" => "generative models",
    "ml" => "machine learning",
    "Paper Reivew" => "paper review",
    "model-evauation-improvement" => "model-evaluation-improvement",
    "pipe-line" => "pipeline",
    "Probability theory" => "probability"
  }.freeze

  def tags_by_post
    @tags_by_post ||= Dir["_posts/**/*.md"].to_h do |f|
      fm = File.read(f)[/\A---\r?\n(.*?)\r?\n---/m, 1]
      [f, (YAML.safe_load(fm, permitted_classes: [Date, Time])["tags"] || []).map(&:to_s)]
    end
  end

  def all_tags = tags_by_post.values.flatten.uniq

  def test_variant_tags_are_gone
    assert_empty MERGED.keys.map(&:downcase) & all_tags
  end

  def test_canonical_tags_in_use
    assert_empty MERGED.values.uniq - all_tags
  end

  def test_all_tags_are_lowercase
    assert_empty all_tags.reject { _1 == _1.downcase }
  end

  def test_no_tag_repeated_within_a_post
    tags_by_post.each { |f, t| assert_equal t.uniq, t, "#{f} repeats a tag" }
  end

  def test_no_tags_differing_only_in_case_or_separators
    groups = all_tags.group_by { _1.downcase.gsub(/[^a-z0-9가-힣]/, "") }.select { |_, v| v.size > 1 }
    assert_empty groups
  end

  def test_tag_pages
    assert site_file?("/tags/data-augmentation/")
    assert site_file?("/tags/machine-learning/")
    assert site_file?("/tags/nlp/") # 소문자로 바꿔도 태그 페이지 주소는 그대로
    %w[dataaugmentation augmented generative-model ml paper-reivew
       model-evauation-improvement pipe-line probability-theory].each do |slug|
      refute site_file?("/tags/#{slug}/"), "old tag page /tags/#{slug}/ still built"
    end
  end
end
