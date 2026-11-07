require_relative "test_helper"
require "yaml"
require "date"

# 철자 오타·표기 변형 태그를 하나로 합친 결과를 고정한다
class TagsTest < Minitest::Test
  MERGED = {
    "DataAugmentation" => "Data Augmentation",
    "Augmented" => "Data Augmentation",
    "Generative-Model" => "Generative Models",
    "ml" => "Machine Learning",
    "Paper Reivew" => "Paper Review",
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
    assert_empty MERGED.keys & all_tags
  end

  def test_canonical_tags_in_use
    assert_empty MERGED.values.uniq - all_tags
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
    %w[dataaugmentation augmented generative-model ml paper-reivew
       model-evauation-improvement pipe-line probability-theory].each do |slug|
      refute site_file?("/tags/#{slug}/"), "old tag page /tags/#{slug}/ still built"
    end
  end
end
