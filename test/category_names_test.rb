require_relative "test_helper"
require "yaml"
require "date"

# 카테고리 이름이 대소문자·기호만 다르면 같은 주소(/categories/<slug>/)로 만들어져
# 한 페이지가 다른 페이지를 덮어쓴다 (Jekyll "Conflict: ... shared by multiple files").
class CategoryNamesTest < Minitest::Test
  def category_names
    Dir["_posts/**/*.md"].flat_map do |f|
      fm = File.read(f)[/\A---\r?\n(.*?)\r?\n---/m, 1]
      (YAML.safe_load(fm, permitted_classes: [Date, Time])["categories"] || []).map(&:to_s)
    end.uniq
  end

  def test_no_two_names_share_a_category_url
    clashes = category_names.group_by { Jekyll::Utils.slugify(_1) }.select { |_, v| v.size > 1 }
    assert_empty clashes
  end

  def test_pytorch_categories
    assert_includes category_names, "PyTorch Basics"
    assert_includes category_names, "PyTorch"
    refute_includes category_names, "Pytorch"
    assert site_file?("/categories/pytorch-basics/")
    assert site_file?("/categories/pytorch/")
  end

  def test_build_has_no_destination_conflict
    refute_includes SiteBuild.log, "Conflict:"
  end
end
