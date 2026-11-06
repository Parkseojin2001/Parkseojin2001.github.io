require_relative "test_helper"

class UrlsTest < Minitest::Test
  REMOVABLE = %r{\A/page\d+/index\.html\z}

  def test_every_baseline_path_still_exists
    missing = File.readlines("test/fixtures/baseline-paths.txt", chomp: true)
                  .reject { |p| p.match?(REMOVABLE) }
                  .reject { |p| site_file?(p) }
    assert_empty missing
  end

  def test_feed_and_sitemap_exist
    assert site_file?("/feed.xml")
    assert site_file?("/sitemap.xml")
  end
end
