require_relative "test_helper"

class AboutTest < Minitest::Test
  def test_sections
    assert_equal ["소개", "기술 스택", "경력 · 활동", "연락처"],
                 html("/about/").css("main h2").map { _1.text.strip }
  end

  def test_contact_prefilled
    contact = html("/about/").at("main").text
    assert_includes contact, "github.com/Parkseojin2001"
  end
end
