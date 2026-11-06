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

  # 대표 이미지는 About 페이지 맨 위, 이미지 파일은 _config.yml의 avatar 설정을 따른다
  def test_avatar_at_top
    main = html("/about/").at("main")
    img = main.at("img.about-avatar")
    refute_nil img
    assert_equal "/assets/img/avatar.jpg", img["src"]
    refute_empty img["alt"].to_s
    # 압축된 HTML은 한 줄이라 줄 번호 대신 문서 순서로 비교한다
    assert_equal img, main.css("img.about-avatar, h2").first, "avatar should come before the first section"
  end
end
