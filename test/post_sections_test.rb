require_relative 'test_helper'
require_relative '../_plugins/post-sections'

class PostSectionsTest < Minitest::Test
  def test_explicit_section_overrides_project_default
    assert_equal 'study', PostSections.section({})
    assert_equal 'devlog', PostSections.section('project' => 'sample-project')
    assert_equal 'study', PostSections.section('project' => 'sample-project', 'section' => 'study')
    assert_equal 'devlog', PostSections.section('section' => 'devlog')
  end

  def test_any_section_slug_is_its_own_section
    assert_equal 'travel', PostSections.section('section' => 'travel')
    assert_equal 'travel', PostSections.section('section' => 'Travel', 'project' => 'sample-project')
    assert_equal 'study', PostSections.section('section' => 'not a slug')
  end

  def test_section_order_and_titles
    config = { 'study' => { 'title' => 'Study' }, 'devlog' => { 'title' => 'Devlog' } }
    assert_equal %w[study devlog book travel], PostSections.ordered(%w[travel devlog book], config)
    assert_equal %w[study], PostSections.ordered([], config)
    assert_equal 'Devlog', PostSections.title('devlog', config)
    assert_equal 'Travel', PostSections.title('travel', config)
    assert_equal 'Side Notes', PostSections.title('side-notes', config)
  end

  def test_navigation_and_active_section
    study = html('/study/')
    assert_equal %w[Projects Study Devlog About], study.css('#site-nav a').map { |a| a.text.strip }
    assert_equal 'Study', study.at('#site-nav a.active').text.strip
    assert_equal 'Devlog', html('/devlog/').at('#site-nav a.active').text.strip
    assert_equal 'Devlog', html('/projects/sample-project/sample-devlog/').at('#site-nav a.active').text.strip
    assert_equal 'Projects', html('/projects/sample-project/').at('#site-nav a.active').text.strip
  end

  def test_lists_do_not_mix_study_and_devlog
    devlog_url = '/projects/sample-project/sample-devlog/'
    assert html('/devlog/').at("article.post-item a[href='#{devlog_url}']")
    files = Dir[File.join(SiteBuild.dir, 'study/**/index.html')]
    refute_empty files
    study_links = files.flat_map do |file|
      Nokogiri::HTML(File.read(file)).css('article.post-item a.post-title').map { |a| a['href'] }
    end
    refute_includes study_links, devlog_url
    assert_equal study_links.uniq.size, study_links.size
    assert_equal Dir['_posts/**/*.md'].size - 1, study_links.size
  end

  def test_study_pagination_stays_in_study
    first = html('/study/')
    assert_equal 10, first.css('article.post-item').size
    assert_equal '/study/page2/', first.at('.pager-next')['href']
    second = html('/study/page2/')
    assert_equal '/study/', second.at('.pager-prev')['href']
    assert_equal '/study/', first.at('.tree-all a')['href']
    assert_equal 1, html('/devlog/').css('article.post-item').size
    assert_nil html('/devlog/').at('.blog-pager')
  end
end
