# frozen_string_literal: true

# 글을 섹션(상단 메뉴)별로 나눈다.
# - front matter의 `section: <slug>`가 있으면 그 섹션 (예: travel → /travel/)
# - 없으면 `project:`가 있는 글은 devlog, 나머지는 study
# 글이 하나라도 있는 섹션마다 /<slug>/ 목록 페이지를 만들고, 메뉴·홈에서 쓸 목록을 site.data['post_sections']에 넣는다.
# study는 글이 없어도 항상 만든다.
module PostSections
  module_function

  DEFAULT = 'study'
  SLUG = /\A[a-z0-9][a-z0-9-]*\z/

  def section(data)
    explicit = data['section'].to_s.strip.downcase
    return explicit if explicit.match?(SLUG)

    data['project'].to_s.strip.empty? ? DEFAULT : 'devlog'
  end

  # 섹션 slug 목록: _data/sections.yml 순서 → 나머지 알파벳순. study 외에는 글이 있는 섹션만.
  def ordered(slugs_with_posts, config)
    configured = config.keys.map(&:to_s)
    present = (slugs_with_posts | [DEFAULT])
    configured.select { |s| present.include?(s) } + (present - configured).sort
  end

  def title(slug, config)
    entry = config[slug] || {}
    entry['title'] || slug.split('-').map(&:capitalize).join(' ')
  end
end

Jekyll::Hooks.register :site, :post_read do |site|
  site.posts.docs.each { |post| post.data['section'] = PostSections.section(post.data) }
end

module Jekyll
  class PostSectionsGenerator < Generator
    safe true
    priority :low

    def generate(site)
      config = site.data['sections'] || {}
      visible = site.posts.docs.reverse.reject { |post| post.data['hidden'] == true }
      per_page = [site.config.fetch('paginate', 10).to_i, 1].max
      slugs = PostSections.ordered(visible.map { |post| post.data['section'] }.uniq, config)

      site.data['post_sections'] = slugs.map do |slug|
        entry = config[slug] || {}
        {
          'slug' => slug,
          'title' => PostSections.title(slug, config),
          'description' => entry['description'],
          'empty' => entry['empty'],
          'more' => entry['more'] || "all #{slug}",
          'count' => visible.count { |post| post.data['section'] == slug }
        }
      end

      site.data['post_sections'].each do |info|
        section = info['slug']
        section_posts = visible.select { |post| post.data['section'] == section }
        pinned, normal = section_posts.partition { |post| post.data['pin'] == true }
        ordered = pinned + normal
        total_pages = [(ordered.size.to_f / per_page).ceil, 1].max

        (1..total_pages).each do |number|
          path = ->(n) { n == 1 ? "/#{section}/" : "/#{section}/page#{n}/" }
          page = PageWithoutAFile.new(site, site.source, path.call(number), 'index.html')
          page.data.merge!(
            'layout' => 'blog',
            'title' => info['title'],
            'section' => section,
            'section_description' => info['description'],
            'section_all_posts' => section_posts,
            'section_posts' => ordered.slice((number - 1) * per_page, per_page) || [],
            'section_paginator' => {
              'page' => number,
              'total_pages' => total_pages,
              'first_page_path' => path.call(1),
              'page_path' => "/#{section}/page:num/",
              'previous_page' => number > 1 ? number - 1 : nil,
              'previous_page_path' => number > 1 ? path.call(number - 1) : nil,
              'next_page_path' => number < total_pages ? path.call(number + 1) : nil
            }
          )
          site.pages << page
        end
      end
    end
  end
end
