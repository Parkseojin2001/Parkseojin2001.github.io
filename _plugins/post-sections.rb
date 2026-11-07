# frozen_string_literal: true

module PostSections
  module_function

  def section(data)
    return data['section'] if %w[study devlog].include?(data['section'])

    data['project'].to_s.strip.empty? ? 'study' : 'devlog'
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
      visible = site.posts.docs.reverse.reject { |post| post.data['hidden'] == true }
      per_page = [site.config.fetch('paginate', 10).to_i, 1].max

      %w[study devlog].each do |section|
        section_posts = visible.select { |post| post.data['section'] == section }
        pinned, normal = section_posts.partition { |post| post.data['pin'] == true }
        ordered = pinned + normal
        total_pages = [(ordered.size.to_f / per_page).ceil, 1].max

        (1..total_pages).each do |number|
          path = ->(n) { n == 1 ? "/#{section}/" : "/#{section}/page#{n}/" }
          page = PageWithoutAFile.new(site, site.source, path.call(number), 'index.html')
          page.data.merge!(
            'layout' => 'blog',
            'title' => section == 'study' ? 'Study' : 'Devlog',
            'section' => section,
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
