# frozen_string_literal: true

# Warns (without failing the build) when a project is missing required
# front matter, or when a post's `project:` points at no _projects file.
module ProjectValidator
  REQUIRED = %w[title summary period role stack].freeze

  module_function

  # projects: [{ "path" => String, "name" => String, "data" => Hash }]
  # posts:    [{ "path" => String, "project" => String|nil }]
  def problems(projects:, posts:)
    missing_fields(projects) + unknown_projects(projects, posts)
  end

  def missing_fields(projects)
    projects.flat_map do |prj|
      REQUIRED.select { |key| blank?(prj["data"][key]) }
              .map { |key| "#{prj['path']}: 필수 항목 '#{key}' 이(가) 비어 있습니다" }
    end
  end

  def unknown_projects(projects, posts)
    names = projects.map { |prj| prj["name"] }
    posts.reject { |post| blank?(post["project"]) || names.include?(post["project"]) }
         .map { |post| "#{post['path']}: project '#{post['project']}' 에 해당하는 _projects 파일이 없습니다" }
  end

  def blank?(value)
    case value
    when nil then true
    when String then value.strip.empty?
    when Array then value.all? { |item| blank?(item) }
    else false
    end
  end
end

if defined?(Jekyll::Hooks)
  Jekyll::Hooks.register :site, :post_read do |site|
    projects = (site.collections["projects"]&.docs || []).map do |doc|
      { "path" => doc.relative_path, "name" => doc.basename_without_ext, "data" => doc.data }
    end
    posts = site.posts.docs.map { |doc| { "path" => doc.relative_path, "project" => doc.data["project"] } }

    ProjectValidator.problems(projects: projects, posts: posts).each do |msg|
      Jekyll.logger.warn "ProjectValidator:", msg
    end
  end
end
