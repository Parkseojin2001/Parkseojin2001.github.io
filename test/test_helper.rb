require "minitest/autorun"
require "nokogiri"
require "open3"

ROOT = File.expand_path("..", __dir__)
Dir.chdir(ROOT)

# Builds the site once per process (with unpublished samples) and exposes the output.
module SiteBuild
  DIR = File.join(ROOT, "test/.site")

  class << self
    def dir
      build unless @built
      DIR
    end

    def log
      dir
      @log
    end

    private

    def build
      @built = true
      if ENV["SKIP_BUILD"] == "1" && Dir.exist?(DIR)
        @log = ""
        return
      end
      env = { "JEKYLL_ENV" => "production" }
      @log, status = Open3.capture2e(env, "bundle", "exec", "jekyll", "build", "--unpublished", "-d", DIR)
      abort "jekyll build failed:\n#{@log}" unless status.success?
    end
  end
end

module SiteHelpers
  def site_path(path)
    path = path.end_with?("/") ? "#{path}index.html" : path
    File.join(SiteBuild.dir, path)
  end

  def site_file?(path)
    File.file?(site_path(path))
  end

  def html(path)
    assert site_file?(path), "missing page: #{path}"
    Nokogiri::HTML(File.read(site_path(path)))
  end
end

Minitest::Test.include(SiteHelpers)
