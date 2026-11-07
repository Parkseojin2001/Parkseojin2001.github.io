require_relative "test_helper"

files = ARGV.empty? ? Dir[File.join(ROOT, "test/**/*_test.rb")] : ARGV.dup
ARGV.clear
files.sort.each { |f| require File.expand_path(f, ROOT) }
