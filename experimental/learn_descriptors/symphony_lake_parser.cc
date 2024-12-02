#include "experimental/learn_descriptors/symphony_lake_parser.hh"

#include <iostream>

#include "common/check.hh"

namespace robot::experimental::learn_descriptors {
DataParser::DataParser(const std::filesystem::path &image_root_dir,
                       const std::vector<std::string> &survey_list) {
    ROBOT_CHECK(std::filesystem::exists(image_root_dir), "Image root dir does not exist!",
          image_root_dir);
    surveys_.load(image_root_dir.string(), survey_list);
}
DataParser::~DataParser() {}
}  // namespace robot::experimental::learn_descriptors