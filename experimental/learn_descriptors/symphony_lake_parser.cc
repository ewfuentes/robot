#include "experimental/learn_descriptors/symphony_lake_parser.hh"

#include <iostream>

namespace robot::experimental::learn_descriptors {
DataParser::DataParser(const std::filesystem::path &image_root_dir,
                       const std::vector<std::string> &survey_list) {
    if (std::filesystem::exists(image_root_dir)) {
        surveys_.load(image_root_dir.string(), survey_list);
    } else {
        throw std::runtime_error("Error: The path '" + image_root_dir.string() +
                                 "' does not exist.");
    }
}
DataParser::~DataParser() {}
}  // namespace robot::experimental::learn_descriptors