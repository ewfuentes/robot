#include "experimental/learn_descriptors/symphony_lake_parser.hh"

#include <iostream>

namespace robot::experimental::learn_descriptors::symphony_lake_parser {
void hello_world(const std::string &msg) { std::cout << msg << std::endl; }
DataParser::DataParser(const std::string &image_root_dir,
                       const std::vector<std::string> &survey_list) {
    _surveys.load(image_root_dir, survey_list);
}
DataParser::~DataParser() {}
}  // namespace robot::experimental::learn_descriptors::symphony_lake_parser