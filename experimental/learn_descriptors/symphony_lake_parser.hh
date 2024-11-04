#pragma once

#include <string.h>

#include "symphony_lake_dataset/SurveyVector.h"

namespace robot::experimental::learn_descriptors::symphony_lake_parser {
void hello_world(const std::string &msg);
class DataParser {
   public:
    DataParser(const std::string &image_root_dir, const std::vector<std::string> &survey_list);
    ~DataParser();

    const symphony_lake_dataset::SurveyVector &getSurveys() const { return _surveys; };

   private:
    symphony_lake_dataset::SurveyVector _surveys;
};
}  // namespace robot::experimental::learn_descriptors::symphony_lake_parser