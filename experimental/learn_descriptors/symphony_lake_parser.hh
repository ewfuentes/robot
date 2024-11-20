#pragma once
#pragma once
#include <string.h>

#include <filesystem>

#include "symphony_lake_dataset/SurveyVector.h"

namespace robot::experimental::learn_descriptors {
class DataParser {
   public:
    DataParser(const std::filesystem::path &image_root_dir,
               const std::vector<std::string> &survey_list);
    ~DataParser();

    const symphony_lake_dataset::SurveyVector &get_surveys() const { return surveys_; };

   private:
    symphony_lake_dataset::SurveyVector surveys_;
};
}  // namespace robot::experimental::learn_descriptors