
#pragma once

#include <filesystem>

#include "kimera-vio/dataprovider/DataProviderInterface.h"

namespace robot::experimental::overhead_matching {

class SpectacularDataProvider : public VIO::DataProviderInterface {
   public:
    explicit SpectacularDataProvider(const std::filesystem::path &data_path);
    bool spin() override;
    bool hasData() const override;

   private:
};
}  // namespace robot::experimental::overhead_matching
