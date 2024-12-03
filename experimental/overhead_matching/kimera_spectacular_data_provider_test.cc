
#include "experimental/overhead_matching/spectacular_log.hh"
#include "experimental/overhead_matching/kimera_spectacular_data_provider.hh"

#include <iostream>
#include <sstream>

#include "fmt/format.h"
#include "gtest/gtest.h"
#include "opencv2/opencv.hpp"

#include "kimera-vio/pipeline/Pipeline-definitions.h"

namespace robot::experimental::overhead_matching {

bool images_equal(cv::Mat img1, cv::Mat img2) {
    if (img1.size() != img2.size() || img1.type() != img2.type()) {
        return false;
    }
    cv::Mat diff;
    cv::absdiff(img1, img2, diff);
    diff = diff.reshape(1);
    return cv::countNonZero(diff) == 0;
}

std::ostream &operator<<(std::ostream &out, const time::RobotTimestamp &t) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(9)
       << std::chrono::duration<double>(t.time_since_epoch()).count();
    out << ss.str();
    return out;
}

TEST(KimeraSpectacularDataProviderTest, happy_case) {
    // Setup
    const std::filesystem::path log_path(
        "external/spectacular_log_snippet/recording_2024-11-21_13-36-30");
    SpectacularLog log(log_path);

    const std::filesystem::path vio_config_path(""); // loads default params
    VIO::VioParams vio_params(vio_config_path);

    SpectacularDataProviderInterface s_interface(
        log_path, 0, std::numeric_limits<int>::max(), vio_params
    );

    // Action

}

}  // namespace robot::experimental::overhead_matching
