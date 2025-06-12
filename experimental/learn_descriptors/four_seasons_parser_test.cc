#include "experimental/learn_descriptors/four_seasons_parser.hh"

#include <filesystem>

#include "gtest/gtest.h"

namespace robot::experimental::learn_descriptors {
TEST(FourSeasonsParserTest, parser_test) {
    const std::filesystem::path snippet_dir =
        "external/four_seasons_snippet/recording_2020-04-07_11-33-45";
    const std::filesystem::path calibration_dir = "external/four_seasons_snippet/calibration";

    std::cout << "snippet_dir: " << snippet_dir
              << "\texists: " << std::filesystem::exists(snippet_dir) << "\n"
              << "calibration_dir: " << calibration_dir << "\texists"
              << std::filesystem::exists(calibration_dir) << std::endl;

    FourSeasonsParser parser(snippet_dir, calibration_dir);

    EXPECT_NE(parser.num_images(), 0);

    cv::Mat img_first_and_last;
    cv::hconcat(parser.load_image(0), parser.load_image(parser.num_images() - 1),
                img_first_and_last);

    for (size_t i = 0; i < parser.num_images(); i++) {
        cv::Mat img = parser.load_image(i);
        const ImagePoint img_pt = parser.get_image_point(i);
        std::cout << "\rGPS: " << img_pt.gps.translation()
                  << "\tGround Truth: " << img_pt.ground_truth.matrix() << std::flush;
    }
}
}  // namespace robot::experimental::learn_descriptors