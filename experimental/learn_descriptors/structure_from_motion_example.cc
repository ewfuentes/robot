#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "cxxopts.hpp"
#include "experimental/learn_descriptors/backend.hh"
#include "experimental/learn_descriptors/four_seasons_parser.hh"
#include "experimental/learn_descriptors/frame.hh"
#include "experimental/learn_descriptors/frontend.hh"
#include "experimental/learn_descriptors/frontend_definitions.hh"
#include "experimental/learn_descriptors/image_point_four_seasons.hh"
#include "experimental/learn_descriptors/structure_from_motion.hh"
#include "experimental/learn_descriptors/structure_from_motion_types.hh"
#include "opencv2/opencv.hpp"
#include "visualization/opencv/opencv_viz.hh"

int main(int argc, const char **argv) {
    using namespace robot::experimental::learn_descriptors;
    // clang-format off
    cxxopts::Options options("four_seasons_parser_example", "Demonstrate usage of four_seasons_parser");
    options.add_options()
    ("data_dir", "Path to dataset root directory", cxxopts::value<std::string>())
    ("calibration_dir", "Path to dataset calibration directory", cxxopts::value<std::string>())
    ("help", "Print usage");
    // clang-format on

    auto args = options.parse(argc, argv);

    const auto check_required = [&](const std::string &opt) {
        if (args.count(opt) == 0) {
            std::cout << "Missing " << opt << " argument" << std::endl;
            std::cout << options.help() << std::endl;
            std::exit(1);
        }
    };

    if (args.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    check_required("data_dir");
    check_required("calibration_dir");

    const std::filesystem::path path_data = args["data_dir"].as<std::string>();
    const std::filesystem::path path_calibration = args["calibration_dir"].as<std::string>();
    FourSeasonsParser parser(path_data, path_calibration);

    FrontendParams frontend_params{FrontendParams::ExtractorType::SIFT,
                                   FrontendParams::MatcherType::KNN, true, false};
    StructureFromMotion sfm(frontend_params);

    // short sequences (maybe about a doezen points that have decent overlap and aren't spaced too
    // far apart on the order of maybe a meter) perform ok. many improvements needed for the overall
    // system
    for (size_t i = 835; i < 855; i += 2) {
        const ImagePointFourSeasons img_pt = parser.get_image_point(i);
        sfm.add_image_point(parser.load_image(i), std::make_shared<ImagePointFourSeasons>(img_pt));
    }

    using epoch = size_t;
    Backend::graph_step_debug_func graph_itr_debug_func = [&](const gtsam::Values &vals,
                                                              const epoch iter) {
        std::cout << "iteration " << iter << " complete!";
        std::string window_name = "Iteration_" + std::to_string(iter);
        StructureFromMotion::graph_values(vals, window_name);
    };
    // passing graph_itr_debug_func will execute the function after each optimization epoch
    // sfm.solve_structure(10, graph_itr_debug_func);
    sfm.solve_structure(10);
    StructureFromMotion::graph_values(sfm.result());
}