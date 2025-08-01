#include "common/sqlite3/sqlite3.hh"
#include "cxxopts.hpp"
#include "experimental/learn_descriptors/four_seasons_parser.hh"

int main(int argc, const char** argv) {
    // clang-format off
    cxxopts::Options options("four_seasons_parser_example", "Demonstrate usage of four_seasons_parser");
    options.add_options()
        ("data_dir", "Path to dataset root directory", cxxopts::value<std::string>())
        ("calibration_dir", "Path to dataset calibration directory", cxxopts::value<std::string>())
        ("help", "Print usage");
    // clang-format on

    auto args = options.parse(argc, argv);

    const auto check_required = [&](const std::string& opt) {
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
}