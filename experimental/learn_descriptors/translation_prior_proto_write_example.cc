#include <filesystem>
#include <fstream>
#include <string>

#include "Eigen/Core"
#include "cxxopts.hpp"
#include "experimental/learn_descriptors/translation_prior.hh"
#include "experimental/learn_descriptors/translation_prior.pb.h"
#include "experimental/learn_descriptors/translation_prior_to_proto.hh"

namespace lrn_desc = robot::experimental::learn_descriptors;

int main(int argc, const char** argv) {
    // clang-format off
    cxxopts::Options options("tranlsation_prior_proto_write_example", "Write a translation_prior_proto");
    options.add_options()
        ("out_dir", "Path to directory to write proto files", cxxopts::value<std::string>())
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

    check_required("out_dir");

    const std::filesystem::path out_dir(args["out_dir"].as<std::string>());
    std::filesystem::create_directory(out_dir);

    Eigen::Matrix3d covariance;
    covariance << 0.94, 0.01, 0.05, 0.01, 0.94, 0.05, 0.05, 0.05, 0.9;
    const lrn_desc::TranslationPrior in{Eigen::Vector3d{1, 1, 1}, covariance};
    lrn_desc::proto::TranslationPrior out;
    lrn_desc::proto::pack_into(in, &out);

    std::ofstream output(out_dir / "translation_prior_proto_test.bin", std::ios::binary);
    if (!out.SerializeToOstream(&output)) {
        std::cerr << "Failed to write TranslationPriorProto." << std::endl;
    }
}