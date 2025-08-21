#include <cstring>
#include <memory>
#include <optional>
#include <sstream>
#include <type_traits>
#include <vector>

#include "Eigen/Core"
#include "common/check.hh"
#include "common/sqlite3/sqlite3.hh"
#include "cxxopts.hpp"
#include "experimental/learn_descriptors/four_seasons_parser.hh"
#include "experimental/learn_descriptors/image_point_four_seasons.hh"

template <typename Derived>
std::vector<unsigned char> eigen_to_blob(const Eigen::MatrixBase<Derived>& mat) {
    static_assert(std::is_trivially_copyable_v<typename Derived::Scalar>,
                  "Only trivially copyable scalar types supported (e.g., float, double, int)");

    using Scalar = typename Derived::Scalar;
    const int rows = static_cast<int>(mat.rows());
    const int cols = static_cast<int>(mat.cols());
    const size_t scalar_size = sizeof(Scalar);
    const size_t total_data_bytes = mat.size() * scalar_size;

    std::vector<unsigned char> blob;
    blob.resize(2 * sizeof(int) + total_data_bytes);

    // Copy shape
    std::memcpy(blob.data(), &rows, sizeof(int));
    std::memcpy(blob.data() + sizeof(int), &cols, sizeof(int));

    // Copy matrix data (Eigen is column-major by default)
    std::memcpy(blob.data() + 2 * sizeof(int), mat.derived().data(), total_data_bytes);

    return blob;
}

template <typename Derived>
std::vector<unsigned char> eigen_to_raw_blob(const Eigen::MatrixBase<Derived>& mat) {
    static_assert(std::is_trivially_copyable_v<typename Derived::Scalar>,
                  "Only trivially copyable scalar types supported (e.g., float, double)");

    using Scalar = typename Derived::Scalar;
    const size_t total_data_bytes = mat.size() * sizeof(Scalar);

    std::vector<unsigned char> blob(total_data_bytes);
    std::memcpy(blob.data(), mat.derived().data(), total_data_bytes);

    return blob;
}

namespace lrn_desc = robot::experimental::learn_descriptors;

void write_priors(const lrn_desc::FourSeasonsParser& parser, robot::sqlite3::Database& colmap_db) {
    std::optional<Eigen::Vector3d> p_first_in_world;
    for (size_t i = 0; i < parser.num_images(); i++) {
        const lrn_desc::ImagePointFourSeasons img_pt = parser.get_image_point(i);
        if (img_pt.world_from_cam_ground_truth()) {
            if (!p_first_in_world) {
                p_first_in_world = img_pt.world_from_cam_ground_truth()->translation();
                break;
            }
        }
    }
    ROBOT_CHECK(p_first_in_world);
    Eigen::Matrix3d grndtrth_covariance = Eigen::Matrix3d::Identity() * 0.05;
    for (size_t i = 0; i < parser.num_images(); i++) {
        const lrn_desc::ImagePointFourSeasons img_pt = parser.get_image_point(i);
        Eigen::Matrix3d translation_covariance;
        Eigen::Vector3d p_in_world;
        if (img_pt.world_from_cam_ground_truth()) {
            p_in_world = img_pt.world_from_cam_ground_truth()->translation() - *p_first_in_world;
            translation_covariance = grndtrth_covariance;
        } else if (img_pt.cam_in_world() && img_pt.translation_covariance_in_cam()) {
            p_in_world = *img_pt.cam_in_world() - *p_first_in_world;
            translation_covariance = *img_pt.translation_covariance_in_cam();
        } else {
            continue;
        }
        std::cout << "writing frame " << i << " pose prior to colmap db" << std::endl;
        const auto statement = colmap_db.prepare(
            "INSERT INTO pose_priors (image_id,position,coordinate_system,position_covariance) "
            "VALUES (:image_id,:position,:coordinate_system,:position_covariance);");
        colmap_db.bind(statement, {{std::string(":image_id"), std::to_string(i + 1)},
                                   {std::string(":position"), eigen_to_raw_blob(p_in_world)},
                                   {std::string(":coordinate_system"), 1},  // 1 for cartesian
                                   {std::string(":position_covariance"),
                                    eigen_to_raw_blob(translation_covariance)}});
        colmap_db.step(statement);
    }
}

int main(int argc, const char** argv) {
    // clang-format off
    cxxopts::Options options("four_seasons_parser_example", "Demonstrate usage of four_seasons_parser");
    options.add_options()
        ("data_dir", "Path to dataset root directory", cxxopts::value<std::string>())
        ("calibration_dir", "Path to dataset calibration directory", cxxopts::value<std::string>())
        ("colmap_database_path", "Path to database.db file for colmap. If it doesn't exist, it will be created", cxxopts::value<std::string>())
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
    check_required("colmap_database_path");

    const std::filesystem::path path_data = args["data_dir"].as<std::string>();
    const std::filesystem::path path_calibration = args["calibration_dir"].as<std::string>();
    const std::filesystem::path path_colmap_database =
        args["colmap_database_path"].as<std::string>();

        robot::sqlite3::Database colmap_db(path_colmap_database);
    ROBOT_CHECK(
        !colmap_db
             .query("SELECT name FROM sqlite_master WHERE type='table' AND name='pose_priors';")
             .empty());
    std::vector<robot::sqlite3::Database::Row> schema_rows =
        colmap_db.query("PRAGMA table_info(pose_priors);");
    ROBOT_CHECK(!schema_rows.empty());
    const std::vector<std::string> expected_columns = {"image_id", "position", "coordinate_system",
                                                       "position_covariance"};
    for (size_t i = 0; i < expected_columns.size(); ++i) {
        ROBOT_CHECK(std::get<std::string>(schema_rows.at(i).value(1)) == expected_columns[i]);
    }
    colmap_db.query("DELETE FROM pose_priors;");

    lrn_desc::FourSeasonsParser parser(path_data, path_calibration);

    ROBOT_CHECK(parser.num_images() != 0);

    write_priors(parser, colmap_db);
}