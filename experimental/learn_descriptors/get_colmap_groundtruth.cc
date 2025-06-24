#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "experimental/learn_descriptors/symphony_lake_parser.hh"

namespace ld = robot::experimental::learn_descriptors;

struct ImagePose {
    std::string filename;
    double X;
    double Y;
    double Z;
    double heading;
};

void writeCsv(const std::string& path, const std::vector<ImagePose>& rows) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open " + path);
    }

    out << "filename,X,Y,Z,heading\n";

    out << std::fixed << std::setprecision(6);
    for (const auto& r : rows) {
        out << r.filename << ',' << r.X << ',' << r.Y << ',' << r.Z << ',' << r.heading << '\n';
    }
}

void writeTxt(const std::string& path, const std::vector<ImagePose>& rows) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open " + path);
    }

    out << std::fixed << std::setprecision(6);
    for (const auto& r : rows) {
        out << r.filename << ' ' << r.X << ' ' << r.Y << ' ' << r.Z << ' ' << r.heading << '\n';
    }
}

int main() {
    ld::DataParser data_parser("/home/pizzaroll04/Documents/datasets",
                               std::vector<std::string>{"symphony_lake_building_test"});

    const symphony_lake_dataset::SurveyVector& survey_vector = data_parser.get_surveys();
    const symphony_lake_dataset::Survey& survey = survey_vector.get(0);

    std::vector<ImagePose> poses;
    for (int i = 0; i < 718; i++) {
        symphony_lake_dataset::ImagePoint img_pt = survey.getImagePoint(i);
        const Eigen::Isometry3d T_wrld_cam =
            data_parser.get_world_from_boat(img_pt) * data_parser.get_boat_from_camera(img_pt);
        ImagePose img_pose;

        std::ostringstream oss;
        oss << std::setw(4) << std::setfill('0') << i;
        img_pose.filename = oss.str() + ".jpg";
        img_pose.X = T_wrld_cam.translation().x();
        img_pose.Y = T_wrld_cam.translation().y();
        img_pose.Z = T_wrld_cam.translation().z();
        Eigen::Matrix3d R = T_wrld_cam.linear();
        img_pose.heading = std::atan2(R(1, 0), R(0, 0));

        poses.push_back(img_pose);
    }

    try {
        writeTxt("/home/pizzaroll04/Documents/datasets/symphony_lake_building_test/ref_images.txt",
                 poses);
        std::cout << "Wrote " << poses.size() << " rows to ref_images.csv\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
