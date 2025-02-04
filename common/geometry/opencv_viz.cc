#include "common/geometry/opencv_viz.hh"
#include <iostream>

namespace robot::geometry::opencv_viz {
template <typename Derived>
cv::Mat eigen_to_cv_mat(const Eigen::MatrixBase<Derived>& eigenMatrix) {
    int cvType;
    if constexpr (std::is_same_v<typename Derived::Scalar, double>) {
        cvType = CV_64F; // 64-bit floating point
    } else if constexpr (std::is_same_v<typename Derived::Scalar, float>) {
        cvType = CV_32F; // 32-bit floating point
    } else if constexpr (std::is_same_v<typename Derived::Scalar, int>) {
        cvType = CV_32S; // 32-bit signed integer
    } else {
        static_assert(!std::is_same_v<Derived, Derived>, "Unsupported data type in Eigen matrix");
    }
    return cv::Mat(eigenMatrix.rows(), eigenMatrix.cols(), cvType, const_cast<void*>(static_cast<const void*>(eigenMatrix.derived().data()))).clone();
}
cv::Vec3d rotation_matrix_to_axis_angle(const cv::Matx33d &R) {
    // Ensure R is a valid rotation matrix
    CV_Assert(cv::determinant(R) > 0.999 && cv::determinant(R) < 1.001);
    
    double trace = R(0, 0) + R(1, 1) + R(2, 2);
    double theta = std::acos((trace - 1) / 2.0);

    if (std::abs(theta) < 1e-6) {
        return cv::Vec3d(0, 0, 0);
    }

    double sinTheta = std::sin(theta);
    cv::Vec3d axis(
        (R(2, 1) - R(1, 2)) / (2.0 * sinTheta),
        (R(0, 2) - R(2, 0)) / (2.0 * sinTheta),
        (R(1, 0) - R(0, 1)) / (2.0 * sinTheta)
    );

    return axis * theta;
}

void viz_scene(const std::vector<Eigen::Isometry3d> &poses, const std::vector<Eigen::Vector3d> &points) {
    // Eigen::Matrix4d test_mat = poses.front().matrix();
    // cv::Mat cv_mat = eigen_to_cv_mat(test_mat);
    // std::cout << "viz_scene called! " << std::endl;
    // (void)poses;
    // (void)points;
    cv::viz::Viz3d window("Viz Scene");

    constexpr double pose_size = 2;    
    for (unsigned int i = 0; i < poses.size(); i++) {       
        // Eigen::Matrix4d mat = poses[i].matrix();
        // std::cout << "mat " << i << ": " << mat << std::endl;
        cv::Affine3d cv_pose(eigen_to_cv_mat(Eigen::Matrix4d(poses[i].matrix())));
        cv_pose = cv_pose.inv();
        // std::string matString = cv::format(cv::Mat(cv_pose.matrix), cv::Formatter::FMT_DEFAULT);
        // std::cout << "matString: " << matString << std::endl;
        window.showWidget(
            "pose_" + std::to_string(i), 
            cv::viz::WCoordinateSystem(pose_size),
            cv_pose
        );
    }
    constexpr double point_radius = 0.08;
    constexpr int sphere_res = 10;
    const cv::viz::Color point_color = cv::viz::Color::celestial_blue();
    for (unsigned int i = 0; i < points.size(); i++) {
        const Eigen::Vector3d &point = points[i];
        window.showWidget(
            "translation_" + std::to_string(i),
            cv::viz::WSphere(
                cv::Point3d(point[0], point[1], point[2]),
                point_radius,
                sphere_res,
                point_color
            )
        );
    }

    window.showWidget("world_frame", cv::viz::WCoordinateSystem());

    constexpr unsigned int num_cells = 6;
    constexpr double units = 1.0;
    const cv::viz::Color color = cv::viz::Color::white();
    const cv::Vec2i cells = cv::Vec2i::all(num_cells);
    const cv::Vec2d cell_sizes = cv::Vec2d::all(units);
    window.showWidget("xy_grid", cv::viz::WGrid(cells, cell_sizes, color));
    window.showWidget("yz_grid", cv::viz::WGrid(cv::Point3d(0.,0.,0.), cv::Point3d(1,0,0), cv::Point3d(0,0,1), cells, cell_sizes, color));
    window.showWidget("xz_grid", cv::viz::WGrid(cv::Point3d(0.,0.,0.), cv::Point3d(0,1,0), cv::Point3d(0,0,1), cells, cell_sizes, color));

    window.spin();
}
}
