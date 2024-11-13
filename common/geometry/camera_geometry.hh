
#include "Eigen/Core"

namespace robot::geometry {

Eigen::Vector2d project_point_to_camera_pixel(const Eigen::Vector3d& point3D, double fx, double fy, double cx, double cy) {
    double u = fx * (point3D.x() / point3D.z()) + cx;
    double v = fy * (point3D.y() / point3D.z()) + cy;

    return Eigen::Vector2d(u, v);
}

Eigen::Vector3d project_camera_pixel_to_3D(const Eigen::Vector2d& pixel, double depth, double fx, double fy, double cx, double cy) {
    double X = (pixel.x() - cx) * depth / fx;
    double Y = (pixel.y() - cy) * depth / fy;
    double Z = depth;

    return Eigen::Vector3d(X, Y, Z);
}
}