
#include "opencv2/viz.hpp"

int main() {
    cv::viz::Viz3d window("My Window");

    window.showWidget("world_frame", cv::viz::WCoordinateSystem());
    window.showWidget("text_overlay", cv::viz::WText("hello world overlay!", cv::Point(20, 50)));
    constexpr bool ALWAYS_FACE_CAMERA = true;
    constexpr double TEXT_SCALE = 0.1;
    window.showWidget("text_3d", cv::viz::WText3D("hello world 3d!", cv::Point3d(0.1, 0.2, 0.3),
                                                  TEXT_SCALE, ALWAYS_FACE_CAMERA));

    constexpr bool FIXED_TEXT = false;
    cv::Affine3d world_from_fixed_text(
        cv::Affine3d::Vec3{M_PI_2, 0.0, 0.0}, // rotation
        {0.2, 0.4, 0.6} // translation
    );

    window.showWidget(
        "text_3d_fixed",
        cv::viz::WText3D("hello world fixed!", cv::Point3d(0.0, 0.0, 0.0), TEXT_SCALE, FIXED_TEXT),
        world_from_fixed_text
    );
    constexpr double COORD_SCALE = 0.2;
    window.showWidget("text_3d_fixed_frame", cv::viz::WCoordinateSystem(COORD_SCALE), world_from_fixed_text);

    constexpr double CIRCLE_RADIUS_M = 0.5;
    cv::Affine3d world_from_circle(
        cv::Affine3d::Vec3{0.0, 0.0, 0.0}, // rotation
        {0.0, 0.0, 1.5} // translation
    );
    window.showWidget(
        "circle",
        cv::viz::WCircle(CIRCLE_RADIUS_M),
        world_from_circle
    );

    window.spin();
}
