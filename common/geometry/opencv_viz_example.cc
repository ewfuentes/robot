#include "opencv2/viz.hpp"

void demo() {
    cv::viz::Viz3d window("My Window");

    window.showWidget("world_frame", cv::viz::WCoordinateSystem());

    constexpr unsigned int num_cells = 6;
    constexpr double units = 1.0;
    const cv::viz::Color color = cv::viz::Color::white();
    const cv::Vec2i cells = cv::Vec2i::all(num_cells);
    const cv::Vec2d cell_sizes = cv::Vec2d::all(units);
    window.showWidget("xy_grid", cv::viz::WGrid(cells, cell_sizes, color));
    window.showWidget("yz_grid", cv::viz::WGrid(cv::Point3d(0., 0., 0.), cv::Point3d(1, 0, 0),
                                                cv::Point3d(0, 0, 1), cells, cell_sizes, color));
    window.showWidget("xz_grid", cv::viz::WGrid(cv::Point3d(0., 0., 0.), cv::Point3d(0, 1, 0),
                                                cv::Point3d(0, 0, 1), cells, cell_sizes, color));
    window.showWidget("text_overlay", cv::viz::WText("hello world overlay!", cv::Point(20, 50)));
    constexpr bool ALWAYS_FACE_CAMERA = true;
    constexpr double TEXT_SCALE = 0.1;
    window.showWidget("text_3d", cv::viz::WText3D("hello world 3d!", cv::Point3d(0.1, 0.2, 0.3),
                                                  TEXT_SCALE, ALWAYS_FACE_CAMERA));

    constexpr bool FIXED_TEXT = false;
    cv::Affine3d world_from_fixed_text(cv::Affine3d::Vec3{M_PI_2, 0.0, 0.0},  // rotation
                                       {0.2, 0.4, 0.6}                        // translation
    );

    window.showWidget(
        "text_3d_fixed",
        cv::viz::WText3D("hello world fixed!", cv::Point3d(0.0, 0.0, 0.0), TEXT_SCALE, FIXED_TEXT),
        world_from_fixed_text);
    constexpr double COORD_SCALE = 0.2;
    window.showWidget("text_3d_fixed_frame", cv::viz::WCoordinateSystem(COORD_SCALE),
                      world_from_fixed_text);

    constexpr double CIRCLE_RADIUS_M = 0.5;
    cv::Affine3d world_from_circle(cv::Affine3d::Vec3{0.0, 0.0, 0.0},  // rotation
                                   {0.0, 0.0, 1.5}                     // translation
    );
    window.showWidget("circle", cv::viz::WCircle(CIRCLE_RADIUS_M), world_from_circle);

    window.spin();
}

int main() { demo(); }
