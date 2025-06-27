#include "open3d/Open3D.h"

int main() {
    open3d::geometry::PointCloud pcd;
    pcd.points_.push_back({0.0, 0.0, 0.0});
    pcd.points_.push_back({1.0, 0.0, 0.0});
    pcd.points_.push_back({0.0, 1.0, 0.0});

    open3d::visualization::Visualizer vis;
    vis.CreateVisualizerWindow("My Viz", 800, 600);
    vis.AddGeometry(std::make_shared<open3d::geometry::PointCloud>(pcd));
    vis.Run();
    vis.DestroyVisualizerWindow();
    return 0;
}