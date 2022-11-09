
#include "experimental/beacon_sim/visualize_beacon_sim.hh"

#include <GL/gl.h>
#include <GL/glu.h>

#include <iostream>

#include "common/liegroups/se3.hh"

namespace robot::experimental::beacon_sim {
namespace {
liegroups::SE3 se3_from_se2(const liegroups::SE2 &a_from_b) {
    Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
    Eigen::Matrix3d a_from_b_mat = a_from_b.matrix();
    mat.topLeftCorner(2, 2) = a_from_b_mat.topLeftCorner(2, 2);
    mat.topRightCorner(2, 1) = a_from_b_mat.topRightCorner(2, 1);
    return liegroups::SE3(mat);
}
}  // namespace

void visualize_beacon_sim(const BeaconSimState &state, const double zoom_factor,
                          const double window_aspect_ratio) {
    constexpr double BEACON_HALF_WIDTH_M = 0.25;
    constexpr double ROBOT_SIZE_M = 0.5;
    constexpr double DEG_FROM_RAD = 180.0 / std::numbers::pi;
    constexpr double DEFAULT_WIDTH_M = 15;
    const double window_width_m = DEFAULT_WIDTH_M * zoom_factor;
    const double window_height_m = window_width_m * window_aspect_ratio;

    const auto gl_error = glGetError();
    if (gl_error != GL_NO_ERROR) {
        std::cout << "GL ERROR: " << gl_error << ": " << gluErrorString(gl_error) << std::endl;
    }
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-window_width_m, window_width_m, -window_height_m, window_height_m, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Draw beacons
    for (const auto &beacon : state.map.visible_beacons(state.time_of_validity)) {
        glBegin(GL_LINE_LOOP);
        glColor4f(1.0, 0.0, 0.0, 0.0);
        for (const auto &corner :
             std::array<Eigen::Vector2d, 4>{{{-1.0, -1.0}, {-1.0, 1.0}, {1.0, 1.0}, {1.0, -1.0}}}) {
            const Eigen::Vector2d point_in_local =
                beacon.pos_in_local + corner * BEACON_HALF_WIDTH_M;
            glVertex2d(point_in_local.x(), point_in_local.y());
        }
        glEnd();
    }

    // Draw robot
    glPushMatrix();
    glTranslated(state.robot.pos_x_m(), state.robot.pos_y_m(), 0.0);
    glRotated(DEG_FROM_RAD * state.robot.heading_rad(), 0.0, 0.0, 1.0);
    glBegin(GL_LINE_LOOP);
    glColor4f(0.5, 0.5, 1.0, 1.0);
    for (const auto &[dx, dy] :
         std::array<std::pair<double, double>, 3>{{{0.0, 0.5}, {1.5, 0.0}, {0.0, -0.5}}}) {
        const double x_in_robot_m = dx * ROBOT_SIZE_M;
        const double y_in_robot_m = dy * ROBOT_SIZE_M;
        glVertex2d(x_in_robot_m, y_in_robot_m);
    }
    glEnd();

    for (const auto &obs : state.observations) {
        glPushMatrix();
        glRotated(DEG_FROM_RAD * obs.maybe_bearing_rad.value(), 0.0, 0.0, 1.0);
        glBegin(GL_LINES);
        glColor4f(0.5, 1.0, 0.5, 1.0);
        glVertex2d(0.0, 0.0);
        glVertex2d(obs.maybe_range_m.value(), 0.0);
        glEnd();
        glPopMatrix();  // Pop from Measurement Frame to robot frame
    }

    glPopMatrix();  // Pop from Robot Frame to world frame

    // Draw Obstacles
    {
        for (const auto &obstacle : state.map.obstacles()) {
            glBegin(GL_LINE_LOOP);
            if (obstacle.is_inside(state.robot.local_from_robot().translation())) {
                glColor4ub(168, 50, 50, 255);
            } else {
                glColor4ub(124, 187, 235, 255);
            }
            for (const Eigen::Vector2d &pt : obstacle.pts_in_frame()) {
                glVertex2d(pt.x(), pt.y());
            }
            glEnd();
        }
    }

    // Draw ekf estimates
    const EkfSlamEstimate estimate = state.ekf.estimate();
    {
        glPushMatrix();
        const liegroups::SE3 est_local_from_robot = se3_from_se2(estimate.local_from_robot());
        glMultMatrixd(est_local_from_robot.matrix().data());

        glBegin(GL_LINE_LOOP);
        glColor4f(0.75, 0.75, 1.0, 1.0);
        for (const auto &[dx, dy] :
             std::array<std::pair<double, double>, 3>{{{0.0, 0.5}, {1.5, 0.0}, {0.0, -0.5}}}) {
            const double x_in_robot_m = dx * ROBOT_SIZE_M;
            const double y_in_robot_m = dy * ROBOT_SIZE_M;
            glVertex2d(x_in_robot_m, y_in_robot_m);
        }
        glEnd();
        glPopMatrix();  // Pop from estimated robot frame to world frame

        const Eigen::Matrix3d pos_cov = estimate.robot_cov();
        const Eigen::LLT<Eigen::Matrix3d> cov_llt(pos_cov);

        glBegin(GL_LINE_LOOP);
        glColor4f(1.0, 0.5, 0.5, 1.0);
        for (double theta = 0.0; theta <= 2 * std::numbers::pi; theta += 0.005) {
            const Eigen::Vector3d tangent_vec =
                cov_llt.matrixL() *
                Eigen::Vector3d{2.0 * std::cos(theta), 2.0 * std::sin(theta), 0.0};
            const liegroups::SE2 local_from_ellipse_pt =
                estimate.local_from_robot() * liegroups::SE2::exp(tangent_vec);
            const Eigen::Vector2d pt = local_from_ellipse_pt.translation();

            glVertex2d(pt.x(), pt.y());
        }
        glEnd();
    }

    for (const auto beacon_id : estimate.beacon_ids) {
        glPushMatrix();
        const liegroups::SE3 local_from_beacon =
            se3_from_se2(liegroups::SE2::trans(estimate.beacon_in_local(beacon_id).value()));
        glMultMatrixd(local_from_beacon.matrix().data());
        const Eigen::Matrix2d pos_cov = estimate.beacon_cov(beacon_id).value();
        const Eigen::LLT<Eigen::Matrix2d> cov_llt(pos_cov);

        glBegin(GL_LINE_LOOP);
        glColor4f(0.75, 0.75, 1.0, 1.0);
        for (double theta = 0; theta < 2 * std::numbers::pi; theta += 0.05) {
            const Eigen::Vector2d pt =
                cov_llt.matrixL() * Eigen::Vector2d{2.0 * std::cos(theta), 2.0 * std::sin(theta)};
            glVertex2d(pt.x(), pt.y());
        }
        glEnd();
        glPopMatrix();  // Pop from beacon frame to world frame
    }

    // Draw Road map
    int num_points = state.road_map.points.size();
    for (int i = 0; i < num_points; i++) {
        // Draw the node
        const Eigen::Vector2d &pt = state.road_map.points.at(i);
        glPushMatrix();
        const liegroups::SE3 local_from_node = se3_from_se2(liegroups::SE2::trans(pt));
        glMultMatrixd(local_from_node.matrix().data());
        glBegin(GL_LINE_LOOP);
        glColor3f(0.0, 0.5, 0.5);
        for (const auto &corner :
             std::array<Eigen::Vector2d, 4>{{{-1.0, -1.0}, {-1.0, 1.0}, {1.0, 1.0}, {1.0, -1.0}}}) {
            constexpr double NODE_HALF_WIDTH_M = 0.25 / 2.0;
            const Eigen::Vector2d corner_in_node = corner * NODE_HALF_WIDTH_M;
            glVertex2d(corner_in_node.x(), corner_in_node.y());
        }
        glEnd();

        glPopMatrix();  // Pop from road map node frame to world frame

        for (int j = i + 1; j < num_points; j++) {
            if (state.road_map.adj(i, j)) {
                // Draw an edge between the two points
                const Eigen::Vector2d other_pt = state.road_map.points.at(j);
                glColor3f(0.4, 0.4, 0.4);
                glBegin(GL_LINES);
                glVertex2d(pt.x(), pt.y());
                glVertex2d(other_pt.x(), other_pt.y());
                glEnd();
            }
        }
    }
}
}  // namespace robot::experimental::beacon_sim
