#pragma once

#include <cstdint>
#include <vector>

#include "gtsam/geometry/Point3.h"
#include "opencv2/opencv.hpp"

using Timestamp = std::int64_t;

using FrameId = std::uint64_t;  // Frame id is used as the index of gtsam symbol
                                // (not as a gtsam key).
using LandmarkId = long;        // -1 for invalid landmarks
using LandmarkIds = std::vector<LandmarkId>;
using Landmark = gtsam::Point3;
using Landmarks = std::vector<Landmark, Eigen::aligned_allocator<Landmark>>;

using KeypointCV = cv::Point2f;
using KeypointsCV = std::vector<cv::Point2f>;
