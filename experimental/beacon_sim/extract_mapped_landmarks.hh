
#pragma once

#include "experimental/beacon_sim/mapped_landmarks.hh"
#include "experimental/beacon_sim/ekf_slam.hh"

namespace robot::experimental::beacon_sim {
MappedLandmarks extract_mapped_landmarks(const EkfSlamEstimate &est);
}
