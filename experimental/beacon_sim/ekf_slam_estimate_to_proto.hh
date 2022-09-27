
#pragma once

#include "experimental/beacon_sim/ekf_slam.hh"
#include "experimental/beacon_sim/ekf_slam_estimate.pb.h"

namespace robot::experimental::beacon_sim::proto {
  void pack_into(const beacon_sim::EkfSlamEstimate &in, EkfSlamEstimate *out);
  beacon_sim::EkfSlamEstimate unpack_from(const EkfSlamEstimate &in);
}
