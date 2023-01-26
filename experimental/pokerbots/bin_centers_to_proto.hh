
#pragma once

#include "experimental/pokerbots/bin_centers.hh"
#include "experimental/pokerbots/bin_centers.pb.h"

namespace robot::experimental::pokerbots::proto {
pokerbots::PerTurnBinCenters unpack_from(const PerTurnBinCenters &in);
pokerbots::BinCenter unpack_from(const BinCenter &in);
}  // namespace robot::experimental::pokerbots::proto
