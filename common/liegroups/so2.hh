
#pragma once

#include "sophus/so2.hpp"

namespace robot::liegroups {
class SO2 : public Sophus::SO2d {
    using Sophus::SO2d::SO2d;
};
}  // namespace robot::liegroups
