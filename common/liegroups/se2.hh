
#pragma once

#include "common/liegroups/so2.hh"
#include "sophus/se2.hpp"

namespace robot::liegroups {

class SE2 : public Sophus::SE2d {
   public:
    using Sophus::SE2d::SE2d;

    double arclength() const;
};
}  // namespace robot::liegroups
