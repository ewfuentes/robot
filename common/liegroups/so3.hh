
#pragma once

#include "sophus/so3.hpp"

namespace robot::liegroups {
class SO3 : public Sophus::SO3d {
    using Sophus::SO3d::SO3d;
};
}  // namespace robot::liegroups
