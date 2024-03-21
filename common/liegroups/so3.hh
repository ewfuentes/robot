
#pragma once

#include "sophus/so3.hpp"

namespace robot::liegroups {
class SO3 : public Sophus::SO3d {
public:
    using Sophus::SO3d::SO3d;
    SO3(const Sophus::SO3d &other) : Sophus::SO3d(other) {}
};
}  // namespace robot::liegroups
