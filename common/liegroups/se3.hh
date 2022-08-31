#include "sophus/se3.hpp"

namespace robot::liegroups {

class SE3 : public Sophus::SE3d {
   public:
    using Sophus::SE3d::SE3d;

    double arclength() const;
};
}  // namespace robot::liegroups
