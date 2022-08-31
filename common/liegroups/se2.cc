
#include "common/liegroups/se2.hh"

namespace robot::liegroups {
double SE2::arclength() const { return log().head<2>().norm(); }
}  // namespace robot::liegroups
