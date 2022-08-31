
#include "common/liegroups/se3.hh"

namespace robot::liegroups {
double SE3::arclength() const { return log().head<TranslationType::RowsAtCompileTime>().norm(); }
}  // namespace robot::liegroups
