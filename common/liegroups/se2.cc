
#include "common/liegroups/se2.hh"

namespace robot::liegroups {
double SE2::arclength() const { return log().head<TranslationType::RowsAtCompileTime>().norm(); }
}  // namespace robot::liegroups
