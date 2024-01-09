
#include "experimental/beacon_sim/beacon_potential.pb.h"
#include "experimental/beacon_sim/precision_matrix_potential.hh"
#include "experimental/beacon_sim/precision_matrix_potential.pb.h"

namespace robot::experimental::beacon_sim {
namespace proto {
void pack_into(const beacon_sim::PrecisionMatrixPotential &in, PrecisionMatrixPotential *out);
beacon_sim::PrecisionMatrixPotential unpack_from(const PrecisionMatrixPotential &in);
}  // namespace proto

void pack_into_potential(const PrecisionMatrixPotential &in, proto::BeaconPotential *out);
}  // namespace robot::experimental::beacon_sim
