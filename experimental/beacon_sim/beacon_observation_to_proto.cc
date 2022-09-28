
#include "experimental/beacon_sim/beacon_observation_to_proto.hh"

namespace robot::experimental::beacon_sim::proto {

void pack_into(const beacon_sim::BeaconObservation &in, BeaconObservation *out) {
    if (in.maybe_id.has_value()) {
        out->set_id(in.maybe_id.value());
    }

    if (in.maybe_range_m.has_value()) {
        out->set_range_m(in.maybe_range_m.value());
    }

    if (in.maybe_bearing_rad.has_value()) {
        out->set_bearing_rad(in.maybe_bearing_rad.value());
    }
}

beacon_sim::BeaconObservation unpack_from(const BeaconObservation &in) {
    beacon_sim::BeaconObservation out;
    if (in.has_id()) {
        out.maybe_id = in.id();
    }

    if (in.has_range_m()) {
        out.maybe_range_m = in.range_m();
    }

    if (in.has_bearing_rad()) {
        out.maybe_bearing_rad = in.bearing_rad();
    }
    return out;
}

void pack_into(const std::vector<beacon_sim::BeaconObservation> &in, BeaconObservations *out) {
    out->clear_observations();
    for (const auto &obs : in) {
        pack_into(obs, out->add_observations());
    }
}

std::vector<beacon_sim::BeaconObservation> unpack_from(const BeaconObservations &in) {
    std::vector<beacon_sim::BeaconObservation> out;
    out.reserve(in.observations_size());

    for (const auto &obs : in.observations()) {
        out.push_back(unpack_from(obs));
    }
    return out;
}
}  // namespace robot::experimental::beacon_sim::proto
