

#include "experimental/beacon_sim/world_map_config_to_proto.hh"

#include "experimental/beacon_sim/beacon_potential_to_proto.hh"
#include "experimental/beacon_sim/world_map.hh"

namespace robot::experimental::beacon_sim::proto {
void pack_into(const beacon_sim::Beacon &in, Beacon *out) {
    out->set_pos_x_m(in.pos_in_local.x());
    out->set_pos_y_m(in.pos_in_local.y());
    out->set_id(in.id);
}

beacon_sim::Beacon unpack_from(const Beacon &in) {
    return beacon_sim::Beacon{
        .id = in.id(),
        .pos_in_local = {in.pos_x_m(), in.pos_y_m()},
    };
}

void pack_into(const beacon_sim::FixedBeaconsConfig &in, FixedBeaconsConfig *out) {
    out->mutable_beacons()->Clear();

    for (const auto &beacon : in.beacons) {
        pack_into(beacon, out->mutable_beacons()->Add());
    }
}

beacon_sim::FixedBeaconsConfig unpack_from(const FixedBeaconsConfig &in) {
    beacon_sim::FixedBeaconsConfig out;

    out.beacons.reserve(in.beacons_size());
    for (const auto &proto_beacon : in.beacons()) {
        out.beacons.push_back(unpack_from(proto_beacon));
    }
    return out;
}

void pack_into(const beacon_sim::BlinkingBeaconsConfig &in, BlinkingBeaconsConfig *out) {
    out->mutable_beacons()->Clear();

    for (const auto &beacon : in.beacons) {
        pack_into(beacon, out->mutable_beacons()->Add());
    }

    out->set_beacon_appear_rate_hz(in.beacon_appear_rate_hz);
    out->set_beacon_disappear_rate_hz(in.beacon_disappear_rate_hz);
}

beacon_sim::BlinkingBeaconsConfig unpack_from(const BlinkingBeaconsConfig &in) {
    beacon_sim::BlinkingBeaconsConfig out;

    out.beacons.reserve(in.beacons_size());
    for (const auto &proto_beacon : in.beacons()) {
        out.beacons.push_back(unpack_from(proto_beacon));
    }

    out.beacon_appear_rate_hz = in.beacon_appear_rate_hz();
    out.beacon_disappear_rate_hz = in.beacon_disappear_rate_hz();

    return out;
}

void pack_into(const beacon_sim::CorrelatedBeaconsConfig &in, CorrelatedBeaconsConfig *out) {
    for (const auto &beacon : in.beacons) {
        pack_into(beacon, out->add_beacons());
    }

    pack_into(in.potential, out->mutable_potential());
    if (in.configuration.has_value()) {
        const int num_bytes = (in.configuration->size() + 8) / 8;
        std::vector<char> bytes(num_bytes, 0);
        for (int i = 0; i < static_cast<int>(in.configuration->size()); i++) {
            const int byte_idx = i / 8;
            const int bit_idx = i % 8;

            if (in.configuration->at(i)) {
                bytes.at(byte_idx) |= (1 << bit_idx);
            }
        }
        out->set_configuration(bytes.data(), bytes.size());
    }
}

beacon_sim::CorrelatedBeaconsConfig unpack_from(const CorrelatedBeaconsConfig &in) {
    std::vector<beacon_sim::Beacon> beacons;
    beacons.reserve(in.beacons_size());
    for (const auto &proto_beacon : in.beacons()) {
        beacons.push_back(unpack_from(proto_beacon));
    }

    std::optional<std::vector<bool>> configuration;
    if (in.has_configuration()) {
        configuration = std::vector<bool>();
        configuration->reserve(in.beacons_size());
        for (const auto &c : in.configuration()) {
            for (int i = 0; i < 8; i++) {
                configuration->push_back(c & (1 << i));
            }
        }
    }

    return {
        .beacons = std::move(beacons),
        .potential = unpack_from(in.potential()),
        .configuration = configuration,
    };
}

void pack_into(const beacon_sim::WorldMapConfig &in, WorldMapConfig *out) {
    pack_into(in.fixed_beacons, out->mutable_fixed_beacons());
    pack_into(in.blinking_beacons, out->mutable_blinking_beacons());
    pack_into(in.correlated_beacons, out->mutable_correlated_beacons());
}

beacon_sim::WorldMapConfig unpack_from(const WorldMapConfig &in) {
    return beacon_sim::WorldMapConfig{
        .fixed_beacons = unpack_from(in.fixed_beacons()),
        .blinking_beacons = unpack_from(in.blinking_beacons()),
        .correlated_beacons = unpack_from(in.correlated_beacons()),
        // TODO add pack/unpack for obstacles
        .obstacles = {},
    };
}
}  // namespace robot::experimental::beacon_sim::proto
