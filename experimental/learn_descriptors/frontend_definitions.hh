#pragma once

#include <ostream>
#include <unordered_map>
#include <vector>

#include "experimental/learn_descriptors/structure_from_motion_types.hh"
#include "opencv2/opencv.hpp"

namespace std {
template <>
struct hash<std::pair<FrameId, KeypointCV>> {
    size_t operator()(const std::pair<FrameId, KeypointCV> &p) const {
        size_t h1 = std::hash<FrameId>()(p.first);
        size_t h2 = std::hash<float>()(p.second.x) ^ (std::hash<float>()(p.second.y) << 1);
        return h1 ^ (h2 << 1);  // combine hashes
    }
};
}  // namespace std

namespace robot::experimental::learn_descriptors {
class FeatureTrack {
   public:
    std::vector<std::pair<FrameId, KeypointCV>> obs_;

    bool in_ba_graph_;

    FeatureTrack(FrameId frame_id, const KeypointCV &px) { obs_.emplace_back(frame_id, px); }

    void print() const {
        std::cout << "Feature track with cameras: ";
        for (size_t i = 0u; i < obs_.size(); i++) {
            std::cout << " " << obs_[i].first << " ";
        }
        std::cout << std::endl;
    }
};

using FeatureTracks = std::vector<FeatureTrack>;  // each idx is the FeatureTracks's LandmarkId
using FrameLandmarkIdMap = std::unordered_map<std::pair<FrameId, KeypointCV>, LandmarkId>;
}  // namespace robot::experimental::learn_descriptors
