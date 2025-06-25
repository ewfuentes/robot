#pragma once
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "gtsam/geometry/Pose3.h"
#include "gtsam/inference/Symbol.h"
#include "opencv2/opencv.hpp"

namespace std {
template <>
struct hash<cv::KeyPoint> {
    size_t operator()(const cv::KeyPoint &kp) const {
        size_t h1 = hash<float>()(kp.pt.x);
        size_t h2 = hash<float>()(kp.pt.y);
        size_t h3 = hash<float>()(kp.size);
        size_t h4 = hash<float>()(kp.angle);
        size_t h5 = hash<float>()(kp.response);
        size_t h6 = hash<int>()(kp.octave);
        size_t h7 = hash<int>()(kp.class_id);

        return (h1 ^ (h2 << 1)) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4) ^ (h6 << 5) ^ (h7 << 6);
    }
};
template <>
struct hash<gtsam::Symbol> {
    size_t operator()(const gtsam::Symbol &s) const noexcept {
        return std::hash<size_t>()(s.key());
    }
};
}  // namespace std
struct KeyPointEqual {
    bool operator()(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2) const {
        return kp1.pt == kp2.pt && kp1.size == kp2.size && kp1.angle == kp2.angle &&
               kp1.response == kp2.response && kp1.octave == kp2.octave &&
               kp1.class_id == kp2.class_id;
    }
};

namespace robot::experimental::learn_descriptors {
class FeatureSet {
   public:
    struct keypoints_descriptors {
        keypoints_descriptors(const std::vector<cv::KeyPoint> &keypoints,
                              const cv::Mat &descriptors)
            : kpts(keypoints), descriptors(descriptors) {}
        std::vector<cv::KeyPoint> kpts;
        cv::Mat descriptors;
    };

    FeatureSet(const std::vector<cv::KeyPoint> &kpts, const cv::Mat &descriptors)
        : kpts_descriptors_(keypoints_descriptors(kpts, descriptors)){};
    ~FeatureSet() = default;

    void insert_symbol(const cv::KeyPoint &kpt, const gtsam::Symbol &symbol) {
        if (kpt_to_symbol_.find(kpt) != kpt_to_symbol_.end()) {
            std::stringstream error_msg;
            error_msg << "Keypoint already has symbol! Keypoint has symbol: "
                      << kpt_to_symbol_.at(kpt);
            throw std::runtime_error(error_msg.str());
        }
        kpt_to_symbol_.emplace(kpt, symbol);
    };
    std::optional<gtsam::Symbol> get_symbol(const cv::KeyPoint &kpt) {
        if (kpt_to_symbol_.find(kpt) != kpt_to_symbol_.end()) {
            return kpt_to_symbol_.at(kpt);
        }
        return std::nullopt;
    };

   private:
    keypoints_descriptors kpts_descriptors_;
    std::unordered_map<cv::KeyPoint, gtsam::Symbol, std::hash<cv::KeyPoint>, KeyPointEqual>
        kpt_to_symbol_;
    std::unordered_map<gtsam::Symbol, cv::KeyPoint> symbol_to_kpt_;
};
}  // namespace robot::experimental::learn_descriptors