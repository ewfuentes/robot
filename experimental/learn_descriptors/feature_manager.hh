#pragma once
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "experimental/learn_descriptors/feature_set.hh"
#include "gtsam/inference/Symbol.h"
#include "opencv2/opencv.hpp"

namespace robot::experimental::learn_descriptors {
class FeatureManager {
   public:
    void append_img_data(const std::vector<cv::KeyPoint> &kpts, const cv::Mat &descriptors) {
        feature_sets_.push_back(FeatureSet(kpts, descriptors));
    };
    void insert_symbol(const size_t idx_img, const cv::KeyPoint &kpt, const gtsam::Symbol &symbol) {
        feature_sets_[idx_img].insert_symbol(kpt, symbol);
    };

    std::optional<gtsam::Symbol> get_symbol(const size_t idx_img, const cv::KeyPoint &kpt) {
        return feature_sets_[idx_img].get_symbol(kpt);
    };

    size_t get_num_images_added() { return feature_sets_.size(); };

   private:
    std::vector<FeatureSet> feature_sets_;
};
}  // namespace robot::experimental::learn_descriptors