#pragma once
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
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
        if (symbols_to_idx_img_.find(symbol) != symbols_to_idx_img_.end()) {
            symbols_to_idx_img_.at(symbol).push_back(idx_img);
        } else {
            symbols_to_idx_img_.emplace(symbol, std::vector<size_t>{idx_img});
        }
    };

    std::optional<gtsam::Symbol> get_symbol(const size_t idx_img, const cv::KeyPoint &kpt) {
        return feature_sets_[idx_img].get_symbol(kpt);
    };

    size_t get_num_images_added() { return feature_sets_.size(); };

    std::unordered_map<gtsam::Symbol, std::vector<size_t>> get_added_symbols() {
        return symbols_to_idx_img_;
    };

    bool get_idxs_for_symbol(const gtsam::Symbol &symbol, std::vector<size_t> &out_cam_idxs) {
        if (symbols_to_idx_img_.find(symbol) != symbols_to_idx_img_.end()) {
            out_cam_idxs = symbols_to_idx_img_.at(symbol);
            return true;
        }
        return false;
    }

   private:
    std::vector<FeatureSet> feature_sets_;
    std::unordered_map<gtsam::Symbol, std::vector<size_t>> symbols_to_idx_img_;
};
}  // namespace robot::experimental::learn_descriptors