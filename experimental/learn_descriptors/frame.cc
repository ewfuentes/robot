#include "experimental/learn_descriptors/frame.hh"

namespace robot::experimental::learn_descriptors {
void Frame::add_keypoints(const KeypointsCV& kpts) {
    kpts_.insert(kpts_.end(), kpts.begin(), kpts.end());
};

void Frame::assign_descriptors(const cv::Mat& descriptors) { descriptors_ = descriptors; };
}  // namespace robot::experimental::learn_descriptors