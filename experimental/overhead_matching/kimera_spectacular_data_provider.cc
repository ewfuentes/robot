#include "experimental/overhead_matching/kimera_spectacular_data_provider.hh"

#include <gflags/gflags.h>
// #include <glog/logging.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/Pose3.h>

#include <algorithm>  // for max
#include <fstream>
#include <map>
#include <string>
#include <utility>  // for pair<>
#include <vector>

#include "kimera-vio/frontend/StereoFrame.h"
#include "kimera-vio/imu-frontend/ImuFrontend-definitions.h"
#include "kimera-vio/logging/Logger.h"
#include "kimera-vio/utils/YamlParser.h"

DEFINE_string(dataset_path, "/Users/Luca/data/MH_01_easy",
              "Path of dataset (i.e. Euroc, /Users/Luca/data/MH_01_easy).");
DEFINE_int64(initial_k, 0,
             "Initial frame to start processing dataset, "
             "previous frames will not be used.");
DEFINE_int64(final_k, 100000,
             "Final frame to finish processing dataset, "
             "subsequent frames will not be used.");

namespace robot::experimental::overhead_matching {

VIO::Timestamp vio_time_from_robot_time(const time::RobotTimestamp& robot_time) {
    /// VIO timestamp is int64_t nanoseconds. time::RobotTimestamp is std::nano
    return robot_time.time_since_epoch().count();
}

/* -------------------------------------------------------------------------- */
SpectacularDataProviderInterface::SpectacularDataProviderInterface(const std::string& dataset_path,
                                                                   const int& initial_k,
                                                                   const int& final_k,
                                                                   const VIO::VioParams& vio_params)
    : DataProviderInterface(),
      vio_params_(vio_params),
      dataset_path_(dataset_path),
      current_k_(std::numeric_limits<VIO::FrameId>::max()),
      initial_k_(initial_k),
      final_k_(final_k),
      imu_measurements_(),
      spec_log_(dataset_path) {
    // CHECK(!dataset_path_.empty()) << "Dataset path for SpectacularDataProvider is empty.";

    // Start processing dataset from frame initial_k.
    // Useful to skip a bunch of images at the beginning (imu calibration).
    CHECK_GE(initial_k_, 0);

    // Finish processing dataset at frame final_k.
    // Last frame to process (to avoid processing the entire dataset),
    // skip last frames.

    if (final_k_ > spec_log_.num_frames()) {
        LOG(WARNING) << "Provided final frame for KimeraSpectacularDataProvider was above the number of frames. Reducing to match size of dataset";
        final_k_ = spec_log_.num_frames();
    }

    CHECK_GT(final_k_, 0);

    CHECK_GT(final_k_, initial_k_)
        << "Value for final_k (" << final_k_ << ") is smaller than value for"
        << " initial_k (" << initial_k_ << ").";
    current_k_ = initial_k_;
}

/* -------------------------------------------------------------------------- */
SpectacularDataProviderInterface::SpectacularDataProviderInterface(const VIO::VioParams& vio_params)
    : SpectacularDataProviderInterface(FLAGS_dataset_path, FLAGS_initial_k, FLAGS_final_k,
                                       vio_params) {}

/* -------------------------------------------------------------------------- */
SpectacularDataProviderInterface::~SpectacularDataProviderInterface() {
    LOG(INFO) << "SpectacularDataProviderInterface destructor called.";
}

/* -------------------------------------------------------------------------- */
bool SpectacularDataProviderInterface::spin() {
    if (!is_imu_data_sent_) {
        // First, send all the IMU data. The flag is to avoid sending it several
        // times if we are running in sequential mode.
        if (imu_single_callback_) {
            sendImuData();
        } else {
            LOG(ERROR) << "Imu callback not registered! Not sending IMU data.";
        }
        is_imu_data_sent_ = true;
    }

    // Spin.
    CHECK_EQ(vio_params_.camera_params_.size(), 2u);
    CHECK_GT(final_k_, initial_k_);
    // We log only the first one, because we may be running in sequential mode.
    LOG_FIRST_N(INFO, 1) << "Running dataset between frame " << initial_k_ << " and frame "
                         << final_k_;
    while (!shutdown_ && spinOnce()) {
        if (!vio_params_.parallel_run_) {
            // Return, instead of blocking, when running in sequential mode.
            return true;
        }
    }
    LOG_IF(INFO, shutdown_) << "shutdown requested.";
    return false;
}

bool SpectacularDataProviderInterface::hasData() const { return current_k_ < final_k_; }

/* -------------------------------------------------------------------------- */
bool SpectacularDataProviderInterface::spinOnce() {
    CHECK_LT(current_k_, std::numeric_limits<VIO::FrameId>::max())
    //     << "Are you sure you've initialized current_k_?";
    if (current_k_ >= final_k_) {
        LOG(INFO) << "Finished spinning dataset.";
        return false;
    }
    //TODO: How do we want to handle camera parameters? 
    // const VIO::CameraParams& left_cam_info = vio_params_.camera_params_.at(0);
    const VIO::CameraParams left_cam_FAKE_PARAMS;
    const bool& equalize_image = false;
        // vio_params_.frontend_params_.stereo_matching_params_.equalize_image_;

    std::optional<robot::experimental::overhead_matching::FrameGroup> maybe_frame = spec_log_.get_frame(current_k_);
    // CHECK(maybe_frame.has_value());
    

    const VIO::Timestamp& timestamp_frame_k = vio_time_from_robot_time(maybe_frame->time_of_validity);
    // LOG(INFO) << "Sending left frames k= " << current_k_ << " with timestamp: " << timestamp_frame_k;

    cv::Mat bgr = maybe_frame->bgr_frame;
    if (bgr.channels() > 1) {
    //   LOG(INFO) << "Converting img from BGR to GRAY...";
      cv::cvtColor(bgr, bgr, cv::COLOR_BGR2GRAY);
    }
    if (equalize_image) {
        LOG(WARNING) << "- Histogram Equalization for image";
        cv::equalizeHist(bgr, bgr);
    }
       
    cv::Mat depth_meters = maybe_frame->depth_frame;
    CHECK(depth_meters.type() == CV_32FC1); // type assumed if depth is distance in meters

    // CHECK(left_frame_callback_);
    left_frame_callback_(std::make_unique<VIO::Frame>(
        current_k_, 
        timestamp_frame_k,
        left_cam_FAKE_PARAMS,
        bgr
    ));

    depth_frame_callback_(std::make_unique<VIO::DepthFrame>(
            current_k_,
            timestamp_frame_k,
            depth_meters
    ));

    // LOG(INFO) << "Finished VIO processing for frame k = " << current_k_;
    current_k_++;
    return true;
}


VIO::ImuMeasurement vio_imu_from_robot_imu(const ImuSample& robot_imu) {
    Eigen::Matrix<double, 6, 1> imu_acc_gyro;
    imu_acc_gyro << robot_imu.accel_mpss, robot_imu.gyro_radps;
    VIO::ImuMeasurement vio_imu(vio_time_from_robot_time(robot_imu.time_of_validity),
                                imu_acc_gyro);

    return vio_imu;
}

void SpectacularDataProviderInterface::sendImuData() const {
    CHECK(imu_single_callback_) << "Did you forget to register the IMU callback?";
    // for each imu measurement..

    const time::RobotTimestamp min_imu_time = spec_log_.min_imu_time();
    const time::RobotTimestamp max_imu_time = spec_log_.max_imu_time();
    // pull times from the accel spline
    const math::CubicHermiteSpline<Eigen::Vector3d> accel_spline = spec_log_.accel_spline();

    for (const double& t : accel_spline.ts()) {
        // exit if before first IMU time
        time::RobotTimestamp t_robot = time::RobotTimestamp() + time::as_duration(t);
        if (t < std::chrono::duration<double>(min_imu_time.time_since_epoch()).count()) {
            continue;
        }
        if (t > std::chrono::duration<double>(max_imu_time.time_since_epoch()).count()) {
            break;
        }
        std::optional<ImuSample> imu_sample = spec_log_.get_imu_sample(t_robot);
        CHECK(imu_sample.has_value());
        auto vio_imu = vio_imu_from_robot_imu(imu_sample.value());
        imu_single_callback_(vio_imu);
    }
}


std::string SpectacularDataProviderInterface::getDatasetName() {
    if (dataset_name_.empty()) {
        // Find and store actual name (rather than path) of the dataset.
        size_t found_last_slash = dataset_path_.find_last_of("/\\");
        std::string dataset_path_tmp = dataset_path_;
        dataset_name_ = dataset_path_tmp.substr(found_last_slash + 1);
        // The dataset name has a slash at the very end
        if (found_last_slash >= dataset_path_tmp.size() - 1) {
            // Cut the last slash.
            dataset_path_tmp = dataset_path_tmp.substr(0, found_last_slash);
            // Repeat the search.
            found_last_slash = dataset_path_tmp.find_last_of("/\\");
            // Try to pick right name.
            dataset_name_ = dataset_path_tmp.substr(found_last_slash + 1);
        }
        LOG(INFO) << "Dataset name: " << dataset_name_;
    }
    return dataset_name_;
}

/* -------------------------------------------------------------------------- */
size_t SpectacularDataProviderInterface::getNumImages() const {
    CHECK_GT(camera_names_.size(), 0u);
    const std::string& left_cam_name = camera_names_.at(0);
    size_t n_left_images = getNumImagesForCamera(left_cam_name);
    return n_left_images;
}

/* -------------------------------------------------------------------------- */
size_t SpectacularDataProviderInterface::getNumImagesForCamera(
    const std::string& camera_name) const {
    const auto& iter = camera_image_lists_.find(camera_name);
    CHECK(iter != camera_image_lists_.end());
    return iter->second.getNumImages();
}

/* -------------------------------------------------------------------------- */
bool SpectacularDataProviderInterface::getImgName(const std::string& camera_name, const size_t& k,
                                                  std::string* img_filename) const {
    CHECK_NOTNULL(img_filename);
    const auto& iter = camera_image_lists_.find(camera_name);
    CHECK(iter != camera_image_lists_.end());
    const auto& img_lists = iter->second.img_lists_;
    if (k < img_lists.size()) {
        *img_filename = img_lists.at(k).second;
        return true;
    } else {
        LOG(ERROR) << "Requested image #: " << k << " but we only have " << img_lists.size()
                   << " images.";
    }
    return false;
}

/* -------------------------------------------------------------------------- */
VIO::Timestamp SpectacularDataProviderInterface::timestampAtFrame(
    const VIO::FrameId& frame_number) {
    CHECK_GT(camera_names_.size(), 0);
    CHECK_LT(frame_number, camera_image_lists_.at(camera_names_[0]).img_lists_.size());
    return camera_image_lists_.at(camera_names_[0]).img_lists_[frame_number].first;
}

void SpectacularDataProviderInterface::clipFinalFrame() {
    // Clip final_k_ to the total number of images.
    const size_t& nr_images = getNumImages();
    if (final_k_ > nr_images) {
        LOG(WARNING) << "Value for final_k, " << final_k_ << " is larger than total"
                     << " number of frames in dataset " << nr_images;
        final_k_ = nr_images;
        LOG(WARNING) << "Using final_k = " << final_k_;
    }
    CHECK_LE(final_k_, nr_images);
}
/* -------------------------------------------------------------------------- */
void SpectacularDataProviderInterface::print() const {
    LOG(INFO) << "------------------ SpectacularDataProviderInterface::print ------------------\n"
              << "Displaying info for dataset: " << dataset_path_;
    // For each of the 2 cameras.
    CHECK_EQ(vio_params_.camera_params_.size(), camera_names_.size());
    for (size_t i = 0; i < camera_names_.size(); i++) {
        LOG(INFO) << "\n"
                  << (i == 0 ? "Left" : "Right") << " camera name: " << camera_names_[i]
                  << ", with params:\n";
        vio_params_.camera_params_.at(i).print();
        camera_image_lists_.at(camera_names_[i]).print();
    }
    LOG(INFO) << "-------------------------------------------------------------";
}
}  // namespace robot::experimental::overhead_matching
