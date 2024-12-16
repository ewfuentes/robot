#include "experimental/overhead_matching/kimera_spectacular_data_provider.hh"

#include <opencv2/core/core.hpp>
#include <string>
#include <vector>

#include "kimera-vio/imu-frontend/ImuFrontend-definitions.h"
#include "kimera-vio/logging/Logger.h"

namespace robot::experimental::overhead_matching {

VIO::Timestamp vio_time_from_robot_time(const time::RobotTimestamp& robot_time) {
    /// VIO timestamp is int64_t nanoseconds. time::RobotTimestamp is std::nano
    return robot_time.time_since_epoch().count();
}

/* -------------------------------------------------------------------------- */
SpectacularDataProviderInterface::SpectacularDataProviderInterface(const std::string& dataset_path,
                                                                   const int initial_k,
                                                                   const int final_k,
                                                                   const VIO::VioParams& vio_params)
    : DataProviderInterface(),
      vio_params_(vio_params),
      dataset_path_(dataset_path),
      current_k_(initial_k),
      final_k_(final_k),
      spec_log_(dataset_path) {
    ROBOT_CHECK(!dataset_path_.empty(), "Dataset path for SpectacularDataProvider is empty.");

    // Start processing dataset from frame initial_k.
    // Useful to skip a bunch of images at the beginning (imu calibration).
    ROBOT_CHECK(current_k_ >= 0);

    // Finish processing dataset at frame final_k.
    // Last frame to process (to avoid processing the entire dataset),
    // skip last frames.

    if (final_k_ > spec_log_.num_frames()) {
        LOG(WARNING) << "Provided final frame for KimeraSpectacularDataProvider was above the "
                        "number of frames. Reducing to match size of dataset";
        final_k_ = spec_log_.num_frames();
    }
    ROBOT_CHECK(final_k_ <= spec_log_.num_frames());

    ROBOT_CHECK(final_k_ > current_k_, "Value for final_k is smaller than value for current_k (initial value)",
                final_k_, current_k_);
}

SpectacularDataProviderInterface::~SpectacularDataProviderInterface() {
    LOG(INFO) << "SpectacularDataProviderInterface destructor called.";
}

bool SpectacularDataProviderInterface::spin() {
    if (!is_imu_data_sent_) {
        // First, send all the IMU data. The flag is to avoid sending it several
        // times if we are running in sequential mode.
        if (imu_single_callback_) {
            send_imu_data();
        } else {
            LOG(ERROR) << "Imu callback not registered! Not sending IMU data.";
        }
        is_imu_data_sent_ = true;
    }

    // Spin.
    // We log only the first one, because we may be running in sequential mode.
    LOG_FIRST_N(INFO, 1) << "Running dataset between frame " << current_k_ << " and frame "
                         << final_k_;
    while (!shutdown_ && spin_once()) {
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
bool SpectacularDataProviderInterface::spin_once() {
    if (current_k_ >= final_k_) {
        LOG(INFO) << "Finished spinning dataset.";
        return false;
    }
    // TODO: How do we want to handle camera parameters?
    //  const VIO::CameraParams& left_cam_info = vio_params_.camera_params_.at(0);
    const VIO::CameraParams left_cam_FAKE_PARAMS;
    const bool equalize_image = false;
    // vio_params_.frontend_params_.stereo_matching_params_.equalize_image_;

    std::optional<FrameGroup> maybe_frame = spec_log_.get_frame(current_k_);

    const VIO::Timestamp timestamp_frame_k =
        vio_time_from_robot_time(maybe_frame->time_of_validity);
    // LOG(INFO) << "Sending left frames k= " << current_k_ << " with timestamp: " <<
    // timestamp_frame_k;

    cv::Mat& bgr = maybe_frame->bgr_frame;
    if (bgr.channels() > 1) {
        //   LOG(INFO) << "Converting img from BGR to GRAY...";
        cv::cvtColor(bgr, bgr, cv::COLOR_BGR2GRAY);
    }
    if (equalize_image) {
        LOG(WARNING) << "- Histogram Equalization for image";
        cv::equalizeHist(bgr, bgr);
    }

    cv::Mat& depth_meters = maybe_frame->depth_frame;
    ROBOT_CHECK(depth_meters.type() == CV_32FC1);  // type assumed if depth is distance in meters

    left_frame_callback_(
        std::make_unique<VIO::Frame>(current_k_, timestamp_frame_k, left_cam_FAKE_PARAMS, bgr));

    depth_frame_callback_(
        std::make_unique<VIO::DepthFrame>(current_k_, timestamp_frame_k, depth_meters));

    // LOG(INFO) << "Finished VIO processing for frame k = " << current_k_;
    current_k_++;
    return true;
}

VIO::ImuMeasurement vio_imu_from_robot_imu(const ImuSample& robot_imu) {
    Eigen::Matrix<double, 6, 1> imu_acc_gyro;
    imu_acc_gyro << robot_imu.accel_mpss, robot_imu.gyro_radps;
    VIO::ImuMeasurement vio_imu(vio_time_from_robot_time(robot_imu.time_of_validity), imu_acc_gyro);

    return vio_imu;
}

void SpectacularDataProviderInterface::send_imu_data() const {
    ROBOT_CHECK(imu_single_callback_, "Did you forget to register the IMU callback?");
    // for each imu measurement..

    const time::RobotTimestamp min_imu_time = spec_log_.min_imu_time();
    const time::RobotTimestamp max_imu_time = spec_log_.max_imu_time();
    // pull times from the accel spline
    const math::CubicHermiteSpline<Eigen::Vector3d>& accel_spline = spec_log_.accel_spline();

    for (const double& t : accel_spline.ts()) {
        // exit if before first IMU time
        const time::RobotTimestamp t_robot = time::RobotTimestamp() + time::as_duration(t);
        if (t_robot < min_imu_time) {
            continue;
        }
        if (t_robot > max_imu_time) {
            break;
        }
        std::optional<ImuSample> imu_sample = spec_log_.get_imu_sample(t_robot);
        ROBOT_CHECK(imu_sample.has_value());
        auto vio_imu = vio_imu_from_robot_imu(imu_sample.value());
        imu_single_callback_(vio_imu);
    }
}

/* -------------------------------------------------------------------------- */
size_t SpectacularDataProviderInterface::get_num_images() const { return spec_log_.num_frames(); }

void SpectacularDataProviderInterface::print() const {
    LOG(INFO) << "------------------ SpectacularDataProviderInterface::print ------------------\n"
              << "Displaying info for dataset: " << dataset_path_;
    LOG(INFO) << "-------------------------------------------------------------";
}
}  // namespace robot::experimental::overhead_matching
