#include "experimental/overhead_matching/kimera_spectacular_data_provider.hh"

#include <algorithm>  // for max
#include <fstream>
#include <map>
#include <string>
#include <utility>  // for pair<>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gflags/gflags.h>

#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/Pose3.h>

#include "kimera-vio/frontend/StereoFrame.h"
#include "kimera-vio/imu-frontend/ImuFrontend-definitions.h"
#include "kimera-vio/logging/Logger.h"
#include "kimera-vio/utils/YamlParser.h"

DEFINE_string(dataset_path,
              "/Users/Luca/data/MH_01_easy",
              "Path of dataset (i.e. Euroc, /Users/Luca/data/MH_01_easy).");
DEFINE_int64(initial_k,
             0,
             "Initial frame to start processing dataset, "
             "previous frames will not be used.");
DEFINE_int64(final_k,
             100000,
             "Final frame to finish processing dataset, "
             "subsequent frames will not be used.");

namespace robot::experimental::overhead_matching {

SpectacularLogDataProvider::SpectacularLogDataProvider(const std::string& dataset_path,
                                     const int& initial_k,
                                     const int& final_k,
                                     const VioParams& vio_params)
    : DataProviderInterface(),
      vio_params_(vio_params),
      dataset_path_(dataset_path),
      current_k_(std::numeric_limits<FrameId>::max()),
      initial_k_(initial_k),
      final_k_(final_k),
      imu_measurements_()) {

  ROBOT_CHECK(!dataset_path_.empty())
      << "Dataset path for SpectacularLogDataProvider is empty.";
  // Start processing dataset from frame initial_k.
  // Useful to skip a bunch of images at the beginning (imu calibration).
  ROBOT_CHECK_GE(initial_k_, 0);

  // Finish processing dataset at frame final_k.
  // Last frame to process (to avoid processing the entire dataset),
  // skip last frames.
  ROBOT_CHECK_GT(final_k_, 0);

  ROBOT_CHECK_GT(final_k_, initial_k_) << "Value for final_k (" << final_k_
                                 << ") is smaller than value for"
                                 << " initial_k (" << initial_k_ << ").";
  current_k_ = initial_k_;

  // Parse the actual dataset first, then run it.
  if (!shutdown_ && !dataset_parsed_) {
    LOG(INFO) << "Parsing Spectacular dataset...";
    parse();
    ROBOT_CHECK_GT(imu_measurements_.size(), 0u);
    dataset_parsed_ = true;
  }
}

/* -------------------------------------------------------------------------- */
SpectacularLogDataProvider::SpectacularLogDataProvider(const VioParams& vio_params)
    : SpectacularLogDataProvider(FLAGS_dataset_path,
                        FLAGS_initial_k,
                        FLAGS_final_k,
                        vio_params) {}

/* -------------------------------------------------------------------------- */
SpectacularLogDataProvider::~SpectacularLogDataProvider() {
  LOG(INFO) << "SpectacularLogDataProvider destructor called.";
}

/* -------------------------------------------------------------------------- */
bool SpectacularLogDataProvider::spin() {
  if (dataset_parsed_) {
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
    ROBOT_CHECK_EQ(vio_params_.camera_params_.size(), 2u);
    ROBOT_CHECK_GT(final_k_, initial_k_);
    // We log only the first one, because we may be running in sequential mode.
    LOG_FIRST_N(INFO, 1) << "Running dataset between frame " << initial_k_
                         << " and frame " << final_k_;
    while (!shutdown_ && spinOnce()) {
      if (!vio_params_.parallel_run_) {
        // Return, instead of blocking, when running in sequential mode.
        return true;
      }
    }
  } else {
    LOG(ERROR) << "Euroc dataset was not parsed.";
  }
  LOG_IF(INFO, shutdown_) << "SpectacularLogDataProvider shutdown requested.";
  return false;
}

bool SpectacularLogDataProvider::hasData() const {
  return current_k_ < final_k_;
}

/* -------------------------------------------------------------------------- */
bool SpectacularLogDataProvider::spinOnce() {
  ROBOT_CHECK_LT(current_k_, std::numeric_limits<FrameId>::max())
      << "Are you sure you've initialized current_k_?";
  if (current_k_ >= final_k_) {
    LOG(INFO) << "Finished spinning Euroc dataset.";
    return false;
  }

  const CameraParams& left_cam_info = vio_params_.camera_params_.at(0);
  const CameraParams& right_cam_info = vio_params_.camera_params_.at(1);
  const bool& equalize_image =
      vio_params_.frontend_params_.stereo_matching_params_.equalize_image_;

  const Timestamp& timestamp_frame_k = timestampAtFrame(current_k_);
  VLOG(10) << "Sending left/right frames k= " << current_k_
           << " with timestamp: " << timestamp_frame_k;

  // TODO(Toni): ideally only send cv::Mat raw images...:
  // - pass params to vio_pipeline ctor
  // - make vio_pipeline actually equalize or transform images as necessary.
  std::string left_img_filename;
  bool available_left_img = getLeftImgName(current_k_, &left_img_filename);
  std::string right_img_filename;
  bool available_right_img = getRightImgName(current_k_, &right_img_filename);
  if (available_left_img && available_right_img) {
    // Both stereo images are available, send data to VIO
    ROBOT_CHECK(left_frame_callback_);
    left_frame_callback_(
        std::make_unique<Frame>(current_k_,
                                timestamp_frame_k,
                                // TODO(Toni): this info should be passed to
                                // the camera... not all the time here...
                                left_cam_info,
                                UtilsOpenCV::ReadAndConvertToGrayScale(
                                    left_img_filename, equalize_image)));
    ROBOT_CHECK(right_frame_callback_);
    right_frame_callback_(
        std::make_unique<Frame>(current_k_,
                                timestamp_frame_k,
                                // TODO(Toni): this info should be passed to
                                // the camera... not all the time here...
                                right_cam_info,
                                UtilsOpenCV::ReadAndConvertToGrayScale(
                                    right_img_filename, equalize_image)));
  } else {
    LOG(ERROR) << "Missing left/right stereo pair, proceeding to the next one.";
  }

  // This is done directly when parsing the Imu data.
  // imu_single_callback_(imu_meas);

  VLOG(10) << "Finished VIO processing for frame k = " << current_k_;
  current_k_++;
  return true;
}

void SpectacularLogDataProvider::sendImuData() const {
  ROBOT_CHECK(imu_single_callback_) << "Did you forget to register the IMU callback?";
  Timestamp previous_timestamp = -1;
  for (const ImuMeasurement& imu_meas : imu_measurements_) {
    ROBOT_CHECK_GT(imu_meas.timestamp_, previous_timestamp)
        << "Euroc IMU data is not in chronological order!";
    previous_timestamp = imu_meas.timestamp_;
    imu_single_callback_(imu_meas);
  }
}

/* -------------------------------------------------------------------------- */
void SpectacularLogDataProvider::parse() {
  VLOG(100) << "Using dataset path: " << dataset_path_;
  // Parse the dataset (ETH format).
  parseDataset();
  if (VLOG_IS_ON(1)) print();

  // Send first ground-truth pose to VIO for initialization if requested.
  if (vio_params_.backend_params_->autoInitialize_ == 0) {
    // We want to initialize from ground-truth.
    vio_params_.backend_params_->initial_ground_truth_state_ =
        getGroundTruthState(timestampAtFrame(initial_k_));
  }
}

/* -------------------------------------------------------------------------- */
bool SpectacularLogDataProvider::parseImuData(const std::string& input_dataset_path,
                                     const std::string& imuName) {
  ///////////////// PARSE ACTUAL DATA //////////////////////////////////////////
  //#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],
  // a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]
  std::string filename_data =
      input_dataset_path + "/mav0/" + imuName + "/data.csv";
  std::ifstream fin(filename_data.c_str());
  LOG_IF(FATAL, !fin.is_open()) << "Cannot open file: " << filename_data;

  // Skip the first line, containing the header.
  std::string line;
  std::getline(fin, line);

  size_t deltaCount = 0u;
  Timestamp sumOfDelta = 0;
  double stdDelta = 0;
  double imu_rate_maxMismatch = 0;
  double maxNormAcc = 0, maxNormRotRate = 0;  // only for debugging
  Timestamp previous_timestamp = -1;

  // Read/store imu measurements, line by line.
  while (std::getline(fin, line)) {
    Timestamp timestamp = 0;
    gtsam::Vector6 gyr_acc_data;
    for (int i = 0u; i < gyr_acc_data.size() + 1u; i++) {
      int idx = line.find_first_of(',');
      if (i == 0) {
        timestamp = std::stoll(line.substr(0, idx));
      } else {
        gyr_acc_data(i - 1) = std::stod(line.substr(0, idx));
      }
      line = line.substr(idx + 1);
    }
    ROBOT_CHECK_GT(timestamp, previous_timestamp)
        << "Euroc IMU data is not in chronological order!";
    Vector6 imu_accgyr;
    // Acceleration first!
    imu_accgyr << gyr_acc_data.tail(3), gyr_acc_data.head(3);

    double normAcc = gyr_acc_data.tail(3).norm();
    if (normAcc > maxNormAcc) maxNormAcc = normAcc;

    double normRotRate = gyr_acc_data.head(3).norm();
    if (normRotRate > maxNormRotRate) maxNormRotRate = normRotRate;

    //! Store imu measurements
    imu_measurements_.push_back(ImuMeasurement(timestamp, imu_accgyr));

    if (previous_timestamp != -1) {
      sumOfDelta += (timestamp - previous_timestamp);
      double deltaMismatch =
          std::fabs(static_cast<double>(
                        timestamp - previous_timestamp -
                        vio_params_.imu_params_.nominal_sampling_time_s_) *
                    1e-9);
      stdDelta += std::pow(deltaMismatch, 2);
      imu_rate_maxMismatch = std::max(imu_rate_maxMismatch, deltaMismatch);
      deltaCount += 1u;
    }
    previous_timestamp = timestamp;
  }

  // Converted to seconds.
  VLOG(10) << "IMU rate: "
           << (static_cast<double>(sumOfDelta) /
               static_cast<double>(deltaCount)) *
                  1e-9
           << '\n'
           << "IMU rate std: "
           << std::sqrt(stdDelta / static_cast<double>(deltaCount - 1u)) << '\n'
           << "IMU rate max mismatch: " << imu_rate_maxMismatch << '\n'
           << "Maximum measured rotation rate (norm):" << maxNormRotRate << '\n'
           << "Maximum measured acceleration (norm): " << maxNormAcc;
  fin.close();

  return true;
}

/* -------------------------------------------------------------------------- */
bool SpectacularLogDataProvider::parseDataset() {
  // Parse IMU data.
  ROBOT_CHECK(parseImuData(dataset_path_, kImuName));

  // Parse Camera data.
  CameraImageLists left_cam_image_list;
  CameraImageLists right_cam_image_list;
  parseCameraData(kLeftCamName, &left_cam_image_list);
  if (VLOG_IS_ON(1)) left_cam_image_list.print();
  parseCameraData(kRightCamName, &right_cam_image_list);
  if (VLOG_IS_ON(1)) right_cam_image_list.print();
  // TODO(Toni): remove camera_names_ and camera_image_lists_...
  camera_names_.push_back(kLeftCamName);
  camera_names_.push_back(kRightCamName);
  // WARNING Use [x] not .at() because we are adding entries that do not exist.
  camera_image_lists_[kLeftCamName] = left_cam_image_list;
  camera_image_lists_[kRightCamName] = right_cam_image_list;
  // ROBOT_CHECK(sanityCheckCameraData(camera_names_, &camera_image_lists_));

  // Parse Ground-Truth data.
  static const std::string ground_truth_name = "state_groundtruth_estimate0";
  is_gt_available_ = parseGtData(dataset_path_, ground_truth_name);

  clipFinalFrame();

  // Log Ground-Truth data.
  if (logger_) {
    if (is_gt_available_) {
      logger_->logGtData(dataset_path_ + "/mav0/" + ground_truth_name +
                         "/data.csv");
    } else {
      LOG(ERROR) << "Requested ground-truth data logging but no ground-truth "
                    "data available.";
    }
  }

  return true;
}

/* -------------------------------------------------------------------------- */
bool SpectacularLogDataProvider::parseCameraData(const std::string& cam_name,
                                        CameraImageLists* cam_list_i) {
  ROBOT_CHECK_NOTNULL(cam_list_i)
      ->parseCamImgList(dataset_path_ + "/mav0/" + cam_name, "data.csv");
  return true;
}

std::string SpectacularLogDataProvider::getDatasetName() {
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
size_t SpectacularLogDataProvider::getNumImages() const {
  ROBOT_CHECK_GT(camera_names_.size(), 0u);
  const std::string& left_cam_name = camera_names_.at(0);
  const std::string& right_cam_name = camera_names_.at(0);
  size_t n_left_images = getNumImagesForCamera(left_cam_name);
  size_t n_right_images = getNumImagesForCamera(right_cam_name);
  ROBOT_CHECK_EQ(n_left_images, n_right_images);
  return n_left_images;
}

/* -------------------------------------------------------------------------- */
size_t SpectacularLogDataProvider::getNumImagesForCamera(
    const std::string& camera_name) const {
  const auto& iter = camera_image_lists_.find(camera_name);
  ROBOT_CHECK(iter != camera_image_lists_.end());
  return iter->second.getNumImages();
}

/* -------------------------------------------------------------------------- */
bool SpectacularLogDataProvider::getImgName(const std::string& camera_name,
                                   const size_t& k,
                                   std::string* img_filename) const {
  ROBOT_CHECK_NOTNULL(img_filename);
  const auto& iter = camera_image_lists_.find(camera_name);
  ROBOT_CHECK(iter != camera_image_lists_.end());
  const auto& img_lists = iter->second.img_lists_;
  if (k < img_lists.size()) {
    *img_filename = img_lists.at(k).second;
    return true;
  } else {
    LOG(ERROR) << "Requested image #: " << k << " but we only have "
               << img_lists.size() << " images.";
  }
  return false;
}

/* -------------------------------------------------------------------------- */
Timestamp SpectacularLogDataProvider::timestampAtFrame(const FrameId& frame_number) {
  ROBOT_CHECK_GT(camera_names_.size(), 0);
  ROBOT_CHECK_LT(frame_number,
           camera_image_lists_.at(camera_names_[0]).img_lists_.size());
  return camera_image_lists_.at(camera_names_[0])
      .img_lists_[frame_number]
      .first;
}

void SpectacularLogDataProvider::clipFinalFrame() {
  // Clip final_k_ to the total number of images.
  const size_t& nr_images = getNumImages();
  if (final_k_ > nr_images) {
    LOG(WARNING) << "Value for final_k, " << final_k_ << " is larger than total"
                 << " number of frames in dataset " << nr_images;
    final_k_ = nr_images;
    LOG(WARNING) << "Using final_k = " << final_k_;
  }
  ROBOT_CHECK_LE(final_k_, nr_images);
}
/* -------------------------------------------------------------------------- */
void SpectacularLogDataProvider::print() const {
  LOG(INFO) << "------------------ ETHDatasetParser::print ------------------\n"
            << "Displaying info for dataset: " << dataset_path_;
  // For each of the 2 cameras.
  ROBOT_CHECK_EQ(vio_params_.camera_params_.size(), camera_names_.size());
  for (size_t i = 0; i < camera_names_.size(); i++) {
    LOG(INFO) << "\n"
              << (i == 0 ? "Left" : "Right")
              << " camera name: " << camera_names_[i] << ", with params:\n";
    vio_params_.camera_params_.at(i).print();
    camera_image_lists_.at(camera_names_[i]).print();
  }
  if (FLAGS_minloglevel < 1) {
    gt_data_.print();
  }
  LOG(INFO) << "-------------------------------------------------------------";
}

//////////////////////////////////////////////////////////////////////////////

/* -------------------------------------------------------------------------- */
MonoEurocDataProvider::MonoEurocDataProvider(const std::string& dataset_path,
                                             const int& initial_k,
                                             const int& final_k,
                                             const VioParams& vio_params)
    : SpectacularLogDataProvider(dataset_path, initial_k, final_k, vio_params) {}

/* -------------------------------------------------------------------------- */
MonoEurocDataProvider::MonoEurocDataProvider(const VioParams& vio_params)
    : SpectacularLogDataProvider(vio_params) {}

/* -------------------------------------------------------------------------- */
MonoEurocDataProvider::~MonoEurocDataProvider() {
  LOG(INFO) << "Mono ETHDataParser destructor called.";
}

/* -------------------------------------------------------------------------- */
bool MonoEurocDataProvider::spin() {
  if (dataset_parsed_) {
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
    // ROBOT_CHECK_EQ(vio_params_.camera_params_.size(), 2u);
    ROBOT_CHECK_GT(final_k_, initial_k_);
    // We log only the first one, because we may be running in sequential mode.
    LOG_FIRST_N(INFO, 1) << "Running dataset between frame " << initial_k_
                         << " and frame " << final_k_;
    while (!shutdown_ && spinOnce()) {
      if (!vio_params_.parallel_run_) {
        // Return, instead of blocking, when running in sequential mode.
        return true;
      }
    }
  } else {
    LOG(ERROR) << "Euroc dataset was not parsed.";
  }
  LOG_IF(INFO, shutdown_) << "SpectacularLogDataProvider shutdown requested.";
  return false;
}

/* -------------------------------------------------------------------------- */
bool MonoEurocDataProvider::spinOnce() {
  ROBOT_CHECK_LT(current_k_, std::numeric_limits<FrameId>::max())
      << "Are you sure you've initialized current_k_?";
  if (current_k_ >= final_k_) {
    LOG(INFO) << "Finished spinning Euroc dataset.";
    return false;
  }

  const CameraParams& left_cam_info = vio_params_.camera_params_.at(0);
  const bool& equalize_image =
      vio_params_.frontend_params_.stereo_matching_params_.equalize_image_;

  const Timestamp& timestamp_frame_k = timestampAtFrame(current_k_);
  VLOG(10) << "Sending left frame k= " << current_k_
           << " with timestamp: " << timestamp_frame_k;

  // TODO(Toni): ideally only send cv::Mat raw images...:
  // - pass params to vio_pipeline ctor
  // - make vio_pipeline actually equalize or transform images as necessary.
  std::string left_img_filename;
  bool available_left_img = getLeftImgName(current_k_, &left_img_filename);
  if (available_left_img) {
    // Both stereo images are available, send data to VIO
    ROBOT_CHECK(left_frame_callback_);
    left_frame_callback_(
        std::make_unique<Frame>(current_k_,
                                timestamp_frame_k,
                                // TODO(Toni): this info should be passed to
                                // the camera... not all the time here...
                                left_cam_info,
                                UtilsOpenCV::ReadAndConvertToGrayScale(
                                    left_img_filename, equalize_image)));
  } else {
    LOG(ERROR) << "Missing left image, proceeding to the next one.";
  }

  // This is done directly when parsing the Imu data.
  // imu_single_callback_(imu_meas);

  VLOG(10) << "Finished VIO processing for frame k = " << current_k_;
  current_k_++;
  return true;
}

}  // namespace
