#pragma once

#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/Pose3.h>

#include <map>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <string>
#include <vector>

#include "experimental/overhead_matching/spectacular_log.hh"
#include "kimera-vio/dataprovider/DataProviderInterface-definitions.h"
#include "kimera-vio/dataprovider/DataProviderInterface.h"
#include "kimera-vio/frontend/Frame.h"
#include "kimera-vio/frontend/StereoImuSyncPacket.h"
#include "kimera-vio/frontend/StereoMatchingParams.h"
#include "kimera-vio/logging/Logger.h"
#include "kimera-vio/utils/Macros.h"

namespace robot::experimental::overhead_matching {

/*
 * Parse all images and camera calibration for an ETH dataset.
 */
class SpectacularDataProviderInterface : public VIO::DataProviderInterface {
   public:
    KIMERA_DELETE_COPY_CONSTRUCTORS(SpectacularDataProviderInterface);
    KIMERA_POINTER_TYPEDEFS(SpectacularDataProviderInterface);
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    //! Ctor with params.
    SpectacularDataProviderInterface(const std::string& dataset_path, const int& initial_k,
                                     const int& final_k, const VIO::VioParams& vio_params);
    //! Ctor from gflags
    explicit SpectacularDataProviderInterface(const VIO::VioParams& vio_params);

    virtual ~SpectacularDataProviderInterface();

   public:
    /**
     * @brief spin Spins the dataset until it finishes. If set in sequential mode,
     * it will return each time a frame is sent. In parallel mode, it will not
     * return until it finishes.
     * @return True if the dataset still has data, false otherwise.
     */
    virtual bool spin() override;

    virtual bool hasData() const override;

    /**
     * @brief print Print info about dataset.
     */
    void print() const;

   public:
    inline std::string getDatasetPath() const { return dataset_path_; }
    std::string getDatasetName();

   protected:
    /**
     * @brief spinOnce Send data to VIO pipeline on a per-frame basis
     * @return if the dataset finished or not
     */
    virtual bool spinOnce();

    /**
     * @brief sendImuData We send IMU data first (before frames) so that the VIO
     * pipeline can query all IMU data between frames.
     */
    void sendImuData() const;


    //! Getters.
    /**
     * @brief getLeftImgName returns the img filename given the frame number
     * @param[in] k frame number
     * @param[out] img_name returned filename of the img
     * @return if k is larger than the number of frames, returns false, otw true.
     */
    inline bool getLeftImgName(const size_t& k, std::string* img_name) const {
        return getImgName("cam0", k, img_name);
    }
    size_t getNumImages() const;
    size_t getNumImagesForCamera(const std::string& camera_name) const;
    /**
     * @brief getImgName returns the img filename given the frame number
     * @param[in] camera_name camera id such as "cam0"/"cam1"
     * @param[in] k frame number
     * @param[out] img_filename returned filename of the img
     * @return if k is larger than the number of frames, returns false, otw true.
     */
    bool getImgName(const std::string& camera_name, const size_t& k,
                    std::string* img_filename) const;

    // Get timestamp of a given pair of stereo images (synchronized).
    VIO::Timestamp timestampAtFrame(const VIO::FrameId& frame_number);

    // Clip final frame to the number of images in the dataset.
    void clipFinalFrame();

   protected:
    VIO::VioParams vio_params_;

    /// Images data.
    // TODO(Toni): remove camera_names_ and camera_image_lists_...
    // This matches the names of the folders in the dataset
    std::vector<std::string> camera_names_;
    // Map from camera name to its images
    std::map<std::string, VIO::CameraImageLists> camera_image_lists_;

    bool is_gt_available_;
    std::string dataset_name_;
    std::string dataset_path_;

    VIO::FrameId current_k_;
    VIO::FrameId initial_k_;  // start frame
    VIO::FrameId final_k_;    // end frame

    //! Flag to signal if the IMU data has been sent to the VIO pipeline
    bool is_imu_data_sent_ = false;

    const std::string kLeftCamName = "cam0";
    const std::string kImuName = "imu0";

    //! Pre-stored imu-measurements
    std::vector<VIO::ImuMeasurement> imu_measurements_;

    SpectacularLog spec_log_;
};

}  // namespace robot::experimental::overhead_matching