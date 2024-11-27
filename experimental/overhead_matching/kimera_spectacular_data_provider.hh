
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
#include "kimera-vio/common/VioNavState.h"
#include "kimera-vio/dataprovider/DataProviderInterface-definitions.h"
#include "kimera-vio/dataprovider/DataProviderInterface.h"
#include "kimera-vio/frontend/Frame.h"
#include "kimera-vio/frontend/StereoImuSyncPacket.h"
#include "kimera-vio/frontend/StereoMatchingParams.h"
#include "kimera-vio/logging/Logger.h"
#include "kimera-vio/utils/Macros.h"

namespace robot::experimental::overhead_matching {

class SpectacularLogDataProvider : public DataProviderInterface {
   public:
    KIMERA_DELETE_COPY_CONSTRUCTORS(SpectacularLogDataProvider);
    KIMERA_POINTER_TYPEDEFS(SpectacularLogDataProvider);
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    //! Ctor with params.
    SpectacularLogDataProvider(const std::string& dataset_path, const int& initial_k,
                               const int& final_k, const VioParams& vio_params);
    //! Ctor from gflags
    explicit SpectacularLogDataProvider(const VioParams& vio_params);

    virtual ~SpectacularLogDataProvider();

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
     * @brief parse Parses Euroc dataset. This is done already in spin() and
     * does not need to be called by the user. Left in public for experimentation.
     */
    void parse();

    /**
     * @brief sendImuData We send IMU data first (before frames) so that the VIO
     * pipeline can query all IMU data between frames.
     */
    void sendImuData() const;

    /**
     * @brief parseDataset Parse camera, gt, and imu data if using
     * different Euroc format.
     * @return
     */
    bool parseDataset();

    //! Parsers
    bool parseImuData(const std::string& input_dataset_path, const std::string& imu_name);

    bool parseGtData(const std::string& input_dataset_path, const std::string& gtSensorName);

    bool parseCameraData(const std::string& cam_name, CameraImageLists* cam_list_i);

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

   protected:

    VioParams vio_params_;
    SpectacularLog dataset_;
    
    std::string dataset_name_;
    std::string dataset_path_;

    FrameId current_k_;
    FrameId initial_k_;  // start frame
    FrameId final_k_;    // end frame

    //! Flag to signal when the dataset has been parsed.
    bool dataset_parsed_ = false;
    //! Flag to signal if the IMU data has been sent to the VIO pipeline
    bool is_imu_data_sent_ = false;

    const std::string kLeftCamName = "cam0";
    const std::string kImuName = "imu0";

    //! Pre-stored imu-measurements
    std::vector<ImuMeasurement> imu_measurements_;

    // EurocGtLogger::UniquePtr logger_;
};

}  // namespace 
