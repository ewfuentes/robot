#pragma once

#include <string>
#include <vector>

#include "experimental/overhead_matching/spectacular_log.hh"
#include "kimera-vio/dataprovider/DataProviderInterface.h"
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

    virtual ~SpectacularDataProviderInterface();

   public:
    /**
     * @brief spin Spins the dataset until it finishes. If set in sequential mode,
     * it will return each time a frame is sent. In parallel mode, it will not
     * return until it finishes.
     * @return True if the dataset still has data, false otherwise.
     */
    bool spin() override;

    bool hasData() const override;

    /**
     * @brief print Print info about dataset.
     */
    void print() const;

   public:
    inline std::string get_dataset_path() const { return dataset_path_; }

   protected:
    /**
     * @brief spin_once Send data to VIO pipeline on a per-frame basis
     * @return if the dataset finished or not
     */
    bool spin_once();

    /**
     * @brief sendImuData We send IMU data first (before frames) so that the VIO
     * pipeline can query all IMU data between frames.
     */
    void send_imu_data() const;

    size_t get_num_images() const;

    // Clip final frame to the number of images in the dataset.
    void clip_final_frame();

   protected:
    VIO::VioParams vio_params_;

    std::string dataset_name_;
    std::string dataset_path_;

    VIO::FrameId current_k_;
    VIO::FrameId initial_k_;  // start frame
    VIO::FrameId final_k_;    // end frame

    //! Flag to signal if the IMU data has been sent to the VIO pipeline
    bool is_imu_data_sent_ = false;

    SpectacularLog spec_log_;
};

}  // namespace robot::experimental::overhead_matching