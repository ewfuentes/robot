#pragma once

#include <filesystem>
#include <string>

#include "common/liegroups/se3.hh"

namespace robot::experimental::learn_descriptors {
struct FourSeasonsTransforms {
    struct StaticTransforms {
        liegroups::SE3 S_from_AS;
        liegroups::SE3 cam_from_imu;
        liegroups::SE3 w_from_gpsw;
        liegroups::SE3 gps_from_imu;
        liegroups::SE3 e_from_gpsw;
        double gnss_scale;  // scale from vio frame to gnss frame. WARNING: will require retooling
                            // if the scales per
        // keyframe (pose) are not all one value. See more here:
        // https://github.com/pmwenzel/4seasons-dataset

        StaticTransforms(const std::filesystem::path& path_transforms);

       private:
        static liegroups::SE3 get_transform_from_line(const std::string& line);
    };
    // might consider defining a nested "DynamicTransforms" or "PerImageTransforms" at some point -
    // Nico
};
}  // namespace robot::experimental::learn_descriptors