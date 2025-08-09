#include "common/gps/frame_translation.hh"

#include "GeographicLib/Constants.hpp"
#include "GeographicLib/Geocentric.hpp"
#include "GeographicLib/Geodesic.hpp"
#include "GeographicLib/LocalCartesian.hpp"

namespace robot::gps {
Eigen::Vector3d lla_from_ecef(const Eigen::Vector3d& t_place_from_ECEF) {
    static const GeographicLib::Geocentric earth = GeographicLib::Geocentric::WGS84();

    double x = t_place_from_ECEF.x(), y = t_place_from_ECEF.y(), z = t_place_from_ECEF.z();

    double lat_deg, lon_deg, alt_m;
    earth.Reverse(x, y, z, lat_deg, lon_deg, alt_m);

    return Eigen::Vector3d(lat_deg, lon_deg, alt_m);
}

Eigen::Vector3d ecef_from_lla(const Eigen::Vector3d& gcs_coordinate) {
    double lat_deg = gcs_coordinate.x();
    double lon_deg = gcs_coordinate.y();
    double alt_m = gcs_coordinate.z();

    double x, y, z;

    static const GeographicLib::Geocentric earth = GeographicLib::Geocentric::WGS84();
    earth.Forward(lat_deg, lon_deg, alt_m, x, y, z);

    return Eigen::Vector3d(x, y, z);
}
}  // namespace robot::gps