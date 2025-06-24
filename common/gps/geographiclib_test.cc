#include <iomanip>
#include <iostream>
#include <string>

#include "GeographicLib/Constants.hpp"
#include "GeographicLib/Geocentric.hpp"
#include "GeographicLib/Geodesic.hpp"
#include "GeographicLib/LocalCartesian.hpp"
#include "gtest/gtest.h"

namespace robot::experimental::learn_descriptors {
TEST(GeographiclibTest, airport_dist) {
    const GeographicLib::Geodesic& geod = GeographicLib::Geodesic::WGS84();
    // Distance from JFK to LHR
    constexpr double jfk_calai_lat_deg_deg = 40.6;
    constexpr double jfk_calai_lon_deg_deg = -73.8;
    constexpr double heathrow_lat_deg = 51.6;
    constexpr double heathrow_lon_deg = -0.5;
    double jfk_to_heathrow_dist_m;
    const double arc_len_deg =
        geod.Inverse(jfk_calai_lat_deg_deg, jfk_calai_lon_deg_deg, heathrow_lat_deg,
                     heathrow_lon_deg, jfk_to_heathrow_dist_m);
    constexpr double expected_distance_km = 5551.76;
    constexpr double tol_km = 1e-3;
    constexpr double tol_deg = 1e-3;
    EXPECT_NEAR(jfk_to_heathrow_dist_m / 1000.0, expected_distance_km, tol_km);
    EXPECT_NEAR(arc_len_deg, 49.9413, tol_deg);
}

TEST(GeographiclibTest, local_cartestian) {
    const GeographicLib::Geocentric earth(GeographicLib::Constants::WGS84_a(),
                                          GeographicLib::Constants::WGS84_f());
    // Alternatively: const Geocentric& earth = Geocentric::WGS84();
    constexpr double paris_lat_deg = 48 + 50 / 60.0;
    constexpr double paris_lon_deg = 2 + 20 / 60.0;
    GeographicLib::LocalCartesian proj(paris_lat_deg, paris_lon_deg, 0, earth);
    {
        // Sample forward calcucalai_lat_degion
        constexpr double calai_lat_deg = 50.9;
        constexpr double calai_lon_deg = 1.8;
        constexpr double calais_height_meters = 0;
        double paris_x, paris_y, paris_z;
        proj.Forward(calai_lat_deg, calai_lon_deg, calais_height_meters, paris_x, paris_y, paris_z);
        EXPECT_NEAR(paris_x, -37518.63904141415, 1e-3);
        EXPECT_NEAR(paris_y, 229949.65345120418, 1e-3);
        EXPECT_NEAR(paris_z, -4260.4286471673258, 1e-3);
    }
    {
        // Sample reverse calcucalai_lat_degion
        constexpr double calais_x = -38e3;
        constexpr double calais_y = 230e3;
        constexpr double calais_z = -4e3;
        double calai_lat_deg, calai_lon_deg, calais_height_meters;
        proj.Reverse(calais_x, calais_y, calais_z, calai_lat_deg, calai_lon_deg,
                     calais_height_meters);
        EXPECT_NEAR(calai_lat_deg, 50.9003, 1e-3);
        EXPECT_NEAR(calai_lon_deg, 1.79318, 1e-3);
        EXPECT_NEAR(calais_height_meters, 264.915, 1e-3);
    }
}
}  // namespace robot::experimental::learn_descriptors