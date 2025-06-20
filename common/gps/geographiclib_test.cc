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
    double lat1 = 40.6, lon1 = -73.8,  // JFK Airport
        lat2 = 51.6, lon2 = -0.5;      // LHR Airport
    double s12;
    geod.Inverse(lat1, lon1, lat2, lon2, s12);
    EXPECT_NEAR(s12 / 1000, 5551.76, 1e-3);
}

TEST(GeographiclibTest, local_cartestian) {
    const GeographicLib::Geocentric earth(GeographicLib::Constants::WGS84_a(),
                                          GeographicLib::Constants::WGS84_f());
    // Alternatively: const Geocentric& earth = Geocentric::WGS84();
    const double lat0 = 48 + 50 / 60.0, lon0 = 2 + 20 / 60.0;  // Paris
    GeographicLib::LocalCartesian proj(lat0, lon0, 0, earth);
    {
        // Sample forward calculation
        double lat = 50.9, lon = 1.8, h = 0;  // Calais
        double x, y, z;
        proj.Forward(lat, lon, h, x, y, z);
        EXPECT_NEAR(x, -37518.63904141415, 1e-3);
        EXPECT_NEAR(y, 229949.65345120418, 1e-3);
        EXPECT_NEAR(z, -4260.4286471673258, 1e-3);
    }
    {
        // Sample reverse calculation
        double x = -38e3, y = 230e3, z = -4e3;
        double lat, lon, h;
        proj.Reverse(x, y, z, lat, lon, h);
        EXPECT_NEAR(lat, 50.9003, 1e-3);
        EXPECT_NEAR(lon, 1.79318, 1e-3);
        EXPECT_NEAR(h, 264.915, 1e-3);
    }
}
}  // namespace robot::experimental::learn_descriptors